"""
Phase 4, Step 5/6: Compute the final per-segment feature table.

One row per (video, garment, segment). Combines:

  1. Physics residual (from simulated_trajectories/, produced by
     run_simulation.py) -- mean/median pixel gap between simulated and
     observed positions, this segment only, VISIBLE frames only.
  2. Stretch/strain -- relative change in distance between neighboring
     tracked points (same k-NN edges as the simulator), compared to their
     distance at the segment's seed frame. Aggregated per-edge first, then
     across edges, so one noisy edge cannot dominate.
  3. Drape angle vs gravity -- for edges that genuinely HANG (within
     DRAPE_MAX_REST_ANGLE_DEG of vertical at rest), how far from straight
     down do they point, and how much does that wobble over time.
  4. Velocity/acceleration smoothness -- mean and spread of frame-to-frame
     acceleration magnitude per point.

Only genuinely VISIBLE frames are used for all features, and all temporal
derivatives are computed only within runs of strictly consecutive visible
frames (never across an occlusion gap).

---------------------------------------------------------------------------
REVISION HISTORY (see PHASE4_LOG.md for full context)
---------------------------------------------------------------------------
v2 fixes six defects found in the correctness audit:

  A. segment_residual() ignored entry["visible"], so interpolated positions
     from frames where CoTracker had LOST the point were being fed into the
     headline feature. Fakes plausibly track worse than reals, so this made
     residual partly a measure of TRACKING FAILURE rather than cloth physics
     -- a confound inside the project's central claim. Now visible-only, and
     visible_frac is reported so the confound can be tested directly.

  B. smooth_positions() was applied to visible-only entries that had NOT
     been split at occlusion gaps, despite its docstring claiming otherwise.
     A 3-frame window spanning frames [11, 12, 50] averaged positions 38
     frames apart, corrupting frame 12's position -- and the resulting bad
     velocity still passed the "consecutive frames" check. Positions are now
     split into strictly-consecutive runs BEFORE smoothing.

  C. segment_drape() averaged the angle of EVERY mesh edge, including
     horizontal ones. Horizontal edges read ~90 deg and vertical ones ~0 deg,
     so a k-NN grid always averaged to ~45 deg regardless of the video
     (measured: 42.61 real vs 42.71 fake, AUC 0.508 -- i.e. pure noise). It
     was measuring MESH TOPOLOGY, not drape. Now only edges that hang within
     DRAPE_MAX_REST_ANGLE_DEG of vertical at rest are considered.

  D. drape was missing on 36% of rows because free_edges came up empty
     whenever every point sat within DRAPE_ANCHOR_THRESHOLD of some joint
     (common for small/distant subjects on a dense point grid). The anchor
     filter is now a PREFERENCE with a documented fallback, not a hard gate.

  E. build_sim_lookup() merged every segment of a garment into one
     frame-keyed dict. Re-seeded segments restart point_id at 0, so where two
     segments shared a frame the later one silently overwrote the earlier
     one's points. Lookups are now built per-segment, keyed by seed_frame.

  F. rest_positions were taken from trajectory[0] with no visibility check.
     An occluded seed frame produced a garbage rest state and garbage spring
     rest lengths for the entire segment. Seed validity is now checked and
     reported via seed_visible_frac.

Not fixed here (requires re-running the simulation -- bundled with the
physics-constant calibration step):

  G. find_anchors() in run_simulation.py records a fixed offset from the
     nearest joint, so anchored cloth TRANSLATES with a limb but never
     ROTATES around it. Cloth on a rotating forearm is therefore simulated
     wrongly on real videos too, inflating every residual and compressing
     the real-vs-fake gap.

Usage:
  python compute_features.py --trajectories trajectories --simulated simulated_trajectories --pose pose --output phase4_features.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from cloth_simulator import build_knn_springs
from run_simulation import (get_joint_positions, load_pose, find_anchors, select_anchors,
                            video_torso_scale, load_material_fits)

LABELS = ("real", "fake")
GARMENTS = ("upper", "lower")
FPS = 30.0

ANCHOR_THRESHOLD = 60.0          # matches run_simulation.py -- generous, keeps physics stable
DRAPE_ANCHOR_THRESHOLD = 25.0    # stricter; used only as a PREFERENCE when picking drape edges
ANCHOR_FRACTION = 0.20           # anchor the closest 30% of points to joints, BY RANK, rather
                                 # than by a pixel threshold. A distance threshold made the
                                 # anchored fraction depend on how close the tracked points
                                 # happened to sit to MediaPipe's joints, which differs
                                 # systematically by label: at 60px, 0.0% of points were free on
                                 # real videos versus 8.0% on fakes. Real and fake were therefore
                                 # having different computations performed on them. Ranking makes
                                 # the anchor/free split identical for every segment.
DRAPE_MAX_REST_ANGLE_DEG = 45.0  # an edge must hang within this of vertical AT REST to count as
                                 # a draping edge. Without this, horizontal mesh edges (~90 deg)
                                 # and vertical ones (~0 deg) average to ~45 deg for every video.
K_NEIGHBORS = 4
SMOOTH_WINDOW = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-segment physics + hand-crafted features.")
    parser.add_argument("--trajectories", type=Path, default=Path("trajectories"))
    parser.add_argument("--simulated", type=Path, default=Path("simulated_trajectories"))
    parser.add_argument("--pose", type=Path, default=Path("pose"))
    parser.add_argument("--output", type=Path, default=Path("phase4_features.csv"))
    parser.add_argument("--material-fits", type=Path, default=None,
                        help="JSON from fit_material.py. Adds the per-video fitted material "
                             "columns and marks which segments were used for fitting, so the "
                             "analysis can be restricted to held-out segments.")
    return parser.parse_args()


def discover_videos(trajectories_root: Path):
    videos = []
    for label in LABELS:
        label_dir = trajectories_root / label
        if not label_dir.is_dir():
            continue
        for json_path in sorted(label_dir.glob("*.json")):
            videos.append((label, json_path.stem))
    return videos


# ---------------------------------------------------------------------------
# Simulated-trajectory lookup  (fix E)
# ---------------------------------------------------------------------------

def build_sim_lookups(sim_garment_data):
    """
    Build ONE lookup PER SEGMENT, keyed by that segment's seed_frame.

    The previous version merged all segments into a single {frame: {point_id: xy}}
    dict. Because re-seeding restarts point_id at 0 in every segment, any frame
    covered by two segments had its point 0 (1, 2, ...) silently overwritten by
    whichever segment was written last -- so residuals near segment boundaries
    compared observed points against a DIFFERENT point's simulated position.
    """
    lookups = {}
    if not sim_garment_data:
        return lookups
    for segment in sim_garment_data.get("segments", []):
        lookup = {}
        for point in segment["points"]:
            for entry in point["trajectory"]:
                lookup.setdefault(entry["frame"], {})[point["point_id"]] = (entry["x"], entry["y"])
        lookups[segment["seed_frame"]] = lookup
    return lookups


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def split_into_runs(entries):
    """
    Split a list of frame-ordered entries into runs of STRICTLY CONSECUTIVE
    frames. Everything temporal (smoothing, velocity, acceleration, strain
    rate) must operate inside a single run -- crossing an occlusion gap
    invents motion that never happened.
    """
    runs, current = [], []
    for entry in entries:
        if current and entry["frame"] - current[-1]["frame"] != 1:
            runs.append(current)
            current = []
        current.append(entry)
    if current:
        runs.append(current)
    return runs


def visible_entries(point):
    return [e for e in point["trajectory"] if e.get("visible")]


def positions_by_frame(points):
    """{point_id: {frame: (x, y, visible)}}"""
    table = {}
    for point in points:
        table[point["point_id"]] = {
            e["frame"]: (e["x"], e["y"], bool(e.get("visible")))
            for e in point["trajectory"]
        }
    return table


# ---------------------------------------------------------------------------
# Feature 1: physics residual  (fix A)
# ---------------------------------------------------------------------------

def segment_residual(segment, sim_lookup):
    """
    Mean and median pixel gap between simulated and observed positions.

    VISIBLE FRAMES ONLY. When CoTracker loses a point it still emits an
    interpolated position with visible=False; those are guesses, not
    observations, and including them made the residual partly a measure of
    how badly the tracker struggled rather than how badly the physics matched.
    """
    gaps = []
    for point in segment["points"]:
        for entry in point["trajectory"]:
            frame = entry["frame"]
            if frame not in sim_lookup or point["point_id"] not in sim_lookup[frame]:
                continue
            if not entry.get("visible"):
                continue
            sx, sy = sim_lookup[frame][point["point_id"]]
            gaps.append(math.hypot(entry["x"] - sx, entry["y"] - sy))

    if not gaps:
        return {"residual_mean_px": None, "residual_median_px": None}

    arr = np.asarray(gaps, dtype=float)
    return {
        "residual_mean_px": float(arr.mean()),
        # Median is robust to a single badly-tracked point dragging the whole
        # segment's residual up; worth comparing against the mean.
        "residual_median_px": float(np.median(arr)),
    }


# ---------------------------------------------------------------------------
# Feature 2: stretch / strain
# ---------------------------------------------------------------------------

def segment_stretch(points, springs):
    """
    Per-edge strain statistics, aggregated across edges.

    The previous version pooled every (edge, frame) strain value into one flat
    list, so stretch_std conflated variation BETWEEN edges with variation OVER
    TIME, and a single badly-tracked edge could dominate the whole segment.
    Here each edge is summarised first, then edges are averaged.

    strain_rate_mean is new: |d(strain)/dt| within consecutive visible frames.
    Real fabric changes length slowly and smoothly; a generator redrawing cloth
    frame by frame has no constraint forcing that, so the RATE of length change
    is physically more discriminative than the amount.
    """
    table = positions_by_frame(points)

    per_edge_mean_abs, per_edge_max_abs = [], []
    pooled_strains, strain_rates = [], []

    for i, j, rest_len in springs:
        if rest_len is None or rest_len < 1e-6:
            continue
        if i not in table or j not in table:
            continue

        shared_frames = sorted(set(table[i]) & set(table[j]))
        series = []
        for frame in shared_frames:
            xi, yi, vis_i = table[i][frame]
            xj, yj, vis_j = table[j][frame]
            if not (vis_i and vis_j):
                continue
            dist = math.hypot(xi - xj, yi - yj)
            series.append({"frame": frame, "strain": (dist - rest_len) / rest_len})

        if not series:
            continue

        strains = np.asarray([s["strain"] for s in series], dtype=float)
        per_edge_mean_abs.append(float(np.mean(np.abs(strains))))
        per_edge_max_abs.append(float(np.max(np.abs(strains))))
        pooled_strains.extend(strains.tolist())

        for run in split_into_runs(series):
            for a, b in zip(run[:-1], run[1:]):
                strain_rates.append(abs(b["strain"] - a["strain"]) * FPS)

    if not per_edge_mean_abs:
        return {
            "stretch_mean_abs": None, "stretch_std": None,
            "stretch_max_abs": None, "strain_rate_mean": None,
        }

    return {
        "stretch_mean_abs": float(np.mean(per_edge_mean_abs)),
        # Kept as a pooled std so the column stays comparable with the v1 CSV.
        "stretch_std": float(np.std(pooled_strains, ddof=1)) if len(pooled_strains) > 1 else 0.0,
        "stretch_max_abs": float(np.max(per_edge_max_abs)),
        "strain_rate_mean": float(np.mean(strain_rates)) if strain_rates else None,
    }


# ---------------------------------------------------------------------------
# Feature 3: drape angle vs gravity  (fixes C and D)
# ---------------------------------------------------------------------------

def select_hanging_edges(springs, rest_positions, anchor_mask):
    """
    Pick the edges that actually represent HANGING cloth.

    Two filters, in order of importance:

      1. Rest orientation. An edge must already point within
         DRAPE_MAX_REST_ANGLE_DEG of straight down at the seed frame. This is
         the fix for the bug that made drape useless: a k-NN grid contains
         roughly as many horizontal edges (~90 deg from vertical) as vertical
         ones (~0 deg), so averaging over ALL of them returned ~45 deg for
         every video, real or fake, forever.

      2. Anchoring, as a PREFERENCE not a gate. An edge with both ends pinned
         to the skeleton mirrors body motion rather than cloth behaviour, so we
         drop those when we can. But on a dense point grid with a small subject
         EVERY point can fall within the anchor threshold, which previously
         produced an empty set and a missing feature on 36% of rows. If the
         preferred set is empty we fall back to all hanging edges rather than
         returning nothing.
    """
    preferred, fallback = [], []

    for i, j, _ in springs:
        upper, lower = (i, j) if rest_positions[i][1] < rest_positions[j][1] else (j, i)
        dx = rest_positions[lower][0] - rest_positions[upper][0]
        dy = rest_positions[lower][1] - rest_positions[upper][1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue

        rest_angle = math.degrees(math.acos(max(-1.0, min(1.0, dy / length))))
        if rest_angle > DRAPE_MAX_REST_ANGLE_DEG:
            continue  # horizontal-ish mesh edge: structural, not a hanging chain

        fallback.append((upper, lower))
        if not (anchor_mask[upper] and anchor_mask[lower]):
            preferred.append((upper, lower))

    if preferred:
        return preferred, "free"
    return fallback, "fallback_all_anchored"


def segment_drape(points, springs, rest_positions, anchor_mask):
    """
    How far from straight-down do hanging edges point, and how much does that
    direction wobble?

    drape_mean_angle_deg - mean deviation from vertical. Cloth hanging freely
                           under gravity should sit near 0; cloth being swung
                           by the body deviates, but it should do so smoothly.
    drape_angle_std_deg  - how much the hanging direction varies over the
                           segment. This may carry more signal than the mean:
                           real cloth swings through angles continuously, while
                           a generator can snap it between orientations.
    """
    table = positions_by_frame(points)
    edges, edge_source = select_hanging_edges(springs, rest_positions, anchor_mask)

    if not edges:
        return {
            "drape_mean_angle_deg": None, "drape_angle_std_deg": None,
            "gravity_p95_angle_deg": None, "gravity_inversion_rate": None,
            "drape_edge_count": 0, "drape_edge_source": "none",
        }

    angles = []
    for upper, lower in edges:
        if upper not in table or lower not in table:
            continue
        for frame in sorted(set(table[upper]) & set(table[lower])):
            xu, yu, vis_u = table[upper][frame]
            xl, yl, vis_l = table[lower][frame]
            if not (vis_u and vis_l):
                continue
            dx, dy = xl - xu, yl - yu
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, dy / length)))))

    if not angles:
        return {
            "drape_mean_angle_deg": None, "drape_angle_std_deg": None,
            "gravity_p95_angle_deg": None, "gravity_inversion_rate": None,
            "drape_edge_count": len(edges), "drape_edge_source": edge_source,
        }

    arr = np.asarray(angles, dtype=float)
    return {
        "drape_mean_angle_deg": float(arr.mean()),
        "drape_angle_std_deg": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        # --- material-independent gravity law ---------------------------------
        # "Loose fabric hangs down, not up." True for silk and denim alike, so it
        # needs no stiffness, no damping and no simulation -- only the tracked
        # motion. Both measures below are threshold-free or physically anchored:
        #
        #   gravity_p95_angle_deg   How far from straight down the hanging fabric
        #                           gets at its most extreme moments (95th
        #                           percentile of the hang angle). No tuning.
        #   gravity_inversion_rate  Fraction of samples where an edge that HUNG
        #                           DOWNWARD at rest now points UPWARD (angle past
        #                           90 deg, i.e. the lower point has risen above
        #                           the upper one). 90 deg is not a chosen number:
        #                           it is literally "pointing up". Gravity makes
        #                           this rare and brief for real cloth.
        #
        # Deliberately NOT used: correlating sway angle with angular acceleration
        # ("does gravity pull it back?"). For any bounded motion that correlation
        # is negative by a mathematical identity (integration by parts), so it
        # would pass on fake video just as readily. It tests nothing.
        "gravity_p95_angle_deg": float(np.percentile(arr, 95)),
        "gravity_inversion_rate": float(np.mean(arr > 90.0)),
        "drape_edge_count": len(edges),
        "drape_edge_source": edge_source,
    }


# ---------------------------------------------------------------------------
# Feature 4: velocity / acceleration smoothness  (fix B)
# ---------------------------------------------------------------------------

def smooth_run(run, window=SMOOTH_WINDOW):
    """
    Moving-average smoothing over ONE run of strictly consecutive frames.

    Acceleration is a second derivative, so a few pixels of ordinary tracking
    jitter gets divided by dt twice and amplified into meaningless spikes.
    Smoothing first is the standard fix.

    The caller MUST pass a single consecutive run. The previous version was
    handed visible-only entries that still contained occlusion gaps, so a
    3-frame window over frames [11, 12, 50] averaged positions 38 frames apart
    and corrupted frame 12 -- and the corrupted velocity from 11 to 12 then
    passed the consecutive-frame check and was kept.
    """
    if len(run) < window:
        return [{"frame": e["frame"], "x": e["x"], "y": e["y"]} for e in run]

    half = window // 2
    smoothed = []
    for idx in range(len(run)):
        lo, hi = max(0, idx - half), min(len(run), idx + half + 1)
        chunk = run[lo:hi]
        smoothed.append({
            "frame": run[idx]["frame"],
            "x": sum(e["x"] for e in chunk) / len(chunk),
            "y": sum(e["y"] for e in chunk) / len(chunk),
        })
    return smoothed


def segment_velocity_smoothness(points):
    accel_magnitudes = []
    dt = 1.0 / FPS

    for point in points:
        for run in split_into_runs(visible_entries(point)):
            if len(run) < 3:
                continue  # need 3 positions for one acceleration value
            smoothed = smooth_run(run)

            velocities = []
            for a, b in zip(smoothed[:-1], smoothed[1:]):
                velocities.append(((b["x"] - a["x"]) / dt, (b["y"] - a["y"]) / dt))

            for v1, v2 in zip(velocities[:-1], velocities[1:]):
                ax = (v2[0] - v1[0]) / dt
                ay = (v2[1] - v1[1]) / dt
                accel_magnitudes.append(math.hypot(ax, ay))

    if not accel_magnitudes:
        return {"accel_mean_magnitude": None, "accel_std_magnitude": None}

    arr = np.asarray(accel_magnitudes, dtype=float)
    return {
        "accel_mean_magnitude": float(arr.mean()),
        "accel_std_magnitude": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def segment_diagnostics(points):
    """
    Not physics features -- these exist to TEST the tracking-quality confound.

    If visible_frac separates real from fake as well as the physics features
    do, then the physics features may simply be riding on the fact that fakes
    track worse. That would need addressing before any physics claim stands.
    """
    total = visible = 0
    for point in points:
        for entry in point["trajectory"]:
            total += 1
            if entry.get("visible"):
                visible += 1

    seed_visible = sum(1 for p in points if p["trajectory"] and p["trajectory"][0].get("visible"))

    return {
        "visible_frac": (visible / total) if total else None,
        # fix F: an occluded seed frame poisons rest_positions and every spring
        # rest length for the whole segment. Surfaced rather than silently used.
        "seed_visible_frac": (seed_visible / len(points)) if points else None,
    }


# ---------------------------------------------------------------------------
# Per-segment driver
# ---------------------------------------------------------------------------

def segment_body_scale(pose_data, frame_numbers):
    """
    How large is the person in the frame, in pixels? Measured as the diagonal
    of the skeleton's bounding box, taken as the MEDIAN across the segment so
    one bad pose estimate cannot skew it.

    This exists because of a scale confound found in the dataset: the subjects
    in the fake videos are roughly twice as large in frame as those in the real
    videos (skeleton spans of ~453px versus ~229px). EVERY quantity measured in
    pixels therefore came out about 2x larger on fakes -- residual, pose
    distance and acceleration all separated by almost exactly the same factor,
    while every DIMENSIONLESS feature (strain ratio, drape angle, strain rate)
    separated by essentially nothing. That pattern is the signature of camera
    framing, not cloth physics.

    Dividing pixel quantities by this scale makes them comparable between a
    person filling the frame and a person standing far away.
    """
    sizes = []
    for frame in frame_numbers:
        joints = get_joint_positions(pose_data, frame)
        if joints is None:
            continue
        sizes.append(float(np.linalg.norm(joints.max(axis=0) - joints.min(axis=0))))
    if not sizes:
        return None
    scale = float(np.median(sizes))
    return scale if scale > 1e-6 else None


def pose_distance_stats(rest_positions, joint_positions, nearest_dist):
    """
    How far do the tracked garment points sit from MediaPipe's detected joints?

    This is the POSE-quality counterpart to visible_frac (which only covers
    CoTracker). It exists because the anchor audit showed the free-point
    fraction differing sharply by label -- 0.0% free on real versus 8.0% on
    fake at a 60px threshold -- which implies the tracked points sit further
    from detected joints on fakes. If MediaPipe simply localises poses worse on
    AI-generated video, the anchor targets are wrong there, and the residual
    would partly measure POSE ERROR rather than cloth physics.

    pose_dist_norm divides by the skeleton's own size, so a subject who is
    simply closer to the camera does not register as a pose problem.
    """
    if len(nearest_dist) == 0:
        return {"pose_dist_mean_px": None, "pose_dist_norm": None}

    mean_px = float(np.mean(nearest_dist))

    spread = float(np.linalg.norm(joint_positions.max(axis=0) - joint_positions.min(axis=0)))
    return {
        "pose_dist_mean_px": mean_px,
        "pose_dist_norm": (mean_px / spread) if spread > 1e-6 else None,
    }


def process_segment(segment, sim_lookup, pose_data, video_scale=None, material=None):
    points = segment["points"]
    seed_frame = segment["seed_frame"]

    rest_positions = np.array(
        [[p["trajectory"][0]["x"], p["trajectory"][0]["y"]] for p in points],
        dtype=float,
    )

    springs = build_knn_springs(rest_positions, k=K_NEIGHBORS)

    joint_positions = get_joint_positions(pose_data, seed_frame) if pose_data else None
    if joint_positions is not None:
        # Same rank-based rule the simulator uses, so "anchored" means the same
        # thing in the simulation and in the drape feature.
        drape_anchor_mask, _, nearest_dist = select_anchors(
            rest_positions, joint_positions, fraction=ANCHOR_FRACTION
        )
        pose_stats = pose_distance_stats(rest_positions, joint_positions, nearest_dist)
    else:
        drape_anchor_mask = np.zeros(len(points), dtype=bool)
        pose_stats = {"pose_dist_mean_px": None, "pose_dist_norm": None}

    # Torso length for the whole video is the chosen yardstick: unlike a
    # bounding box it does not grow when the person raises their arms, and the
    # per-video median removes pose-to-pose wobble. Falls back to the
    # bounding-box diagonal only if torso landmarks are unavailable.
    body_scale = video_scale
    if body_scale is None and pose_data:
        frame_numbers = [e["frame"] for e in points[0]["trajectory"]] if points else []
        body_scale = segment_body_scale(pose_data, frame_numbers)

    residual = segment_residual(segment, sim_lookup)
    stretch = segment_stretch(points, springs)
    drape = segment_drape(points, springs, rest_positions, drape_anchor_mask)
    smoothness = segment_velocity_smoothness(points)
    diagnostics = segment_diagnostics(points)

    row = {"seed_frame": seed_frame, "num_points": len(points)}
    row.update(diagnostics)
    row.update(residual)
    row.update(stretch)
    row.update(drape)
    row.update(smoothness)
    row.update(pose_stats)

    # Scale-invariant versions of every pixel-valued feature. These, not the
    # raw pixel columns, are what any real physics claim must rest on.
    scaled = {"body_scale_px": body_scale}
    pairs = (
        ("residual_mean_px", "residual_mean_norm"),
        ("residual_median_px", "residual_median_norm"),
        ("accel_mean_magnitude", "accel_mean_norm"),
        ("accel_std_magnitude", "accel_std_norm"),
    )
    source = {**residual, **smoothness}
    for src, dst in pairs:
        value = source.get(src)
        scaled[dst] = (value / body_scale) if (value is not None and body_scale) else None
    row.update(scaled)

    # Per-video fitted material (fit_material.py). fit_error_eval is measured on
    # segments the material was NOT fitted on, so it is an honest number; the
    # in_fit_set flag lets the analysis drop the fitted segments entirely.
    if material:
        row.update({
            "fitted_stiffness": material.get("fitted_stiffness"),
            "fitted_damping": material.get("fitted_damping"),
            "fit_skill_eval": material.get("fit_skill_eval"),
            "fit_rigid_error": material.get("fit_rigid_error"),
            "fit_physics_error": material.get("fit_physics_error"),
            "fit_contrast": material.get("fit_contrast"),
            "fit_at_boundary": int(bool(material.get("fit_at_boundary"))),
            "in_fit_set": int(seed_frame in (material.get("fit_seed_frames") or [])),
        })
    else:
        row.update({
            "fitted_stiffness": None, "fitted_damping": None,
            "fit_skill_eval": None, "fit_rigid_error": None,
            "fit_physics_error": None, "fit_contrast": None,
            "fit_at_boundary": None, "in_fit_set": 0,
        })
    return row


FIELDNAMES = [
    "video", "label", "garment", "seed_frame", "num_points",
    "visible_frac", "seed_visible_frac", "pose_dist_mean_px", "pose_dist_norm",
    "body_scale_px",
    "residual_mean_px", "residual_median_px",
    "residual_mean_norm", "residual_median_norm",
    "stretch_mean_abs", "stretch_std", "stretch_max_abs", "strain_rate_mean",
    "drape_mean_angle_deg", "drape_angle_std_deg",
    "gravity_p95_angle_deg", "gravity_inversion_rate",
    "drape_edge_count", "drape_edge_source",
    "accel_mean_magnitude", "accel_std_magnitude",
    "accel_mean_norm", "accel_std_norm",
    "fitted_stiffness", "fitted_damping", "fit_skill_eval",
    "fit_rigid_error", "fit_physics_error", "fit_contrast",
    "fit_at_boundary", "in_fit_set",
]


def main():
    args = parse_args()

    material_fits = load_material_fits(args.material_fits)
    if material_fits:
        print(f"Using per-video materials from {args.material_fits}")

    videos = discover_videos(args.trajectories)
    if not videos:
        raise SystemExit(f"ERROR: No videos found under {args.trajectories}")

    rows = []
    missing_sim = []
    print(f"Videos to process: {len(videos)}")
    print("-" * 60)

    for label, video_name in videos:
        traj_path = args.trajectories / label / f"{video_name}.json"
        sim_path = args.simulated / label / f"{video_name}.json"

        with traj_path.open(encoding="utf-8") as f:
            traj_data = json.load(f)

        if sim_path.is_file():
            with sim_path.open(encoding="utf-8") as f:
                sim_data = json.load(f)
        else:
            sim_data = {}
            missing_sim.append(f"{label}/{video_name}")

        pose_data = load_pose(args.pose, label, video_name)
        video_scale = video_torso_scale(pose_data) if pose_data else None
        fits_for_video = material_fits.get(f"{label}/{video_name}", {}).get("garments", {})

        print(f"\n{label}/{video_name}")
        if pose_data is None:
            print("  WARNING: no pose data -- drape anchor preference disabled for this video")

        for garment in GARMENTS:
            if garment not in traj_data:
                continue

            sim_lookups = build_sim_lookups(sim_data.get(garment))

            unmatched = 0
            for segment in traj_data[garment]["segments"]:
                seed_frame = segment["seed_frame"]
                sim_lookup = sim_lookups.get(seed_frame, {})
                if not sim_lookup:
                    unmatched += 1
                rows.append({
                    "video": video_name,
                    "label": label,
                    "garment": garment,
                    **process_segment(segment, sim_lookup, pose_data,
                                      video_scale=video_scale,
                                      material=fits_for_video.get(garment)),
                })

            note = f"  {garment}: {len(traj_data[garment]['segments'])} segment(s) processed"
            if unmatched:
                note += f"  ({unmatched} with no matching simulation -- residual will be blank)"
            print(note)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} row(s) to: {args.output.resolve()}")
    if missing_sim:
        print(f"\nWARNING: no simulated trajectories for {len(missing_sim)} video(s):")
        for name in missing_sim:
            print(f"  - {name}")
        print("Run run_simulation.py --all first if this was not intended.")


if __name__ == "__main__":
    main()