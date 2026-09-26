"""
Phase 4, Step 2: Run the cloth simulator driven by real tracked skeleton
motion, per re-seed segment, and save the simulated trajectories in the same
format as Phase 3's observed trajectories -- so they can be directly compared.

For each segment:
  1. Take the tracked points' positions at the segment's first frame (seed
     frame) as the simulation's rest state.
  2. Build a spring mesh connecting each point to its ~k nearest neighbors.
  3. Find "anchor" points -- any tracked point within --anchor-threshold
     pixels of a real skeleton joint at the seed frame. These get driven
     directly by that joint's real motion each frame, rather than being
     freely simulated. (Points with no nearby joint are simulated freely.)
  4. Step the simulation forward frame by frame across the segment, at 30fps
     (matching Phase 1's extraction rate), re-pinning each anchor's position
     from the real skeleton before each step.

Output structure matches trajectories/ exactly:
{
  "upper": {"segments": [{"seed_frame": 0, "points": [{"point_id": 0, "trajectory": [...]}]}]},
  "lower": {...}
}
Each trajectory entry has "frame", "x", "y" (no "visible" flag -- these are
simulated, not observed).

---------------------------------------------------------------------------
v2 change: rotation-aware anchors (audit bug G)
---------------------------------------------------------------------------
The original anchor model stored a FIXED offset from the nearest joint:

    anchor_offset[i] = pos - joint_positions[nearest]
    target           = joint_positions[nearest] + anchor_offset[i]

so an anchored cloth point TRANSLATED with its joint but never ROTATED around
it. When a forearm swings through 90 degrees, cloth on that forearm should
swing with it; instead it slid along keeping its original orientation. That is
wrong on REAL videos too, so it inflated every residual and compressed the
real-vs-fake gap we are trying to measure.

The fix builds a local coordinate frame per anchor from TWO joints:

    u = unit(joint_b - joint_a)        along the limb
    n = perpendicular(u)               across the limb
    offset = a_coeff * u + b_coeff * n

The coefficients are measured once at the seed frame and re-applied each frame
using that frame's u and n, so the cloth rotates with the limb. The frame also
carries a scale factor (current limb length / seed limb length) so the cloth
follows the body as the subject moves toward or away from the camera.

Use --anchor-mode translate to restore the old behaviour for A/B comparison.

Usage:
  python run_simulation.py --all
  python run_simulation.py --all --anchor-mode translate --output simulated_trajectories_translate
  python run_simulation.py --video "real/Man boxing" --gravity 800 --stiffness 40 --damping 6
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cloth_simulator import ClothSimulator, build_knn_springs

LABELS = ("real", "fake")
GARMENTS = ("upper", "lower")
FPS = 30.0

# A local frame is only meaningful if the two joints defining it are far
# enough apart; otherwise tiny MediaPipe jitter produces wild rotations.
MIN_FRAME_LENGTH_PX = 15.0
# Limb-length ratios outside this range are almost certainly pose error, not
# genuine motion toward/away from the camera.
SCALE_LIMITS = (0.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cloth simulation driven by real skeleton motion.")
    parser.add_argument("--trajectories", type=Path, default=Path("trajectories"))
    parser.add_argument("--pose", type=Path, default=Path("pose"))
    parser.add_argument("--output", type=Path, default=Path("simulated_trajectories"))
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--k", type=int, default=4, help="Nearest neighbors per point for spring mesh (default: 4).")
    parser.add_argument("--anchor-threshold", type=float, default=60.0,
                        help="Max pixel distance to a skeleton joint to count as an anchor (default: 60).")
    parser.add_argument("--anchor-fraction", type=float, default=0.20,
                        help="Anchor the closest this-fraction of points to joints, by rank "
                             "(default: 0.30). Guarantees the same anchor/free split on every "
                             "segment, so real and fake are compared like with like. Pass a "
                             "negative value to fall back to the --anchor-threshold rule.")
    parser.add_argument("--anchor-mode", choices=("frame", "translate"), default="frame",
                        help="'frame' rotates cloth with the limb (default). 'translate' is the old "
                             "fixed-offset behaviour, kept for A/B comparison.")
    parser.add_argument("--gravity", type=float, default=800.0)
    parser.add_argument("--stiffness", type=float, default=40.0)
    parser.add_argument("--damping", type=float, default=6.0)
    parser.add_argument("--material-fits", type=Path, default=None,
                        help="JSON from fit_material.py. When given, each video/garment uses "
                             "ITS OWN fitted stiffness and damping instead of the global ones, "
                             "because those are material properties rather than constants.")
    return parser.parse_args()


def discover_videos(trajectories_root: Path) -> list[tuple[str, str]]:
    videos = []
    for label in LABELS:
        label_dir = trajectories_root / label
        if not label_dir.is_dir():
            continue
        for json_path in sorted(label_dir.glob("*.json")):
            videos.append((label, json_path.stem))
    return videos


def load_pose(pose_root: Path, label: str, video_name: str) -> dict | None:
    path = pose_root / label / video_name / "keypoints.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_joint_positions(pose_data: dict, frame_idx: int) -> np.ndarray | None:
    """Returns (num_joints, 2) array of joint pixel positions for a frame, or None."""
    entry = pose_data.get(f"{frame_idx:04d}")
    if not entry:
        return None
    return np.array([[lm["x_px"], lm["y_px"]] for lm in entry], dtype=np.float64)


# MediaPipe Pose landmark indices for the torso.
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24


def video_torso_scale(pose_data: dict) -> float | None:
    """
    Median shoulder-midpoint to hip-midpoint distance across the whole video.

    The yardstick for turning pixel measurements into scale-free ones. The
    dataset has a framing confound -- subjects in the fake videos are roughly
    twice the size in frame as those in the real ones -- which made every
    pixel-valued feature separate by ~2x for reasons unrelated to cloth.

    Torso length beats a bounding box because it does not grow when the person
    raises their arms, and the per-video median removes pose-to-pose wobble.
    """
    import statistics as _stats
    lengths = []
    for key in pose_data:
        entry = pose_data.get(key)
        if not entry or len(entry) <= max(L_HIP, R_HIP):
            continue
        joints = np.array([[lm["x_px"], lm["y_px"]] for lm in entry], dtype=float)
        shoulder = (joints[L_SHOULDER] + joints[R_SHOULDER]) / 2.0
        hip = (joints[L_HIP] + joints[R_HIP]) / 2.0
        length = float(np.linalg.norm(shoulder - hip))
        if length > 1e-6:
            lengths.append(length)
    return float(_stats.median(lengths)) if lengths else None


def load_material_fits(path: Path | None) -> dict:
    """Load per-video fitted materials produced by fit_material.py."""
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"ERROR: material fits file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f).get("fits", {})


def material_for(fits: dict, label: str, video: str, garment: str,
                 default_stiffness: float, default_damping: float):
    """(stiffness, damping, source) for one video and garment."""
    entry = fits.get(f"{label}/{video}", {}).get("garments", {}).get(garment)
    if not entry:
        return default_stiffness, default_damping, "global"
    return float(entry["fitted_stiffness"]), float(entry["fitted_damping"]), "fitted"


def find_anchors(rest_positions: np.ndarray, joint_positions: np.ndarray, threshold: float):
    """
    For each tracked point, find the nearest joint. If within threshold,
    mark as anchor and record which joint index drives it and the initial
    offset (point - joint), so the point keeps its relative position as the
    joint moves rather than snapping exactly onto the joint.

    NOTE: this is the TRANSLATE-only model. It is kept unchanged because
    compute_features.py imports it to decide which points count as anchors
    when selecting drape edges (where only the mask matters). The simulation
    itself now prefers build_anchor_frames() -- see the module docstring.
    """
    n = len(rest_positions)
    anchor_mask = np.zeros(n, dtype=bool)
    anchor_joint_idx = np.full(n, -1, dtype=int)
    anchor_offset = np.zeros((n, 2), dtype=np.float64)

    for i, pos in enumerate(rest_positions):
        dists = np.linalg.norm(joint_positions - pos, axis=1)
        nearest_joint = int(np.argmin(dists))
        if dists[nearest_joint] <= threshold:
            anchor_mask[i] = True
            anchor_joint_idx[i] = nearest_joint
            anchor_offset[i] = pos - joint_positions[nearest_joint]

    return anchor_mask, anchor_joint_idx, anchor_offset


# ---------------------------------------------------------------------------
# Rotation-aware anchors (fix G)
# ---------------------------------------------------------------------------

def _perpendicular(vec: np.ndarray) -> np.ndarray:
    """Rotate a 2D vector 90 degrees counter-clockwise."""
    return np.array([-vec[1], vec[0]], dtype=np.float64)


def select_anchors(rest_positions: np.ndarray, joint_positions: np.ndarray,
                   threshold: float | None = None, fraction: float | None = None):
    """
    Decide which tracked points are anchors. Returns (mask, nearest_joint_idx,
    nearest_distance).

    Two strategies:

      threshold - anchor every point within `threshold` pixels of a joint.
                  Intuitive, but the anchored FRACTION then depends on how
                  close the tracked points happen to sit to MediaPipe's
                  detected joints, which is NOT constant across the dataset.
                  Measured on this project's data at 60px: 0.0% of points were
                  free on real videos versus 8.0% on fakes. Real and fake were
                  therefore having *different computations* performed on them
                  -- no cloth physics at all on one side -- which invalidates
                  any real-vs-fake comparison of the residual.

      fraction  - anchor the closest `fraction` of points to joints, by rank.
                  Every segment then has exactly the same anchor/free split
                  regardless of label, so the comparison is like with like and
                  any remaining separation must come from the physics. This is
                  the preferred strategy.
    """
    n = len(rest_positions)
    nearest_idx = np.zeros(n, dtype=int)
    nearest_dist = np.zeros(n, dtype=np.float64)

    for i, pos in enumerate(rest_positions):
        dists = np.linalg.norm(joint_positions - pos, axis=1)
        j = int(np.argmin(dists))
        nearest_idx[i] = j
        nearest_dist[i] = dists[j]

    mask = np.zeros(n, dtype=bool)
    if fraction is not None:
        # At least one anchor (otherwise nothing drives the simulation) and at
        # least one free point (otherwise nothing is simulated).
        n_anchor = int(round(fraction * n))
        n_anchor = max(1, min(n - 1, n_anchor)) if n > 1 else n
        mask[np.argsort(nearest_dist)[:n_anchor]] = True
    else:
        mask = nearest_dist <= float(threshold)

    return mask, nearest_idx, nearest_dist


def build_anchor_frames(rest_positions: np.ndarray, joint_positions: np.ndarray,
                        threshold: float | None = None, fraction: float | None = None):
    """
    Build, for every anchored point, a local coordinate frame tied to two
    skeleton joints, so the point can rotate and scale with the limb rather
    than merely translating with a single joint.

    Returns (anchor_mask, frames) where frames[i] is None for a non-anchor and
    otherwise a dict:
        mode      "frame" (two usable joints) or "translate" (fallback)
        j_a       index of the primary (nearest) joint
        j_b       index of the secondary joint defining the limb direction
        a_coeff   offset component along the limb at the seed frame
        b_coeff   offset component across the limb at the seed frame
        length0   seed-frame distance between the two joints
        offset    plain seed-frame offset, used by the translate fallback
    """
    n = len(rest_positions)
    frames: list[dict | None] = [None] * n
    num_joints = len(joint_positions)

    anchor_mask, nearest_idx, _ = select_anchors(
        rest_positions, joint_positions, threshold=threshold, fraction=fraction
    )

    for i, pos in enumerate(rest_positions):
        if not anchor_mask[i]:
            continue

        dists = np.linalg.norm(joint_positions - pos, axis=1)
        order = np.argsort(dists)
        j_a = int(nearest_idx[i])
        offset = pos - joint_positions[j_a]

        # Find the closest OTHER joint that is far enough from j_a to define a
        # stable direction. Joints almost coincident with j_a would make the
        # frame spin wildly under normal pose jitter.
        j_b = -1
        for candidate in order[1:]:
            candidate = int(candidate)
            if candidate >= num_joints:
                continue
            if np.linalg.norm(joint_positions[candidate] - joint_positions[j_a]) >= MIN_FRAME_LENGTH_PX:
                j_b = candidate
                break

        if j_b < 0:
            frames[i] = {"mode": "translate", "j_a": j_a, "j_b": -1,
                         "a_coeff": 0.0, "b_coeff": 0.0, "length0": 0.0,
                         "offset": offset}
            continue

        limb = joint_positions[j_b] - joint_positions[j_a]
        length0 = float(np.linalg.norm(limb))
        unit = limb / length0
        normal = _perpendicular(unit)

        frames[i] = {
            "mode": "frame",
            "j_a": j_a,
            "j_b": j_b,
            "a_coeff": float(np.dot(offset, unit)),
            "b_coeff": float(np.dot(offset, normal)),
            "length0": length0,
            "offset": offset,
        }

    return anchor_mask, frames


def anchor_targets_for_frame(frames, anchor_mask, joint_positions: np.ndarray) -> np.ndarray:
    """
    Where should each anchored point be, given this frame's joint positions?
    Returns (num_anchors, 2) in the same order as anchor_mask's True entries,
    which is what ClothSimulator.set_anchor_positions() expects.
    """
    targets = []
    num_joints = len(joint_positions)

    for i, is_anchor in enumerate(anchor_mask):
        if not is_anchor:
            continue
        frame = frames[i]
        base = joint_positions[frame["j_a"]] if frame["j_a"] < num_joints else joint_positions[0]

        if frame["mode"] != "frame" or frame["j_b"] < 0 or frame["j_b"] >= num_joints:
            targets.append(base + frame["offset"])
            continue

        limb = joint_positions[frame["j_b"]] - joint_positions[frame["j_a"]]
        length = float(np.linalg.norm(limb))
        if length < 1e-6 or frame["length0"] < 1e-6:
            targets.append(base + frame["offset"])
            continue

        unit = limb / length
        normal = _perpendicular(unit)
        scale = min(max(length / frame["length0"], SCALE_LIMITS[0]), SCALE_LIMITS[1])
        targets.append(base + scale * (frame["a_coeff"] * unit + frame["b_coeff"] * normal))

    return np.asarray(targets, dtype=np.float64)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_segment(
    points: list[dict],
    pose_data: dict | None,
    seed_frame: int,
    k: int,
    anchor_threshold: float,
    gravity: float = 800.0,
    stiffness: float = 40.0,
    damping: float = 6.0,
    anchor_mode: str = "frame",
    anchor_fraction: float | None = None,
) -> list[dict] | None:
    """Returns a new list of points with simulated trajectories, or None if it can't be simulated."""
    num_points = len(points)
    num_frames = len(points[0]["trajectory"])
    frame_numbers = [e["frame"] for e in points[0]["trajectory"]]

    rest_positions = np.array(
        [[points[i]["trajectory"][0]["x"], points[i]["trajectory"][0]["y"]] for i in range(num_points)]
    )

    joint_positions_seed = get_joint_positions(pose_data, seed_frame) if pose_data else None
    if joint_positions_seed is None:
        return None

    # anchor_fraction, when set, overrides anchor_threshold. See select_anchors().
    threshold_arg = None if anchor_fraction is not None else anchor_threshold

    if anchor_mode == "frame":
        anchor_mask, anchor_frames = build_anchor_frames(
            rest_positions, joint_positions_seed,
            threshold=threshold_arg, fraction=anchor_fraction,
        )
    else:
        anchor_mask, anchor_joint_idx, _ = select_anchors(
            rest_positions, joint_positions_seed,
            threshold=threshold_arg, fraction=anchor_fraction,
        )
        anchor_frames = [
            {"mode": "translate", "j_a": int(anchor_joint_idx[i]), "j_b": -1,
             "a_coeff": 0.0, "b_coeff": 0.0, "length0": 0.0,
             "offset": rest_positions[i] - joint_positions_seed[anchor_joint_idx[i]]}
            if anchor_mask[i] else None
            for i in range(num_points)
        ]

    if not anchor_mask.any():
        return None

    springs = build_knn_springs(rest_positions, k=k)
    sim = ClothSimulator(
        rest_positions, springs, anchor_mask,
        gravity=gravity, stiffness=stiffness, damping=damping,
    )

    last_known_joint_positions = joint_positions_seed
    simulated = [[rest_positions[i].copy()] for i in range(num_points)]

    dt = 1.0 / FPS
    for frame_num in frame_numbers[1:]:
        joint_positions = get_joint_positions(pose_data, frame_num) if pose_data else None
        if joint_positions is None:
            joint_positions = last_known_joint_positions
        else:
            last_known_joint_positions = joint_positions

        if joint_positions is not None:
            targets = anchor_targets_for_frame(anchor_frames, anchor_mask, joint_positions)
            if len(targets):
                sim.set_anchor_positions(targets)

        sim.step(dt)

        for i in range(num_points):
            simulated[i].append(sim.positions[i].copy())

    result = []
    for i in range(num_points):
        trajectory = [
            {"frame": frame_numbers[t], "x": float(simulated[i][t][0]), "y": float(simulated[i][t][1])}
            for t in range(num_frames)
        ]
        result.append({"point_id": points[i]["point_id"], "trajectory": trajectory})

    return result


def process_video(trajectories_root: Path, pose_root: Path, output_root: Path,
                  label: str, video_name: str, k: int, anchor_threshold: float,
                  gravity: float, stiffness: float, damping: float,
                  anchor_mode: str, anchor_fraction: float | None,
                  material_fits: dict | None = None) -> None:
    traj_path = trajectories_root / label / f"{video_name}.json"
    with traj_path.open("r", encoding="utf-8") as f:
        traj_data = json.load(f)

    pose_data = load_pose(pose_root, label, video_name)
    if pose_data is None:
        print(f"  WARNING: No pose data found for {label}/{video_name}, skipping")
        return

    result = {}
    for garment in GARMENTS:
        if garment not in traj_data:
            continue

        k_use, d_use, source = material_for(
            material_fits or {}, label, video_name, garment, stiffness, damping
        )
        if source == "fitted":
            print(f"  {garment}: using fitted material stiffness={k_use:.0f} damping={d_use:.1f}")

        segments_out = []
        for segment in traj_data[garment]["segments"]:
            seed_frame = segment["seed_frame"]
            sim_points = simulate_segment(
                segment["points"], pose_data, seed_frame, k, anchor_threshold,
                gravity=gravity, stiffness=k_use, damping=d_use,
                anchor_mode=anchor_mode, anchor_fraction=anchor_fraction,
            )
            if sim_points is None:
                print(f"  {garment} segment seed_frame={seed_frame}: no anchors found, skipped")
                continue
            segments_out.append({"seed_frame": seed_frame, "points": sim_points})
        if segments_out:
            result[garment] = {"segments": segments_out}
        print(f"  {garment}: {len(segments_out)}/{len(traj_data[garment]['segments'])} segment(s) simulated")

    if not result:
        print(f"  WARNING: No segments simulated for {label}/{video_name}")
        return

    output_path = output_root / label / f"{video_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {output_path}")


def main() -> None:
    args = parse_args()

    trajectories_root = args.trajectories.resolve()
    pose_root = args.pose.resolve()
    output_root = args.output.resolve()

    if args.video:
        label, video_name = args.video.split("/", 1)
        videos = [(label, video_name)]
    elif args.all:
        videos = discover_videos(trajectories_root)
    else:
        raise SystemExit("ERROR: Specify --video <label/video_name> or --all")

    anchor_fraction = None if args.anchor_fraction is not None and args.anchor_fraction < 0 else args.anchor_fraction
    selection = (f"closest {anchor_fraction:.0%} of points"
                 if anchor_fraction is not None else f"within {args.anchor_threshold:.0f}px")

    material_fits = load_material_fits(args.material_fits)

    print(f"Videos to process: {len(videos)}")
    print(f"Anchor selection: {selection}")
    if material_fits:
        print(f"Materials: per-video fits from {args.material_fits} "
              f"({len(material_fits)} video(s))")
    print(f"Anchor mode: {args.anchor_mode}   "
          f"gravity={args.gravity}  stiffness={args.stiffness}  damping={args.damping}")
    print("-" * 60)

    for label, video_name in videos:
        print(f"\n{label}/{video_name}")
        process_video(trajectories_root, pose_root, output_root, label, video_name,
                      args.k, args.anchor_threshold,
                      args.gravity, args.stiffness, args.damping, args.anchor_mode,
                      anchor_fraction, material_fits)

    print("\nDone.")


if __name__ == "__main__":
    main()