"""
Phase 4, Step 7: Calibrate the cloth simulator's physics constants.

gravity=800, stiffness=40, damping=6 were reasonable-looking GUESSES. This
script replaces them with values fitted to the data.

---------------------------------------------------------------------------
METHOD, and why it is set up this way
---------------------------------------------------------------------------
The constants are fitted on REAL VIDEOS ONLY. This matters twice over:

  1. No label leakage. If fakes influenced the fit, any later real-vs-fake
     separation would be partly circular and an examiner would be right to
     throw it out.
  2. It is the correct scientific framing. The claim becomes: "we tuned a
     Newtonian cloth model until it reproduced REAL cloth as closely as it
     could, then measured how far AI-generated cloth departs from it." The
     fake residual is then a genuine out-of-sample measurement.

THE DEGENERACY TRAP
-------------------
Minimising residual on real videos alone has an obvious cheat: crank stiffness
and damping high enough and the cloth stops behaving like cloth and simply
follows the anchor points rigidly. That yields a beautifully low residual on
real videos -- and an equally low one on fakes, because the simulation is no
longer making any physical prediction at all. Separation collapses to zero
and the whole project quietly stops working.

So every candidate is also scored on free_dynamics: how far the freely
simulated points travel RELATIVE to rigid anchor-following. A candidate whose
free_dynamics falls below --min-free-dynamics is rejected as degenerate,
however good its residual looks.

The fake residual IS reported at the end, clearly marked. It is a diagnostic
readout only and is never used to choose parameters.

Usage:
  python calibrate_physics.py --quick
  python calibrate_physics.py
  python calibrate_physics.py --loo            # leave-one-real-video-out stability check
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from run_simulation import (
    GARMENTS,
    build_anchor_frames,
    select_anchors,
    discover_videos,
    get_joint_positions,
    load_pose,
    simulate_segment,
)

# Ranges extended after the first coarse run put all three constants on a grid
# BOUNDARY (gravity at the low end, stiffness and damping at the high end),
# which means the real optimum lay outside the values tested.
DEFAULT_GRIDS = {
    "gravity": [25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1200.0, 1800.0, 2600.0],
    "stiffness": [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0],
    "damping": [0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 16.0, 25.0, 40.0, 60.0],
}

QUICK_GRIDS = {
    "gravity": [200.0, 800.0, 1800.0],
    "stiffness": [10.0, 40.0, 160.0],
    "damping": [1.0, 6.0, 16.0],
}

# Order matters: stiffness and damping dominate the simulator's behaviour, so
# they are settled before gravity's finer effect is tuned.
COORDINATE_ORDER = ["stiffness", "damping", "gravity"]


def anchor_audit(dataset, args):
    """
    How much of the garment is actually being SIMULATED?

    Every tracked point within --anchor-threshold of a skeleton joint is pinned
    to that joint and driven kinematically. Only the remaining "free" points
    obey gravity and the springs. If the free fraction is ~0, then the physics
    residual is not measuring cloth physics at all -- it is measuring how well
    the anchor model tracks the body, and no choice of constants can change
    that. This is the same phenomenon that silently broke the drape feature,
    so it is checked explicitly and up front.
    """
    print("=" * 70)
    print("ANCHOR AUDIT - how much cloth is actually simulated?")
    print("=" * 70)
    print("Points within the anchor threshold are pinned to the skeleton. Only the")
    print("remaining FREE points are governed by gravity and springs.")
    print()
    print("%-12s %14s %14s" % ("threshold", "free % (real)", "free % (fake)"))
    print("-" * 70)

    thresholds = sorted({10.0, 20.0, 30.0, 45.0, args.anchor_threshold, 80.0})
    current_free = {}

    for threshold in thresholds:
        by_label = {}
        for label in ("real", "fake"):
            fractions = []
            for record in dataset:
                if record["label"] != label:
                    continue
                for garment in GARMENTS:
                    if garment not in record["traj"]:
                        continue
                    for segment in record["traj"][garment]["segments"][:args.max_segments or None]:
                        points = segment["points"]
                        rest = np.array(
                            [[p["trajectory"][0]["x"], p["trajectory"][0]["y"]] for p in points],
                            dtype=float,
                        )
                        joints = get_joint_positions(record["pose"], segment["seed_frame"])
                        if joints is None:
                            continue
                        mask, _ = build_anchor_frames(rest, joints, threshold)
                        fractions.append(float((~mask).mean()))
            by_label[label] = 100.0 * float(np.mean(fractions)) if fractions else float("nan")

        marker = "   <-- current setting" if threshold == args.anchor_threshold else ""
        print("%9.0f px %13.1f%% %13.1f%%%s"
              % (threshold, by_label["real"], by_label["fake"], marker))
        if threshold == args.anchor_threshold:
            current_free = by_label

    # --- pose-quality readout ------------------------------------------------
    # The free-fraction gap between labels is driven by how far tracked points
    # sit from MediaPipe's joints. Quantify that directly, normalised by the
    # skeleton's own size so subject distance from camera does not confound it.
    print()
    print("Distance from tracked points to nearest joint (normalised by skeleton size):")
    print("%-12s %16s %16s" % ("label", "mean dist/size", "segments"))
    print("-" * 70)
    pose_norm = {}
    for label in ("real", "fake"):
        values = []
        for record in dataset:
            if record["label"] != label:
                continue
            for garment in GARMENTS:
                if garment not in record["traj"]:
                    continue
                for segment in record["traj"][garment]["segments"][:args.max_segments or None]:
                    points = segment["points"]
                    rest = np.array(
                        [[q["trajectory"][0]["x"], q["trajectory"][0]["y"]] for q in points],
                        dtype=float,
                    )
                    joints = get_joint_positions(record["pose"], segment["seed_frame"])
                    if joints is None:
                        continue
                    _, _, nearest = select_anchors(rest, joints, fraction=0.5)
                    size = float(np.linalg.norm(joints.max(axis=0) - joints.min(axis=0)))
                    if size > 1e-6:
                        values.append(float(np.mean(nearest)) / size)
        pose_norm[label] = float(np.mean(values)) if values else float("nan")
        print("%-12s %16.4f %16d" % (label, pose_norm[label], len(values)))

    print()
    if pose_norm.get("real") == pose_norm.get("real") and pose_norm.get("fake") == pose_norm.get("fake"):
        if pose_norm["real"] > 1e-9:
            ratio = pose_norm["fake"] / pose_norm["real"]
            print("  fake/real ratio: %.2fx" % ratio)
            if ratio > 1.3 or ratio < 0.77:
                print("  WARNING: tracked points sit at systematically different distances from")
                print("           the detected skeleton depending on the label. That points to")
                print("           MediaPipe behaving differently on AI-generated video, which")
                print("           makes anchor targets less accurate there. Part of any residual")
                print("           separation would then be POSE error, not cloth physics.")
                print("           Rank-based anchoring (--anchor-fraction) equalises the anchor")
                print("           STRUCTURE, but cannot fix inaccurate joint positions. Report")
                print("           pose_dist_norm from the feature table alongside your results.")
            else:
                print("  OK: pose-to-garment distances are comparable across labels.")

    print()
    if args.anchor_fraction is not None:
        print("NOTE: --anchor-fraction %.2f is active, so the threshold column above is"
              % args.anchor_fraction)
        print("      informational only. The run itself anchors the closest %.0f%% of points"
              % (100 * args.anchor_fraction))
        print("      in EVERY segment, making the anchor/free split identical by construction.")
        print()
        return

    free_now = current_free.get("real", float("nan"))
    if not (free_now == free_now):          # NaN
        print("Could not audit anchors (no usable pose data).")
    elif free_now < 2.0:
        print("CRITICAL: at the current threshold almost NOTHING is freely simulated.")
        print("          The physics residual is therefore measuring the kinematic anchor")
        print("          model, not cloth physics, and calibrating constants cannot help.")
        print("          Lower --anchor-threshold until a meaningful fraction is free.")
    elif free_now < 15.0:
        print("WARNING: only a small fraction of points are freely simulated, so the")
        print("         residual is dominated by anchor tracking rather than cloth")
        print("         behaviour. Consider lowering --anchor-threshold.")
    else:
        print("OK: a meaningful fraction of the garment is freely simulated, so the")
        print("    residual genuinely reflects cloth physics.")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Fit cloth physics constants on real videos only.")
    parser.add_argument("--trajectories", type=Path, default=Path("trajectories"))
    parser.add_argument("--pose", type=Path, default=Path("pose"))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--anchor-threshold", type=float, default=60.0)
    parser.add_argument("--anchor-mode", choices=("frame", "translate"), default="frame")
    parser.add_argument("--anchor-fraction", type=float, default=0.30,
                        help="Anchor the closest this-fraction of points to joints, by rank "
                             "(default: 0.30). Negative falls back to --anchor-threshold.")
    parser.add_argument("--max-segments", type=int, default=3,
                        help="Segments per garment per video used during the search (default: 3). "
                             "The final chosen parameters are re-scored on ALL segments.")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Coordinate-descent passes over the three parameters (default: 2).")
    parser.add_argument("--quick", action="store_true", help="Coarse 3-value grids; much faster.")
    parser.add_argument("--min-free-dynamics", type=float, default=1.0,
                        help="Reject parameter sets where freely simulated points move less than this "
                             "many pixels (mean) relative to rigid anchor-following (default: 1.0).")
    parser.add_argument("--loo", action="store_true",
                        help="Also re-fit holding out each real video in turn, to check stability.")
    parser.add_argument("--output", type=Path, default=Path("calibration_result.json"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(trajectories_root: Path, pose_root: Path):
    """Load every video's observed trajectories and pose once, up front."""
    dataset = []
    for label, video_name in discover_videos(trajectories_root):
        traj_path = trajectories_root / label / f"{video_name}.json"
        with traj_path.open(encoding="utf-8") as f:
            traj_data = json.load(f)
        pose_data = load_pose(pose_root, label, video_name)
        if pose_data is None:
            print(f"  skipping {label}/{video_name}: no pose data")
            continue
        dataset.append({"label": label, "video": video_name,
                        "traj": traj_data, "pose": pose_data})
    return dataset


# ---------------------------------------------------------------------------
# Scoring one segment
# ---------------------------------------------------------------------------

def segment_scores(segment, pose_data, k, anchor_threshold, gravity, stiffness, damping,
                   anchor_mode, anchor_fraction=None):
    """
    Simulate one segment and return (mean_residual_px, free_dynamics_px), or
    None if it could not be simulated.

    mean_residual_px uses VISIBLE observed frames only -- matching
    compute_features.py, so calibration optimises exactly the quantity the
    feature table later reports.

    free_dynamics_px measures how far the freely simulated points end up from
    where they would be under pure rigid translation of the anchor cloud. It
    is the guard against the degenerate "cloth just follows the body" optimum
    described in the module docstring.
    """
    points = segment["points"]
    seed_frame = segment["seed_frame"]

    sim_points = simulate_segment(
        points, pose_data, seed_frame, k, anchor_threshold,
        gravity=gravity, stiffness=stiffness, damping=damping,
        anchor_mode=anchor_mode, anchor_fraction=anchor_fraction,
    )
    if sim_points is None:
        return None

    sim_by_id = {
        p["point_id"]: {e["frame"]: (e["x"], e["y"]) for e in p["trajectory"]}
        for p in sim_points
    }

    # --- residual, visible frames only -------------------------------------
    gaps = []
    for point in points:
        table = sim_by_id.get(point["point_id"])
        if not table:
            continue
        for entry in point["trajectory"]:
            if not entry.get("visible"):
                continue
            simulated = table.get(entry["frame"])
            if simulated is None:
                continue
            gaps.append(math.hypot(entry["x"] - simulated[0], entry["y"] - simulated[1]))

    if not gaps:
        return None

    # --- free dynamics ------------------------------------------------------
    rest_positions = np.array(
        [[p["trajectory"][0]["x"], p["trajectory"][0]["y"]] for p in points], dtype=float
    )
    joint_seed = get_joint_positions(pose_data, seed_frame)
    if joint_seed is None:
        return float(np.mean(gaps)), 0.0
    anchor_mask, _ = build_anchor_frames(
        rest_positions, joint_seed,
        threshold=None if anchor_fraction is not None else anchor_threshold,
        fraction=anchor_fraction,
    )

    free_idx = [i for i, a in enumerate(anchor_mask) if not a]
    anchor_idx = [i for i, a in enumerate(anchor_mask) if a]
    free_frac = len(free_idx) / len(anchor_mask) if len(anchor_mask) else 0.0

    # Two very different situations that must NOT be conflated:
    #   free_frac == 0  -> every point is pinned to the skeleton. Nothing is
    #                      being simulated at all, so the "physics residual"
    #                      is really just measuring the kinematic anchor model.
    #                      No choice of gravity/stiffness/damping can change it.
    #   free_dynamics 0 -> there ARE free points, but the chosen constants make
    #                      them follow the body rigidly (the degenerate optimum).
    if not free_idx or not anchor_idx:
        return float(np.mean(gaps)), 0.0, free_frac

    id_by_index = [p["point_id"] for p in points]
    frame_numbers = [e["frame"] for e in points[0]["trajectory"]]

    def sim_pos(index, frame):
        return sim_by_id.get(id_by_index[index], {}).get(frame)

    anchor_seed_centroid = rest_positions[anchor_idx].mean(axis=0)
    deviations = []
    for frame in frame_numbers[1:]:
        anchor_now = [sim_pos(i, frame) for i in anchor_idx]
        anchor_now = [p for p in anchor_now if p is not None]
        if not anchor_now:
            continue
        centroid_now = np.asarray(anchor_now, dtype=float).mean(axis=0)
        shift = centroid_now - anchor_seed_centroid
        for i in free_idx:
            actual = sim_pos(i, frame)
            if actual is None:
                continue
            rigid = rest_positions[i] + shift
            deviations.append(math.hypot(actual[0] - rigid[0], actual[1] - rigid[1]))

    free_dynamics = float(np.mean(deviations)) if deviations else 0.0
    return float(np.mean(gaps)), free_dynamics, free_frac


def evaluate(dataset, params, args, labels=("real",), exclude_video=None, max_segments=None):
    """
    Score a candidate parameter set over the requested labels.

    Returns a dict with the median per-segment residual (the objective; median
    rather than mean so one badly-tracked segment cannot steer the fit), the
    mean free-dynamics, and how many segments contributed.
    """
    limit = args.max_segments if max_segments is None else max_segments
    residuals, free_dynamics, free_fracs = [], [], []

    for record in dataset:
        if record["label"] not in labels:
            continue
        if exclude_video is not None and record["video"] == exclude_video:
            continue

        for garment in GARMENTS:
            if garment not in record["traj"]:
                continue
            segments = record["traj"][garment]["segments"]
            if limit:
                segments = segments[:limit]
            for segment in segments:
                scored = segment_scores(
                    segment, record["pose"], args.k, args.anchor_threshold,
                    params["gravity"], params["stiffness"], params["damping"],
                    args.anchor_mode, args.anchor_fraction,
                )
                if scored is None:
                    continue
                residuals.append(scored[0])
                free_dynamics.append(scored[1])
                free_fracs.append(scored[2])

    if not residuals:
        return {"residual": float("inf"), "free_dynamics": 0.0,
                "free_frac": 0.0, "n_segments": 0}

    return {
        "residual": float(statistics.median(residuals)),
        "residual_mean": float(np.mean(residuals)),
        "free_dynamics": float(np.mean(free_dynamics)),
        "free_frac": float(np.mean(free_fracs)),
        "n_segments": len(residuals),
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def coordinate_descent(dataset, args, grids, exclude_video=None, verbose=True):
    params = {"gravity": 800.0, "stiffness": 40.0, "damping": 6.0}
    history = []

    baseline = evaluate(dataset, params, args, exclude_video=exclude_video)
    if verbose:
        print(f"  baseline  g={params['gravity']:.0f} k={params['stiffness']:.0f} "
              f"d={params['damping']:.1f}  ->  residual={baseline['residual']:.2f}px  "
              f"free_dyn={baseline['free_dynamics']:.2f}px")

    best = baseline
    for round_idx in range(args.rounds):
        improved = False
        for name in COORDINATE_ORDER:
            candidates = grids[name]
            local_best_value, local_best_score = params[name], best

            for value in candidates:
                if value == params[name]:
                    continue
                trial = dict(params)
                trial[name] = value
                score = evaluate(dataset, trial, args, exclude_video=exclude_video)

                # If nothing is freely simulated, no parameter choice can matter;
                # rejecting every candidate as "degenerate" would be misleading.
                no_free_points = score["free_frac"] <= 0.0
                degenerate = (not no_free_points) and score["free_dynamics"] < args.min_free_dynamics
                history.append({**trial, **score, "degenerate": degenerate})

                if verbose:
                    flag = "  REJECTED (degenerate)" if degenerate else ""
                    print(f"    {name}={value:<7.1f} residual={score['residual']:7.2f}px  "
                          f"free_dyn={score['free_dynamics']:6.2f}px{flag}")

                if degenerate:
                    continue
                if score["residual"] < local_best_score["residual"]:
                    local_best_value, local_best_score = value, score

            if local_best_value != params[name]:
                params[name] = local_best_value
                best = local_best_score
                improved = True

        if verbose:
            print(f"  after round {round_idx + 1}: g={params['gravity']:.0f} "
                  f"k={params['stiffness']:.0f} d={params['damping']:.1f}  "
                  f"residual={best['residual']:.2f}px")
        if not improved:
            break

    return params, best, baseline, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.anchor_fraction is not None and args.anchor_fraction < 0:
        args.anchor_fraction = None
    grids = QUICK_GRIDS if args.quick else DEFAULT_GRIDS

    print("Loading dataset...")
    dataset = load_dataset(args.trajectories.resolve(), args.pose.resolve())
    n_real = sum(1 for r in dataset if r["label"] == "real")
    n_fake = sum(1 for r in dataset if r["label"] == "fake")
    print(f"  {len(dataset)} videos ({n_real} real, {n_fake} fake)")
    print(f"  fitting on REAL only; up to {args.max_segments} segment(s) per garment per video")
    print(f"  anchor mode: {args.anchor_mode}")
    print()

    if n_real == 0:
        raise SystemExit("ERROR: no real videos with pose data - cannot calibrate.")

    anchor_audit(dataset, args)

    print("=" * 70)
    print("COORDINATE DESCENT (real videos only)")
    print("=" * 70)
    params, best, baseline, history = coordinate_descent(dataset, args, grids)

    print()
    print("=" * 70)
    print("FINAL SCORING (all segments)")
    print("=" * 70)

    old = {"gravity": 800.0, "stiffness": 40.0, "damping": 6.0}
    old_real = evaluate(dataset, old, args, labels=("real",), max_segments=0)
    new_real = evaluate(dataset, params, args, labels=("real",), max_segments=0)

    # Diagnostic only. Never used to select parameters -- see module docstring.
    old_fake = evaluate(dataset, old, args, labels=("fake",), max_segments=0)
    new_fake = evaluate(dataset, params, args, labels=("fake",), max_segments=0)

    print()
    print("%-22s %12s %12s %12s" % ("", "gravity/k/d", "REAL resid", "free_dyn"))
    print("-" * 70)
    print("%-22s %12s %10.2fpx %10.2fpx"
          % ("old (guessed)", f"{old['gravity']:.0f}/{old['stiffness']:.0f}/{old['damping']:.1f}",
             old_real["residual"], old_real["free_dynamics"]))
    print("%-22s %12s %10.2fpx %10.2fpx"
          % ("new (calibrated)", f"{params['gravity']:.0f}/{params['stiffness']:.0f}/{params['damping']:.1f}",
             new_real["residual"], new_real["free_dynamics"]))

    print()
    print("DIAGNOSTIC ONLY - fake residuals were NOT used to choose parameters:")
    print("%-22s %10s %10s %10s" % ("", "real", "fake", "ratio"))
    for name, real_score, fake_score in (("old (guessed)", old_real, old_fake),
                                         ("new (calibrated)", new_real, new_fake)):
        ratio = (fake_score["residual"] / real_score["residual"]) if real_score["residual"] > 0 else float("nan")
        print("%-22s %8.2fpx %8.2fpx %10.2fx" % (name, real_score["residual"], fake_score["residual"], ratio))

    print()
    old_ratio = old_fake["residual"] / old_real["residual"] if old_real["residual"] > 0 else 0.0
    new_ratio = new_fake["residual"] / new_real["residual"] if new_real["residual"] > 0 else 0.0
    if new_real["free_dynamics"] < args.min_free_dynamics:
        print("WARNING: the calibrated simulation is close to degenerate (free points barely")
        print("         move relative to the body). Raise --min-free-dynamics and re-run.")
    elif new_ratio > old_ratio:
        print(f"GOOD: calibration lowered the real residual AND widened the fake/real ratio")
        print(f"      ({old_ratio:.2f}x -> {new_ratio:.2f}x). The physics model now fits real")
        print(f"      cloth better, and fake cloth stands out more against it.")
    else:
        print(f"CAUTION: the fake/real ratio did not improve ({old_ratio:.2f}x -> {new_ratio:.2f}x).")
        print( "         A better fit to real cloth did not translate into better separation.")
        print( "         Report this honestly - it is a real result, not a failure.")

    result = {
        "calibrated": params,
        "previous": old,
        "anchor_mode": args.anchor_mode,
        "real_only_fit": True,
        "scores": {
            "old_real": old_real, "new_real": new_real,
            "old_fake_DIAGNOSTIC_ONLY": old_fake, "new_fake_DIAGNOSTIC_ONLY": new_fake,
        },
        "search_history": history,
    }

    # ----------------------------------------------------------------------
    # Optional stability check
    # ----------------------------------------------------------------------
    if args.loo:
        print()
        print("=" * 70)
        print("LEAVE-ONE-REAL-VIDEO-OUT STABILITY CHECK")
        print("=" * 70)
        print("If the chosen constants swing wildly when one video is removed, the fit")
        print("is driven by that video rather than by cloth physics in general.")
        print()

        loo_params = []
        for record in dataset:
            if record["label"] != "real":
                continue
            fitted, _, _, _ = coordinate_descent(
                dataset, args, grids, exclude_video=record["video"], verbose=False
            )
            loo_params.append({"held_out": record["video"], **fitted})
            print("  without %-32s -> g=%.0f k=%.0f d=%.1f"
                  % (record["video"][:32], fitted["gravity"], fitted["stiffness"], fitted["damping"]))

        print()
        for name in ("gravity", "stiffness", "damping"):
            values = [p[name] for p in loo_params]
            spread = (max(values) / min(values)) if min(values) > 0 else float("inf")
            verdict = "stable" if spread <= 2.0 else "UNSTABLE"
            print("  %-10s min=%8.1f  max=%8.1f  max/min=%5.2fx   %s"
                  % (name, min(values), max(values), spread, verdict))
        result["loo"] = loo_params

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print()
    print(f"Saved: {args.output.resolve()}")
    print()
    print("To apply these constants:")
    print("  python run_simulation.py --all --gravity %.0f --stiffness %.0f --damping %.1f"
          % (params["gravity"], params["stiffness"], params["damping"]))
    print("  python compute_features.py")
    print("  python analyze_features.py --features phase4_features.csv --plots")


if __name__ == "__main__":
    main()