"""
Phase 4, Step 8: Fit each video's OWN cloth material, scored by how much
simulating physics beats NOT simulating physics.

---------------------------------------------------------------------------
WHY THE OBJECTIVE IS A SKILL SCORE, NOT AN ERROR
---------------------------------------------------------------------------
Two earlier attempts minimised prediction error directly, and both failed the
same way: the fit ran to the stiffness ceiling every single time (100% of
video/garment fits came back marked *edge*). The reason is structural, not a
matter of search range. Stiffer fabric hugs the body more closely, and hugging
the body is a serviceable prediction for ANY video, real or fake. So error
falls monotonically with stiffness and there is no best value to find --
widening the range just moves the wall.

Worse, the winning "material" in that regime is a rigid sheet welded to the
skeleton. It predicts real and fake video equally well and therefore detects
nothing, which is exactly backwards.

So the question changes from

    "what error does this material achieve?"
to
    "does simulating physics beat NOT simulating physics?"

The rigid baseline is the whole garment pinned to the skeleton -- every point
carried along by its nearest joint, with limb rotation, but no gravity and no
springs. Then

    skill = rigid baseline error / physics simulation error

    skill  > 1   physics explains the motion better than rigid following
    skill == 1   physics adds nothing
    skill  < 1   physics is actively worse than doing nothing

Crank stiffness towards infinity and the simulation BECOMES the rigid
baseline, so skill falls to 1. Degenerate settings can no longer win, and a
genuine interior optimum exists: the material for which physics beats "glued
to the body" by the widest margin.

It also sharpens the detection claim. Real footage was produced by real
physics, so simulating physics should recover something the trivial baseline
misses. AI-generated motion was never produced by physics at all, so there
should be no material that beats the baseline -- skill near 1.

---------------------------------------------------------------------------
THE 30 / 70 SPLIT
---------------------------------------------------------------------------
Material is fitted on the first 30% of a video's segments (in time) and skill
is reported on the remaining 70%, so a video cannot win by memorising its own
quirks. It also mirrors deployment: watch a moment of footage, learn the
fabric, then check whether the rest behaves like that fabric.

---------------------------------------------------------------------------
WHAT COMES OUT
---------------------------------------------------------------------------
  fit_skill_eval    Skill on the held-out 70%. The primary feature.
  fitted_stiffness  Which material the motion implies.
  fitted_damping
  fit_contrast      best skill / median skill across the grid. How sharply
                    defined the best material is. Real cloth should be some
                    SPECIFIC fabric; motion from no physics should fit every
                    material about equally (contrast near 1).
  fit_at_boundary   Whether the best value sat at the edge of the range.
  fit_rigid_error   The baseline's own error, in torso lengths (diagnostic).
  fit_physics_error The simulation's error, in torso lengths (diagnostic).

Gravity stays GLOBAL: it is not a material property, and calibration showed
the data cannot pin it down anyway.

Usage:
  python fit_material.py --quick --gravity 400
  python fit_material.py --gravity 400
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from calibrate_physics import load_dataset, segment_scores
from run_simulation import (
    GARMENTS,
    anchor_targets_for_frame,
    build_anchor_frames,
    get_joint_positions,
    video_torso_scale,
)

# Large enough that every point counts as an anchor, giving the rigid baseline.
ALL_ANCHORED_THRESHOLD = 1e9

STIFFNESS_GRID = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0]
DAMPING_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

QUICK_STIFFNESS = [5.0, 20.0, 80.0, 320.0, 1280.0]
QUICK_DAMPING = [0.5, 2.0, 8.0, 32.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit per-video cloth material by how much physics beats a rigid baseline."
    )
    parser.add_argument("--trajectories", type=Path, default=Path("trajectories"))
    parser.add_argument("--pose", type=Path, default=Path("pose"))
    parser.add_argument("--output", type=Path, default=Path("material_fits.json"))
    parser.add_argument("--gravity", type=float, default=400.0,
                        help="Global gravity, not fitted per video (default: 400).")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--anchor-threshold", type=float, default=60.0)
    parser.add_argument("--anchor-fraction", type=float, default=0.20)
    parser.add_argument("--anchor-mode", choices=("frame", "translate"), default="frame")
    parser.add_argument("--fit-frac", type=float, default=0.30,
                        help="Fraction of each video's segments used to FIT (default: 0.30).")
    parser.add_argument("--quick", action="store_true", help="Coarse grid, much faster.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# The rigid baseline
# ---------------------------------------------------------------------------

def segment_rigid_error(segment, pose_data, anchor_threshold):
    """
    Error of the null model: the ENTIRE garment pinned to the skeleton.

    Every point is treated as an anchor, so each is carried along by its
    nearest joint (including limb rotation and scale) with no gravity and no
    springs. This is "the clothes are glued to the body" -- the prediction you
    get for free, without simulating any physics.

    Returns the mean pixel error against the observed VISIBLE positions, or
    None if it cannot be computed. Does not depend on stiffness or damping, so
    it is computed once per segment and reused across the whole grid.
    """
    points = segment["points"]
    if not points:
        return None
    seed_frame = segment["seed_frame"]

    rest_positions = np.array(
        [[p["trajectory"][0]["x"], p["trajectory"][0]["y"]] for p in points], dtype=float
    )
    joints_seed = get_joint_positions(pose_data, seed_frame)
    if joints_seed is None:
        return None

    anchor_mask, frames = build_anchor_frames(
        rest_positions, joints_seed, threshold=ALL_ANCHORED_THRESHOLD, fraction=None
    )
    if not anchor_mask.all():
        return None

    observed = {
        p["point_id"]: {e["frame"]: (e["x"], e["y"], bool(e.get("visible")))
                        for e in p["trajectory"]}
        for p in points
    }
    ids = [p["point_id"] for p in points]
    frame_numbers = [e["frame"] for e in points[0]["trajectory"]]

    last_joints = joints_seed
    errors = []
    for frame in frame_numbers:
        joints = get_joint_positions(pose_data, frame)
        if joints is None:
            joints = last_joints
        else:
            last_joints = joints

        targets = anchor_targets_for_frame(frames, anchor_mask, joints)
        if len(targets) != len(points):
            continue

        for index, point_id in enumerate(ids):
            entry = observed.get(point_id, {}).get(frame)
            if entry is None or not entry[2]:
                continue
            errors.append(math.hypot(entry[0] - targets[index][0],
                                     entry[1] - targets[index][1]))

    return float(np.mean(errors)) if errors else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_segments(segments, rigid_errors, pose_data, scale, stiffness, damping, args):
    """
    Mean skill across a set of segments, plus the two errors that produced it.

    skill = rigid baseline error / physics error, per segment, then averaged.
    Averaging the per-segment RATIO rather than dividing two averages keeps one
    long or badly-tracked segment from dominating.
    """
    skills, rigid_norm, physics_norm = [], [], []

    for segment, rigid in zip(segments, rigid_errors):
        if rigid is None or rigid <= 1e-9:
            continue
        scored = segment_scores(
            segment, pose_data, args.k, args.anchor_threshold,
            args.gravity, stiffness, damping,
            args.anchor_mode, args.anchor_fraction,
        )
        if scored is None:
            continue
        physics = scored[0]
        if physics <= 1e-9:
            continue
        skills.append(rigid / physics)
        rigid_norm.append(rigid / scale)
        physics_norm.append(physics / scale)

    if not skills:
        return None
    return {
        "skill": float(np.mean(skills)),
        "rigid_error": float(np.mean(rigid_norm)),
        "physics_error": float(np.mean(physics_norm)),
        "n_segments": len(skills),
    }


def fit_one(segments, pose_data, scale, args, stiffness_grid, damping_grid):
    """Fit material on the first fit_frac of segments, report skill on the rest."""
    ordered = sorted(segments, key=lambda s: s["seed_frame"])
    if len(ordered) < 2:
        return None

    n_fit = max(1, min(len(ordered) - 1, int(round(args.fit_frac * len(ordered)))))
    fit_segments, eval_segments = ordered[:n_fit], ordered[n_fit:]

    # The baseline is independent of the material, so compute it once.
    fit_rigid = [segment_rigid_error(s, pose_data, args.anchor_threshold) for s in fit_segments]
    eval_rigid = [segment_rigid_error(s, pose_data, args.anchor_threshold) for s in eval_segments]
    if all(r is None for r in fit_rigid):
        return None

    results = []
    for stiffness in stiffness_grid:
        for damping in damping_grid:
            scored = score_segments(fit_segments, fit_rigid, pose_data, scale,
                                    stiffness, damping, args)
            if scored is None:
                continue
            scored.update({"stiffness": stiffness, "damping": damping})
            results.append(scored)

    if not results:
        return None

    best = max(results, key=lambda r: r["skill"])

    # How sharply defined is the best material? Real cloth should be some
    # SPECIFIC fabric and stand out; motion produced by no physics should fit
    # every material about equally, giving a contrast near 1.
    typical = float(statistics.median([r["skill"] for r in results]))
    contrast = (best["skill"] / typical) if typical > 1e-9 else float("nan")

    at_boundary = (
        best["stiffness"] in (stiffness_grid[0], stiffness_grid[-1])
        or best["damping"] in (damping_grid[0], damping_grid[-1])
    )

    held_out = score_segments(eval_segments, eval_rigid, pose_data, scale,
                              best["stiffness"], best["damping"], args)

    return {
        "fitted_stiffness": best["stiffness"],
        "fitted_damping": best["damping"],
        "fit_skill_train": best["skill"],
        "fit_skill_eval": held_out["skill"] if held_out else None,
        "fit_rigid_error": held_out["rigid_error"] if held_out else best["rigid_error"],
        "fit_physics_error": held_out["physics_error"] if held_out else best["physics_error"],
        "fit_contrast": contrast,
        "fit_at_boundary": bool(at_boundary),
        "n_fit_segments": len(fit_segments),
        "n_eval_segments": len(eval_segments),
        "fit_seed_frames": [s["seed_frame"] for s in fit_segments],
        "torso_scale_px": scale,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.anchor_fraction is not None and args.anchor_fraction < 0:
        args.anchor_fraction = None

    stiffness_grid = QUICK_STIFFNESS if args.quick else STIFFNESS_GRID
    damping_grid = QUICK_DAMPING if args.quick else DAMPING_GRID

    print("Loading dataset...")
    dataset = load_dataset(args.trajectories.resolve(), args.pose.resolve())
    print(f"  {len(dataset)} videos")
    print(f"  gravity fixed at {args.gravity:.0f} (global, not a material property)")
    print(f"  grid: {len(stiffness_grid)} stiffness x {len(damping_grid)} damping "
          f"= {len(stiffness_grid) * len(damping_grid)} materials per video/garment")
    print(f"  objective: MAXIMISE skill = rigid-baseline error / physics error")
    print(f"  fit on first {args.fit_frac:.0%} of segments, evaluate on the rest")
    print()

    fits = {}
    print("%-32s %-7s %8s %7s %8s %8s %7s %7s"
          % ("video", "garment", "stiff", "damp", "skill_tr", "skill_ev", "contr", "rigid"))
    print("-" * 96)

    for record in dataset:
        label, video = record["label"], record["video"]
        scale = video_torso_scale(record["pose"])
        if scale is None:
            print("%-32s  SKIPPED - torso landmarks unavailable" % video[:32])
            continue

        key = f"{label}/{video}"
        fits[key] = {"label": label, "video": video, "torso_scale_px": scale, "garments": {}}

        for garment in GARMENTS:
            if garment not in record["traj"]:
                continue
            result = fit_one(record["traj"][garment]["segments"], record["pose"],
                             scale, args, stiffness_grid, damping_grid)
            if result is None:
                print("%-32s %-7s  could not fit" % (video[:32], garment))
                continue
            fits[key]["garments"][garment] = result

            skill_ev = ("%8.3f" % result["fit_skill_eval"]) if result["fit_skill_eval"] is not None else "     n/a"
            print("%-32s %-7s %8.0f %7.2f %8.3f %s %7.2f %7.4f%s"
                  % (video[:32], garment, result["fitted_stiffness"], result["fitted_damping"],
                     result["fit_skill_train"], skill_ev, result["fit_contrast"],
                     result["fit_rigid_error"], "  *edge*" if result["fit_at_boundary"] else ""))

    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"gravity": args.gravity, "anchor_fraction": args.anchor_fraction,
                   "anchor_mode": args.anchor_mode, "fit_frac": args.fit_frac,
                   "objective": "skill = rigid_error / physics_error",
                   "fits": fits}, f, indent=2)

    # ------------------------------------------------------------------
    # Summary. Descriptive only -- labels played no part in fitting.
    # ------------------------------------------------------------------
    print()
    print("=" * 96)
    print("SUMMARY BY LABEL  (labels were NOT used during fitting)")
    print("=" * 96)
    print("%-8s %8s %11s %9s %11s %10s %9s"
          % ("label", "fits", "stiffness", "damping", "skill (eval)", "contrast", "at edge"))
    print("-" * 96)

    summary = {}
    for label in ("real", "fake"):
        rows = [g for entry in fits.values() if entry["label"] == label
                for g in entry["garments"].values()]
        if not rows:
            continue

        def mean_of(field):
            values = [float(r[field]) for r in rows
                      if r.get(field) is not None and math.isfinite(float(r[field]))]
            return float(np.mean(values)) if values else float("nan")

        summary[label] = {"skill": mean_of("fit_skill_eval"), "contrast": mean_of("fit_contrast")}
        edge = 100.0 * float(np.mean([1.0 if r["fit_at_boundary"] else 0.0 for r in rows]))
        print("%-8s %8d %11.0f %9.2f %11.3f %10.2f %8.0f%%"
              % (label, len(rows), mean_of("fitted_stiffness"), mean_of("fitted_damping"),
                 summary[label]["skill"], summary[label]["contrast"], edge))

    print()
    print("Reading the skill column: >1 means simulating physics beat gluing the")
    print("clothes to the body. =1 means physics added nothing. <1 means it hurt.")
    if "real" in summary and "fake" in summary:
        real_skill, fake_skill = summary["real"]["skill"], summary["fake"]["skill"]
        if math.isfinite(real_skill) and math.isfinite(fake_skill):
            print()
            if real_skill <= 1.02:
                print("  WARNING: physics barely beats the trivial baseline even on REAL video")
                print("           (skill %.3f). Before reading anything into the real-vs-fake"
                      % real_skill)
                print("           comparison, the simulation needs to earn its place at all.")
            elif real_skill > fake_skill:
                print("  Physics helps more on real video (%.3f) than on fake (%.3f), which is"
                      % (real_skill, fake_skill))
                print("  the hypothesised direction: real motion was produced by real physics,")
                print("  so a physics model recovers something the baseline misses.")
            else:
                print("  Physics helps MORE on fake video (%.3f) than on real (%.3f)."
                      % (fake_skill, real_skill))
                print("  That contradicts the hypothesis and needs reporting as such, not")
                print("  explaining away.")

    print()
    print(f"Saved: {args.output.resolve()}")
    print()
    print("Next:")
    print("  python run_simulation.py --all --material-fits material_fits.json --gravity %.0f"
          % args.gravity)
    print("  python compute_features.py --material-fits material_fits.json")
    print("  python analyze_features.py --features phase4_features.csv --eval-only --plots")


if __name__ == "__main__":
    main()