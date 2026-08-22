"""
Phase 3 QC: automated acceptance checks for tracked point trajectories.

Computes three metrics per video, per garment, without requiring any manual
video review:

1. Visibility rate - % of point-frames CoTracker itself reported as
   confidently tracked (not occluded/lost/interpolated).

2. Mask-containment rate - % of point-frames whose (possibly interpolated)
   position actually falls inside that frame's real Phase 2 garment mask.
   This is an INDEPENDENT check from visibility, since it validates the
   FILLED-IN position too, not just CoTracker's own confidence signal.

3. Jump rate - % of frame-to-frame point movements that exceed a plausible
   displacement threshold, flagged as likely tracking glitches rather than
   real cloth motion.

A video passes automatically if both visibility rate and mask-containment
rate meet the threshold (default 90%). Videos below threshold are flagged
for manual review with visualize_trajectories.py.

Example usage:
  python qc_trajectories.py --trajectories trajectories --masks masks --frames frames
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LABELS = ("real", "fake")
GARMENTS = ("upper", "lower")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC checks for Phase 3 tracked trajectories.")
    parser.add_argument("--trajectories", type=Path, default=Path("trajectories"))
    parser.add_argument("--masks", type=Path, default=Path("masks"))
    parser.add_argument("--frames", type=Path, default=Path("frames"))
    parser.add_argument(
        "--visibility-threshold", type=float, default=0.90,
        help="Minimum visibility rate to pass automatically (default: 0.90).",
    )
    parser.add_argument(
        "--mask-threshold", type=float, default=0.90,
        help="Minimum mask-containment rate to pass automatically (default: 0.90).",
    )
    parser.add_argument(
        "--jump-threshold-px", type=float, default=40.0,
        help="Max plausible per-frame pixel displacement before flagging as a jump (default: 40.0).",
    )
    parser.add_argument("--output", type=Path, default=Path("qc_report.json"))
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


def load_mask(masks_root: Path, garment: str, label: str, video_name: str, frame_idx: int) -> np.ndarray | None:
    mask_dir = masks_root / garment / label / video_name
    for candidate in (mask_dir / f"frame_{frame_idx:04d}.png", mask_dir / f"{frame_idx:04d}.png"):
        if candidate.is_file():
            m = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return m
    return None


def point_in_mask(x: float, y: float, mask: np.ndarray | None) -> bool | None:
    """Returns None (skip) if no mask available, else True/False."""
    if mask is None:
        return None
    h, w = mask.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return False
    return bool(mask[yi, xi] > 0)


def qc_garment(
    garment_data: dict,
    masks_root: Path,
    garment: str,
    label: str,
    video_name: str,
    jump_threshold_px: float,
) -> dict:
    """
    garment_data is now {"segments": [{"seed_frame": N, "points": [...]}, ...]}
    -- each segment's points are independent point IDs (re-seeded), so jump
    detection resets at each segment boundary rather than comparing across
    a re-seed (which would be comparing two unrelated points, not real motion).
    """
    segments = garment_data.get("segments", [])
    if not segments:
        return {"visibility_rate": None, "mask_containment_rate": None, "jump_rate": None, "num_points": 0}

    mask_cache: dict[int, np.ndarray | None] = {}

    total_slots = 0
    visible_count = 0
    mask_checked = 0
    mask_ok = 0
    jump_checked = 0
    jump_count = 0
    total_points = 0

    for segment in segments:
        points = segment["points"]
        total_points += len(points)

        for point in points:
            traj = point["trajectory"]
            prev_xy = None  # resets per point, per segment -- no cross-segment jump comparisons
            for entry in traj:
                frame_idx = entry["frame"]
                x, y = entry["x"], entry["y"]
                visible = entry["visible"]

                total_slots += 1
                if visible:
                    visible_count += 1

                if frame_idx not in mask_cache:
                    mask_cache[frame_idx] = load_mask(masks_root, garment, label, video_name, frame_idx)
                mask = mask_cache[frame_idx]
                in_mask = point_in_mask(x, y, mask)
                if in_mask is not None:
                    mask_checked += 1
                    if in_mask:
                        mask_ok += 1

                if prev_xy is not None:
                    dist = ((x - prev_xy[0]) ** 2 + (y - prev_xy[1]) ** 2) ** 0.5
                    jump_checked += 1
                    if dist > jump_threshold_px:
                        jump_count += 1
                prev_xy = (x, y)

    return {
        "num_points": total_points,
        "num_segments": len(segments),
        "visibility_rate": visible_count / total_slots if total_slots else None,
        "mask_containment_rate": mask_ok / mask_checked if mask_checked else None,
        "jump_rate": jump_count / jump_checked if jump_checked else None,
    }


def main() -> None:
    args = parse_args()

    trajectories_root = args.trajectories.resolve()
    masks_root = args.masks.resolve()

    videos = discover_videos(trajectories_root)
    if not videos:
        raise SystemExit(f"ERROR: No trajectory files found under {trajectories_root}")

    print(f"Trajectories root: {trajectories_root}")
    print(f"Masks root       : {masks_root}")
    print(f"Videos found     : {len(videos)}")
    print(f"Thresholds       : visibility >= {args.visibility_threshold:.0%}, "
          f"mask-containment >= {args.mask_threshold:.0%}")
    print("-" * 100)
    print(f"{'Video':<40} {'Garment':<8} {'Visible%':>9} {'InMask%':>9} {'Jump%':>7}  {'Status'}")
    print("-" * 100)

    report = []
    flagged_videos = set()

    for label, video_name in videos:
        traj_path = trajectories_root / label / f"{video_name}.json"
        with traj_path.open("r", encoding="utf-8") as f:
            traj_data = json.load(f)

        for garment in GARMENTS:
            if garment not in traj_data:
                continue
            metrics = qc_garment(
                traj_data[garment], masks_root, garment, label, video_name, args.jump_threshold_px
            )

            vis = metrics["visibility_rate"]
            mask_rate = metrics["mask_containment_rate"]
            jump = metrics["jump_rate"]

            passed = (
                vis is not None and vis >= args.visibility_threshold
                and mask_rate is not None and mask_rate >= args.mask_threshold
            )
            status = "PASS" if passed else "REVIEW"
            if not passed:
                flagged_videos.add(f"{label}/{video_name}")

            vis_str = f"{vis:.1%}" if vis is not None else "n/a"
            mask_str = f"{mask_rate:.1%}" if mask_rate is not None else "n/a"
            jump_str = f"{jump:.1%}" if jump is not None else "n/a"

            print(f"{label + '/' + video_name:<40} {garment:<8} {vis_str:>9} {mask_str:>9} {jump_str:>7}  {status}")

            report.append({
                "video": f"{label}/{video_name}",
                "garment": garment,
                **metrics,
                "status": status,
            })

    print("-" * 100)
    print(f"\n{len(flagged_videos)} video(s) flagged for manual review:")
    for v in sorted(flagged_videos):
        print(f"  - {v}")
    if not flagged_videos:
        print("  (none - all videos passed automated checks)")

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()