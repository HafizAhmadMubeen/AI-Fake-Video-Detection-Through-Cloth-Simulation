"""
Save tracked points overlaid on individual frame images (not a video), so
tracking quality can be inspected frame-by-frame carefully rather than
relying on catching details while a video plays.

Same coloring convention as visualize_trajectories.py:
  - Upper garment points: blue
  - Lower garment points: orange
  - Bright/solid color  = genuinely tracked this frame (visible: true)
  - Faded/dim color     = interpolated (visible: false)

Example usage:
  python visualize_frames.py --frames frames --trajectories trajectories --output frame_review --video "real/Man Excersing"

  # Only save every Nth frame instead of all of them, to review fewer images
  python visualize_frames.py --frames frames --trajectories trajectories --output frame_review --video "real/Man Excersing" --step 3
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LABELS = ("real", "fake")
GARMENT_COLORS = {
    "upper": (255, 140, 0),    # BGR: bright blue
    "lower": (0, 140, 255),    # BGR: bright orange
}
FADED_ALPHA = 0.35
POINT_RADIUS = 5
FRAME_GLOB = "*.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save annotated tracking frames as individual images.")
    parser.add_argument("--frames", type=Path, default=Path("frames"))
    parser.add_argument("--trajectories", type=Path, default=Path("trajectories"))
    parser.add_argument("--output", type=Path, default=Path("frame_review"))
    parser.add_argument("--video", type=str, required=True, help='e.g. "real/Man Excersing"')
    parser.add_argument("--step", type=int, default=1, help="Save every Nth frame (default: 1, all frames).")
    return parser.parse_args()


def list_frame_files(video_dir: Path) -> list[Path]:
    frame_files = [p for p in video_dir.glob(FRAME_GLOB) if p.is_file()]
    return sorted(frame_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def blend_color(color_bgr: tuple, alpha: float) -> tuple:
    gray = 160
    return tuple(int(c * alpha + gray * (1 - alpha)) for c in color_bgr)


def draw_points_on_frame(frame: np.ndarray, garment_data: dict, garment: str, frame_idx: int) -> None:
    """garment_data is now {"segments": [{"seed_frame": N, "points": [...]}, ...]}.
    Each trajectory entry stores its actual video frame number, so we just look
    up the matching entry directly rather than needing to pick "the active segment" --
    a point simply has no entry for frames outside its own segment's range."""
    color_full = GARMENT_COLORS[garment]
    color_faded = blend_color(color_full, FADED_ALPHA)

    for segment in garment_data.get("segments", []):
        seed_frame = segment["seed_frame"]
        for point in segment["points"]:
            traj = point["trajectory"]
            # trajectory entries are keyed by actual frame number (frame_offset applied
            # at build time), so find the entry matching this frame_idx if present.
            match = next((e for e in traj if e["frame"] == frame_idx), None)
            if match is None:
                continue
            x, y = int(round(match["x"])), int(round(match["y"]))
            visible = match["visible"]
            color = color_full if visible else color_faded
            cv2.circle(frame, (x, y), POINT_RADIUS, color, thickness=-1, lineType=cv2.LINE_AA)
            if visible:
                cv2.circle(frame, (x, y), POINT_RADIUS + 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            # Label with seed_frame.point_id so points from different re-seed
            # groups are distinguishable (point IDs reset at each re-seed).
            cv2.putText(
                frame, f"{seed_frame}.{point['point_id']}", (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA,
            )


def main() -> None:
    args = parse_args()

    if "/" not in args.video:
        raise SystemExit('ERROR: --video must be in "label/video_name" format, e.g. "real/Man Excersing"')
    label, video_name = args.video.split("/", 1)

    frames_root = args.frames.resolve()
    trajectories_root = args.trajectories.resolve()
    output_root = args.output.resolve()

    video_dir = frames_root / label / video_name
    traj_path = trajectories_root / label / f"{video_name}.json"

    if not video_dir.is_dir():
        raise SystemExit(f"ERROR: Frames folder not found: {video_dir}")
    if not traj_path.is_file():
        raise SystemExit(f"ERROR: Trajectory file not found: {traj_path}")

    with traj_path.open("r", encoding="utf-8") as f:
        traj_data = json.load(f)

    frame_files = list_frame_files(video_dir)
    if not frame_files:
        raise SystemExit(f"ERROR: No frames found in {video_dir}")

    out_dir = output_root / label / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for frame_idx, frame_path in enumerate(frame_files):
        if frame_idx % args.step != 0:
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        for garment in ("upper", "lower"):
            if garment in traj_data:
                draw_points_on_frame(frame, traj_data[garment], garment, frame_idx)

        cv2.putText(
            frame, f"frame {frame_idx}", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

        out_path = out_dir / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    print(f"Saved {saved} annotated frame(s) to: {out_dir}")
    print("Blue = upper garment, Orange = lower garment.")
    print("Bright/solid = tracked this frame. Faded/dim = interpolated (tracking failed that frame).")
    print("Each point is labeled with its point_id so you can follow a specific point across images.")


if __name__ == "__main__":
    main()