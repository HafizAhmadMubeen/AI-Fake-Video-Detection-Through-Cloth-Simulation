"""
Visualize tracked garment points overlaid on video frames, saved as an
annotated .mp4 per video, so tracking quality can be checked by eye.

Point coloring:
  - Upper garment points: shades of blue
  - Lower garment points: shades of orange
  - Bright/solid color  = this frame was genuinely tracked (visible: true)
  - Faded/dim color     = this frame's position was interpolated (visible: false)

Example usage:
  # Visualize one specific video
  python visualize_trajectories.py --frames frames --trajectories trajectories --output visualizations --video fake/man_dancing

  # Visualize every video in the dataset
  python visualize_trajectories.py --frames frames --trajectories trajectories --output visualizations --all
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LABELS = ("real", "fake")
GARMENT_COLORS = {
    "upper": (255, 140, 0),    # blue-ish (BGR): bright blue
    "lower": (0, 140, 255),    # orange-ish (BGR): bright orange
}
FADED_ALPHA = 0.35  # how dim interpolated points look relative to tracked ones
POINT_RADIUS = 4
FRAME_GLOB = "*.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize tracked point trajectories.")
    parser.add_argument("--frames", type=Path, default=Path("frames"))
    parser.add_argument("--trajectories", type=Path, default=Path("trajectories"))
    parser.add_argument("--output", type=Path, default=Path("visualizations"))
    parser.add_argument(
        "--video", type=str, default=None,
        help='Specific video to visualize, as "label/video_name" (e.g. "fake/man dancing"). '
             "If omitted, use --all instead.",
    )
    parser.add_argument("--all", action="store_true", help="Visualize every video found.")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video fps (default: 10).")
    return parser.parse_args()


def list_frame_files(video_dir: Path) -> list[Path]:
    frame_files = [p for p in video_dir.glob(FRAME_GLOB) if p.is_file()]
    return sorted(frame_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def discover_videos(trajectories_root: Path) -> list[tuple[str, str]]:
    videos = []
    for label in LABELS:
        label_dir = trajectories_root / label
        if not label_dir.is_dir():
            continue
        for json_path in sorted(label_dir.glob("*.json")):
            videos.append((label, json_path.stem))
    return videos


def blend_color(color_bgr: tuple, alpha: float) -> tuple:
    """Fade a color toward gray for interpolated (less-trusted) points."""
    gray = 160
    return tuple(int(c * alpha + gray * (1 - alpha)) for c in color_bgr)


def draw_points_on_frame(
    frame: np.ndarray,
    garment_data: dict,
    garment: str,
    frame_idx: int,
) -> None:
    color_full = GARMENT_COLORS[garment]
    color_faded = blend_color(color_full, FADED_ALPHA)

    for point in garment_data["points"]:
        traj = point["trajectory"]
        if frame_idx >= len(traj):
            continue
        entry = traj[frame_idx]
        x, y = int(round(entry["x"])), int(round(entry["y"]))
        visible = entry["visible"]
        color = color_full if visible else color_faded
        cv2.circle(frame, (x, y), POINT_RADIUS, color, thickness=-1, lineType=cv2.LINE_AA)
        if visible:
            cv2.circle(frame, (x, y), POINT_RADIUS + 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


def visualize_video(
    frames_root: Path,
    trajectories_root: Path,
    output_root: Path,
    label: str,
    video_name: str,
    fps: float,
) -> None:
    video_dir = frames_root / label / video_name
    traj_path = trajectories_root / label / f"{video_name}.json"

    if not video_dir.is_dir():
        print(f"  WARNING: Frames folder not found, skipping: {video_dir}")
        return
    if not traj_path.is_file():
        print(f"  WARNING: Trajectory file not found, skipping: {traj_path}")
        return

    with traj_path.open("r", encoding="utf-8") as f:
        traj_data = json.load(f)

    frame_files = list_frame_files(video_dir)
    if not frame_files:
        print(f"  WARNING: No frames found, skipping: {video_dir}")
        return

    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"  WARNING: Could not read first frame, skipping: {video_dir}")
        return
    h, w = first_frame.shape[:2]

    output_path = output_root / label / f"{video_name}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame_idx, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        for garment in ("upper", "lower"):
            if garment in traj_data:
                draw_points_on_frame(frame, traj_data[garment], garment, frame_idx)

        cv2.putText(
            frame, f"frame {frame_idx}", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    print(f"  Saved: {output_path}")


def main() -> None:
    args = parse_args()

    frames_root = args.frames.resolve()
    trajectories_root = args.trajectories.resolve()
    output_root = args.output.resolve()

    if args.video:
        if "/" not in args.video:
            raise SystemExit('ERROR: --video must be in "label/video_name" format, e.g. "fake/man dancing"')
        label, video_name = args.video.split("/", 1)
        videos = [(label, video_name)]
    elif args.all:
        videos = discover_videos(trajectories_root)
        if not videos:
            raise SystemExit(f"ERROR: No trajectory files found under {trajectories_root}")
    else:
        raise SystemExit("ERROR: Specify either --video <label/video_name> or --all")

    print(f"Frames root      : {frames_root}")
    print(f"Trajectories root: {trajectories_root}")
    print(f"Output root      : {output_root}")
    print(f"Videos to process: {len(videos)}")
    print("-" * 60)

    for label, video_name in videos:
        print(f"Processing: {label}/{video_name}")
        visualize_video(frames_root, trajectories_root, output_root, label, video_name, args.fps)

    print("\nDone. Open the .mp4 files in visualizations/ to inspect tracking quality.")
    print("Bright dots = genuinely tracked. Faded/gray dots = interpolated (tracking failed that frame).")


if __name__ == "__main__":
    main()
