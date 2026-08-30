"""
Save frames with upper/lower garment masks overlaid, as individual images,
for careful visual review of Phase 2 (SAM2) segmentation quality.

Coloring: upper garment mask = blue overlay, lower garment mask = orange
overlay, blended semi-transparently onto the original frame.

Usage:
  python visualize_masks.py --frames frames --masks masks --output mask_review --video "real/Man boxing"
  python visualize_masks.py --frames frames --masks masks --output mask_review --video "real/Man boxing" --step 5
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

UPPER_COLOR = (255, 140, 0)   # BGR: blue
LOWER_COLOR = (0, 140, 255)   # BGR: orange
ALPHA = 0.45
FRAME_GLOB = "*.jpg"


LABELS = ("real", "fake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SAM2 garment masks overlaid on frames.")
    parser.add_argument("--frames", type=Path, default=Path("frames"))
    parser.add_argument("--masks", type=Path, default=Path("masks"))
    parser.add_argument("--output", type=Path, default=Path("mask_review"))
    parser.add_argument("--video", type=str, default=None, help='e.g. "real/Man boxing". Omit if using --all.')
    parser.add_argument("--all", action="store_true", help="Process every video found under --frames.")
    parser.add_argument("--step", type=int, default=1, help="Save every Nth frame (default: 1, all frames).")
    return parser.parse_args()


def discover_videos(frames_root: Path) -> list[tuple[str, str]]:
    videos = []
    for label in LABELS:
        label_dir = frames_root / label
        if not label_dir.is_dir():
            continue
        for video_dir in sorted(label_dir.iterdir()):
            if video_dir.is_dir():
                videos.append((label, video_dir.name))
    return videos


def list_frame_files(video_dir: Path) -> list[Path]:
    frame_files = [p for p in video_dir.glob(FRAME_GLOB) if p.is_file()]
    return sorted(frame_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def load_mask(masks_root: Path, garment: str, label: str, video_name: str, frame_idx: int) -> np.ndarray | None:
    mask_dir = masks_root / garment / label / video_name
    for candidate in (mask_dir / f"frame_{frame_idx:04d}.png", mask_dir / f"{frame_idx:04d}.png"):
        if candidate.is_file():
            m = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return m
    return None


def overlay_mask(frame: np.ndarray, mask: np.ndarray | None, color: tuple, alpha: float) -> np.ndarray:
    if mask is None:
        return frame
    overlay = frame.astype(np.float32).copy()
    color_layer = np.zeros_like(overlay)
    color_layer[:, :] = color
    mask_bool = mask > 0
    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + color_layer[mask_bool] * alpha
    return overlay.astype(np.uint8)


def process_video(frames_root: Path, masks_root: Path, output_root: Path, label: str, video_name: str, step: int) -> None:
    video_dir = frames_root / label / video_name
    frame_files = list_frame_files(video_dir)
    if not frame_files:
        print(f"  WARNING: No frames found in {video_dir}, skipping")
        return

    out_dir = output_root / label / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    empty_upper = 0
    empty_lower = 0
    fullframe_upper = 0
    fullframe_lower = 0

    for frame_idx, frame_path in enumerate(frame_files):
        if frame_idx % step != 0:
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        upper_mask = load_mask(masks_root, "upper", label, video_name, frame_idx)
        lower_mask = load_mask(masks_root, "lower", label, video_name, frame_idx)

        total_px = frame.shape[0] * frame.shape[1]
        if upper_mask is not None:
            frac = int((upper_mask > 0).sum())
            if frac == 0:
                empty_upper += 1
            elif frac > 0.6 * total_px:
                fullframe_upper += 1
        if lower_mask is not None:
            frac = int((lower_mask > 0).sum())
            if frac == 0:
                empty_lower += 1
            elif frac > 0.6 * total_px:
                fullframe_lower += 1

        result = overlay_mask(frame, upper_mask, UPPER_COLOR, ALPHA)
        result = overlay_mask(result, lower_mask, LOWER_COLOR, ALPHA)
        cv2.putText(result, f"frame {frame_idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = out_dir / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(out_path), result)
        saved += 1

    print(f"  Saved {saved} frame(s) to {out_dir}")
    problems = []
    if empty_upper:
        problems.append(f"upper empty on {empty_upper} frame(s)")
    if empty_lower:
        problems.append(f"lower empty on {empty_lower} frame(s)")
    if fullframe_upper:
        problems.append(f"upper >60% of frame on {fullframe_upper} frame(s)")
    if fullframe_lower:
        problems.append(f"lower >60% of frame on {fullframe_lower} frame(s)")
    if problems:
        print(f"  WARNING: {'; '.join(problems)}")


def main() -> None:
    args = parse_args()

    frames_root = args.frames.resolve()
    masks_root = args.masks.resolve()
    output_root = args.output.resolve()

    if args.video:
        if "/" not in args.video:
            raise SystemExit('ERROR: --video must be "label/video_name", e.g. "real/Man boxing"')
        label, video_name = args.video.split("/", 1)
        videos = [(label, video_name)]
    elif args.all:
        videos = discover_videos(frames_root)
        if not videos:
            raise SystemExit(f"ERROR: No videos found under {frames_root}")
    else:
        raise SystemExit("ERROR: Specify either --video <label/video_name> or --all")

    print(f"Videos to process: {len(videos)}")
    print("-" * 60)

    for label, video_name in videos:
        print(f"\n{label}/{video_name}")
        process_video(frames_root, masks_root, output_root, label, video_name, args.step)

    print("\nDone. Blue = upper garment mask, Orange = lower garment mask.")
    print("Check the WARNING lines above for any video with empty or oversized masks.")


if __name__ == "__main__":
    main()