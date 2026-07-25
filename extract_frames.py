"""
Phase 1: Extract frames from video files for cloth-physics AI-detection research.

Reads videos from dataset/real/ and dataset/fake/, samples frames at a target FPS,
optionally resizes, and writes JPEGs to an output directory.
"""

import argparse
import os
from pathlib import Path

import cv2

# Supported video file extensions (matched case-insensitively).
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Fallback when OpenCV cannot read native FPS from a video.
DEFAULT_FPS = 30.0

# JPEG compression quality (0–100).
JPEG_QUALITY = 95


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frames from videos in dataset/real/ and dataset/fake/."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to dataset folder containing real/ and fake/ subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output folder for extracted frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Target frames per second to extract (default: 10.0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Resize frame width in pixels, preserving aspect ratio. "
        "Only applied when the frame is wider than this value. "
        "0 disables resizing (default: 512).",
    )
    return parser.parse_args()


def list_videos(folder: Path) -> list[Path]:
    """Return sorted list of video files in *folder* (non-recursive)."""
    if not folder.is_dir():
        return []
    videos = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return videos


def get_native_fps(cap: cv2.VideoCapture) -> float:
    """
    Read the video's native FPS from OpenCV.

    Falls back to DEFAULT_FPS when the value is missing or invalid.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps != fps:  # NaN check
        return DEFAULT_FPS
    return float(fps)


def resize_frame(frame, max_width: int):
    """
    Resize *frame* so its width is at most *max_width*, preserving aspect ratio.

    Returns the original frame unchanged when *max_width* is 0 or the frame
    is already narrow enough.
    """
    if max_width <= 0:
        return frame

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / width
    new_width = max_width
    new_height = int(round(height * scale))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    max_width: int,
) -> int:
    """
    Extract evenly sampled frames from a single video.

    Sampling uses the ratio native_fps / target_fps so frames are distributed
    across the full video duration rather than taken from the start only.

    Returns the number of frames saved, or 0 on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: Could not open video, skipping: {video_path}")
        return 0

    native_fps = get_native_fps(cap)
    if native_fps != cap.get(cv2.CAP_PROP_FPS):
        print(f"  NOTE: Could not read native FPS for {video_path.name}, "
              f"assuming {DEFAULT_FPS} fps.")

    # Frames between consecutive samples (may be fractional, e.g. 2.4 for 24→10 fps).
    frame_interval = native_fps / target_fps

    output_dir.mkdir(parents=True, exist_ok=True)
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample when we reach the next target frame index (rounded for stability).
        if frame_idx >= round(saved_count * frame_interval):
            frame = resize_frame(frame, max_width)
            out_name = f"{saved_count:04d}.jpg"
            out_path = output_dir / out_name
            if not cv2.imwrite(str(out_path), frame, jpeg_params):
                print(f"  WARNING: Failed to write frame: {out_path}")
            else:
                saved_count += 1

        frame_idx += 1

    cap.release()
    return saved_count


def process_label(
    input_root: Path,
    output_root: Path,
    label: str,
    target_fps: float,
    max_width: int,
) -> tuple[int, int]:
    """
    Process all videos for one label (real or fake).

    Returns (successful_count, total_video_count).
    """
    label_input = input_root / label
    if not label_input.is_dir():
        print(f"WARNING: Label folder not found, skipping: {label_input}")
        return 0, 0

    videos = list_videos(label_input)
    if not videos:
        print(f"No video files found in {label_input}")
        return 0, 0

    successful = 0
    for video_path in videos:
        # Strip leading/trailing whitespace: Windows silently drops trailing
        # spaces from path components, which can desync the folder actually
        # created on disk from the one OpenCV tries to write into, causing
        # every cv2.imwrite() call to fail silently.
        video_stem = video_path.stem.strip()
        output_dir = output_root / label / video_stem

        print(f"Processing [{label}] {video_path.name}")
        print(f"  Output: {output_dir}")

        saved = extract_frames_from_video(
            video_path, output_dir, target_fps, max_width
        )
        print(f"  Saved {saved} frame(s)")

        if saved > 0:
            successful += 1

    return successful, len(videos)


def main() -> None:
    args = parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.is_dir():
        raise SystemExit(f"ERROR: Input path is not a directory: {input_root}")

    if args.fps <= 0:
        raise SystemExit("ERROR: --fps must be greater than 0.")

    if args.width < 0:
        raise SystemExit("ERROR: --width must be >= 0.")

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Input dataset : {input_root}")
    print(f"Output folder : {output_root}")
    print(f"Target FPS    : {args.fps}")
    print(f"Max width     : {args.width if args.width > 0 else 'no resize'}")
    print("-" * 60)

    summary: dict[str, tuple[int, int]] = {}

    for label in ("real", "fake"):
        print(f"\n=== Label: {label} ===")
        successful, total = process_label(
            input_root, output_root, label, args.fps, args.width
        )
        summary[label] = (successful, total)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label in ("real", "fake"):
        successful, total = summary[label]
        if total == 0:
            print(f"  {label}: no videos found")
        else:
            print(f"  {label}: {successful}/{total} video(s) processed successfully")


if __name__ == "__main__":
    main()