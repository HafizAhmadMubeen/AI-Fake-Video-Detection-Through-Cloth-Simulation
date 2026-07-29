"""
Phase 3, step 1: Sample seed tracking points from garment masks on the first valid frame.

For each video, reads upper/lower SAM2 masks, lays an evenly spaced grid over each
mask's bounding box, and keeps only grid points that fall on foreground pixels.
Outputs one JSON file per video for use as CoTracker seed points in the next step.
"""

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np

LABELS = ("real", "fake")
GARMENTS = ("upper", "lower")
MIN_FOREGROUND_PIXELS = 100
FRAME_NAME_FMT = "frame_{:04d}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample grid seed points from upper/lower garment masks."
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=Path("masks"),
        help='Root folder for mask PNGs (default: "masks").',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("seed_points"),
        help='Output root for seed-point JSON files (default: "seed_points").',
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="Grid dimension N for an NxN sample layout (default: 8).",
    )
    return parser.parse_args()


def list_video_folders(masks_root: Path, garment: str, label: str) -> list[Path]:
    garment_dir = masks_root / garment / label
    if not garment_dir.is_dir():
        print(f"WARNING: Mask folder not found, skipping: {garment_dir}")
        return []
    folders = sorted(path for path in garment_dir.iterdir() if path.is_dir())
    if not folders:
        print(f"No video folders found in {garment_dir}")
    return folders


def discover_videos(masks_root: Path) -> list[tuple[str, str]]:
    """Return sorted (label, video_name) pairs found under masks/upper/."""
    seen: set[tuple[str, str]] = set()
    for label in LABELS:
        for video_dir in list_video_folders(masks_root, "upper", label):
            seen.add((label, video_dir.name))
    return sorted(seen)


def list_mask_frames(mask_dir: Path) -> list[Path]:
    if not mask_dir.is_dir():
        return []
    frame_files = [path for path in mask_dir.glob("frame_*.png") if path.is_file()]
    return sorted(
        frame_files,
        key=lambda path: int(path.stem.split("_", 1)[1]),
    )


def load_first_valid_mask(mask_dir: Path) -> tuple[np.ndarray | None, str | None]:
    """
    Load the mask for frame 0, or the first frame whose mask has foreground pixels.

    Returns (mask_array, frame_stem) or (None, None) when no usable mask exists.
    """
    frame_files = list_mask_frames(mask_dir)
    if not frame_files:
        return None, None

    preferred = mask_dir / FRAME_NAME_FMT.format(0)
    ordered: list[Path] = []
    if preferred in frame_files:
        ordered.append(preferred)
    ordered.extend(path for path in frame_files if path != preferred)

    for path in ordered:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if np.count_nonzero(mask > 0) > 0:
            return mask, path.stem

    return None, None


def effective_grid_size(grid_size: int, foreground_count: int) -> int:
    """Use half density when the mask is very small."""
    if foreground_count < MIN_FOREGROUND_PIXELS:
        reduced = max(2, grid_size // 2)
        return reduced
    return grid_size


def sample_grid_points(
    mask: np.ndarray,
    grid_size: int,
) -> tuple[list[list[int]], int]:
    """
    Sample an NxN grid of (x, y) points inside the mask foreground.

    Computes the mask bounding box, places cell centers on a regular grid over
    that box, and keeps only centers that lie on a foreground pixel. Grid cells
    whose center falls outside the mask are skipped (not snapped to a nearby
    pixel) — simpler and avoids pulling points onto mask edges arbitrarily.
    """
    foreground = mask > 0
    fg_count = int(np.count_nonzero(foreground))
    if fg_count == 0:
        return [], grid_size

    n = effective_grid_size(grid_size, fg_count)
    if n != grid_size:
        warnings.warn(
            f"Mask has only {fg_count} foreground pixels (< {MIN_FOREGROUND_PIXELS}); "
            f"reducing grid from {grid_size}x{grid_size} to {n}x{n}.",
            stacklevel=2,
        )

    ys, xs = np.where(foreground)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    height, width = mask.shape[:2]
    points: list[list[int]] = []

    for row in range(n):
        for col in range(n):
            if n == 1:
                x = (x_min + x_max) // 2
                y = (y_min + y_max) // 2
            else:
                x = int(round(x_min + (col + 0.5) * (x_max - x_min) / n))
                y = int(round(y_min + (row + 0.5) * (y_max - y_min) / n))

            if 0 <= x < width and 0 <= y < height and foreground[y, x]:
                points.append([x, y])

    return points, n


def process_garment(
    masks_root: Path,
    label: str,
    video_name: str,
    garment: str,
    grid_size: int,
) -> tuple[list[list[int]] | None, str | None]:
    """
    Sample seed points for one garment mask.

    Returns (points, source_frame_stem) or (None, None) when the mask is missing.
    """
    mask_dir = masks_root / garment / label / video_name
    if not mask_dir.is_dir():
        print(f"  WARNING: {garment} mask folder missing, skipping garment: {mask_dir}")
        return None, None

    mask, frame_stem = load_first_valid_mask(mask_dir)
    if mask is None:
        print(
            f"  WARNING: No valid {garment} mask found (missing or all empty): {mask_dir}"
        )
        return None, None

    points, effective_n = sample_grid_points(mask, grid_size)
    if effective_n != grid_size:
        print(
            f"  NOTE: {garment} used {effective_n}x{effective_n} grid "
            f"(source frame: {frame_stem})"
        )
    elif frame_stem != "frame_0000":
        print(f"  NOTE: {garment} used source frame {frame_stem} (frame_0000 unavailable/empty)")

    return points, frame_stem


def process_video(
    masks_root: Path,
    output_root: Path,
    label: str,
    video_name: str,
    grid_size: int,
) -> bool:
    """Sample upper/lower seed points and write JSON. Returns True if file was saved."""
    result: dict[str, list[list[int]]] = {}
    counts: dict[str, int | str] = {}

    for garment in GARMENTS:
        points, _frame_stem = process_garment(
            masks_root, label, video_name, garment, grid_size
        )
        if points is None:
            counts[garment] = "skipped"
            continue
        result[garment] = points
        counts[garment] = len(points)

    if not result:
        print(f"  Skipping video (no garment masks available): [{label}] {video_name}")
        return False

    output_path = output_root / label / f"{video_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    upper_count = counts.get("upper", "skipped")
    lower_count = counts.get("lower", "skipped")
    print(
        f"  Saved {output_path} — upper: {upper_count} point(s), "
        f"lower: {lower_count} point(s)"
    )
    return True


def main() -> None:
    args = parse_args()
    masks_root = args.masks.resolve()
    output_root = args.output.resolve()
    grid_size = args.grid_size

    if not masks_root.is_dir():
        raise SystemExit(f"ERROR: Masks folder is not a directory: {masks_root}")
    if grid_size < 1:
        raise SystemExit("ERROR: --grid-size must be >= 1.")

    videos = discover_videos(masks_root)
    if not videos:
        raise SystemExit(f"ERROR: No videos found under {masks_root / 'upper'}")

    print(f"Masks root  : {masks_root}")
    print(f"Output root : {output_root}")
    print(f"Grid size   : {grid_size}x{grid_size}")
    print(f"Videos found: {len(videos)}")
    print("-" * 60)

    saved = 0
    skipped = 0

    for label, video_name in videos:
        print(f"\nProcessing [{label}] {video_name}")
        if process_video(masks_root, output_root, label, video_name, grid_size):
            saved += 1
        else:
            skipped += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  JSON files written : {saved}")
    print(f"  Videos skipped     : {skipped}")


if __name__ == "__main__":
    main()
