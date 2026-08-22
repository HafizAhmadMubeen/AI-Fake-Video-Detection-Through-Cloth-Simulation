"""
Phase 3, step 1 (reworked): Sample garment tracking points at MULTIPLE frames
throughout the video, not just frame 0.

Why this changed: seeding only at frame 0 means points are permanently lost
once the body rotates them out of camera view (confirmed via visual review -
e.g. "Man Excersing", where the front-facing shirt points became invisible
for the rest of the clip once the person turned). Re-seeding periodically
means fresh points get picked on whatever garment surface IS visible at that
point in the video, so tracking recovers after a rotation instead of leaving
a frozen/interpolated gap for the rest of the clip.

Output structure per video:
{
  "upper": {"0": [[x,y],...], "15": [[x,y],...], "30": [[x,y],...], ...},
  "lower": {"0": [[x,y],...], "15": [[x,y],...], ...}
}
Each key is the seed frame number (as a string, since JSON object keys must
be strings), mapping to the grid of points sampled from that frame's mask.

Usage:
  python sample_seed_points.py --masks masks --output seed_points --grid-size 8 --reseed-interval 15
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LABELS = ("real", "fake")
GARMENTS = ("upper", "lower")
MASK_GLOB = "*.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample garment seed points at multiple frames.")
    parser.add_argument("--masks", type=Path, default=Path("masks"))
    parser.add_argument("--output", type=Path, default=Path("seed_points"))
    parser.add_argument("--grid-size", type=int, default=8, help="Grid density per re-seed (default: 8x8).")
    parser.add_argument(
        "--reseed-interval", type=int, default=15,
        help="Re-seed a fresh grid of points every N frames (default: 15).",
    )
    parser.add_argument(
        "--min-mask-pixels", type=int, default=100,
        help="Minimum foreground pixels required to sample from a mask; smaller grids fall back automatically.",
    )
    return parser.parse_args()


def discover_videos(masks_root: Path) -> list[tuple[str, str]]:
    """Return sorted (label, video_name) pairs found under masks/upper/<label>/<video>/."""
    videos = []
    upper_root = masks_root / "upper"
    for label in LABELS:
        label_dir = upper_root / label
        if not label_dir.is_dir():
            continue
        for video_dir in sorted(label_dir.iterdir()):
            if video_dir.is_dir():
                videos.append((label, video_dir.name))
    return videos


def load_mask(masks_root: Path, garment: str, label: str, video_name: str, frame_idx: int) -> np.ndarray | None:
    mask_dir = masks_root / garment / label / video_name
    for candidate in (mask_dir / f"frame_{frame_idx:04d}.png", mask_dir / f"{frame_idx:04d}.png"):
        if candidate.is_file():
            m = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return m
    return None


def count_frames(masks_root: Path, garment: str, label: str, video_name: str) -> int:
    mask_dir = masks_root / garment / label / video_name
    if not mask_dir.is_dir():
        return 0
    return len(list(mask_dir.glob(MASK_GLOB)))


def sample_grid_in_mask(mask: np.ndarray, grid_size: int, min_mask_pixels: int) -> list[list[float]]:
    """Sample an evenly spaced grid_size x grid_size grid of points inside the mask's bounding box,
    keeping only points that fall on foreground pixels. Falls back to a sparser grid if too few
    foreground pixels exist to support the requested density."""
    ys, xs = np.where(mask > 0)
    if len(xs) < min_mask_pixels:
        return []

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    def try_grid(size: int) -> list[list[float]]:
        pts = []
        for i in range(size):
            for j in range(size):
                gx = x_min + (x_max - x_min) * (i + 0.5) / size
                gy = y_min + (y_max - y_min) * (j + 0.5) / size
                xi, yi = int(round(gx)), int(round(gy))
                if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi] > 0:
                    pts.append([float(xi), float(yi)])
        return pts

    points = try_grid(grid_size)
    if len(points) < (grid_size * grid_size) // 4 and grid_size > 4:
        # Sparse mask region -- fall back to a coarser grid rather than too few/clustered points
        points = try_grid(max(4, grid_size // 2))
    return points


def main() -> None:
    args = parse_args()

    masks_root = args.masks.resolve()
    output_root = args.output.resolve()

    videos = discover_videos(masks_root)
    if not videos:
        raise SystemExit(f"ERROR: No videos found under {masks_root}/upper/<label>/")

    print(f"Masks root       : {masks_root}")
    print(f"Output root      : {output_root}")
    print(f"Grid size        : {args.grid_size}x{args.grid_size}")
    print(f"Re-seed interval : every {args.reseed_interval} frames")
    print(f"Videos found     : {len(videos)}")
    print("-" * 60)

    for label, video_name in videos:
        print(f"\nProcessing: {label}/{video_name}")
        result: dict[str, dict[str, list]] = {}

        for garment in GARMENTS:
            num_frames = count_frames(masks_root, garment, label, video_name)
            if num_frames == 0:
                print(f"  WARNING: No {garment} masks found, skipping garment")
                continue

            seed_frames = list(range(0, num_frames, args.reseed_interval))
            garment_result: dict[str, list] = {}

            for seed_frame in seed_frames:
                mask = load_mask(masks_root, garment, label, video_name, seed_frame)
                if mask is None:
                    print(f"  WARNING: {garment} mask missing at frame {seed_frame}, skipping this re-seed point")
                    continue
                points = sample_grid_in_mask(mask, args.grid_size, args.min_mask_pixels)
                if not points:
                    print(f"  WARNING: {garment} mask at frame {seed_frame} too small, no points sampled")
                    continue
                garment_result[str(seed_frame)] = points

            result[garment] = garment_result
            total_points = sum(len(v) for v in garment_result.values())
            print(f"  {garment}: {len(garment_result)} re-seed group(s), {total_points} total point(s)")

        output_path = output_root / label / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()