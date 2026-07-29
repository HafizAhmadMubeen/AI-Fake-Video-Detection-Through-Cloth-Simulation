"""
Phase 3, step 2 (alternate): Track garment seed points using classical
Lucas-Kanade optical flow instead of a deep-learning point tracker.

This replaces the CoTracker-based approach, which did not fit reliably in
8GB VRAM. Lucas-Kanade runs entirely on CPU via OpenCV, no model download,
no GPU required.

Example invocation from the project root:

  python track_points_lk.py --frames frames --seed-points seed_points --output trajectories
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

LABELS = ("real", "fake")
GARMENTS = ("upper", "lower")
FRAME_GLOB = "*.jpg"
FAILURES_LOG = "phase3_failures.txt"

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track garment seed points across frames with Lucas-Kanade optical flow."
    )
    parser.add_argument(
        "--frames", type=Path, default=Path("frames"),
        help='Path to extracted frames root (default: "frames").',
    )
    parser.add_argument(
        "--seed-points", type=Path, default=Path("seed_points"),
        help='Root folder for seed-point JSON files (default: "seed_points").',
    )
    parser.add_argument(
        "--output", type=Path, default=Path("trajectories"),
        help='Output root for trajectory JSON files (default: "trajectories").',
    )
    parser.add_argument(
        "--fb-threshold", type=float, default=2.0,
        help="Forward-backward error threshold in pixels (default: 2.0).",
    )
    return parser.parse_args()


def discover_videos(seed_points_root: Path) -> list[tuple[str, str]]:
    videos: list[tuple[str, str]] = []
    for label in LABELS:
        label_dir = seed_points_root / label
        if not label_dir.is_dir():
            print(f"WARNING: Seed-points label folder not found, skipping: {label_dir}")
            continue
        for json_path in sorted(label_dir.glob("*.json")):
            videos.append((label, json_path.stem))
    return videos


def list_frame_files(video_dir: Path) -> list[Path]:
    if not video_dir.is_dir():
        return []
    frame_files = [p for p in video_dir.glob(FRAME_GLOB) if p.is_file()]
    return sorted(
        frame_files,
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )


def load_video_frames_gray(video_dir: Path) -> list[np.ndarray]:
    """Load all frames as grayscale uint8 arrays, in frame order."""
    frame_files = list_frame_files(video_dir)
    if not frame_files:
        raise FileNotFoundError(f"No frame images found in {video_dir}")

    frames: list[np.ndarray] = []
    for frame_path in frame_files:
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            raise IOError(f"Could not read frame: {frame_path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return frames


def load_seed_points(path: Path) -> dict[str, list[list[float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Seed points file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def interpolate_occluded_positions(
    xs: np.ndarray, ys: np.ndarray, visible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill not-visible positions with linear interpolation between visible frames.
    Edge occlusions hold the nearest visible position constant.
    """
    xs_out = xs.astype(np.float64, copy=True)
    ys_out = ys.astype(np.float64, copy=True)
    visible = visible.astype(bool, copy=False)

    vis_indices = np.flatnonzero(visible)
    if vis_indices.size == 0:
        return xs_out, ys_out

    for left, right in zip(vis_indices[:-1], vis_indices[1:]):
        if right - left <= 1:
            continue
        gap = np.arange(left + 1, right)
        alpha = (gap - left) / (right - left)
        xs_out[gap] = xs[left] * (1.0 - alpha) + xs[right] * alpha
        ys_out[gap] = ys[left] * (1.0 - alpha) + ys[right] * alpha

    first_vis = int(vis_indices[0])
    if first_vis > 0:
        xs_out[:first_vis] = xs[first_vis]
        ys_out[:first_vis] = ys[first_vis]

    last_vis = int(vis_indices[-1])
    if last_vis < len(visible) - 1:
        xs_out[last_vis + 1:] = xs[last_vis]
        ys_out[last_vis + 1:] = ys[last_vis]

    return xs_out, ys_out


def track_garment_points_lk(
    frames_gray: list[np.ndarray],
    seed_points: list[list[float]],
    fb_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track points across all frames using pyramidal Lucas-Kanade, with a
    forward-backward consistency check to flag unreliable frames.

    Returns:
        tracks: (T, N, 2) float array of x, y positions (raw, pre-interpolation)
        visibility: (T, N) bool array (True = passed FB check / found)
    """
    num_frames = len(frames_gray)
    num_points = len(seed_points)

    tracks = np.zeros((num_frames, num_points, 2), dtype=np.float64)
    visibility = np.zeros((num_frames, num_points), dtype=bool)

    current_pts = np.array(seed_points, dtype=np.float32).reshape(-1, 1, 2)
    tracks[0] = current_pts.reshape(-1, 2)
    visibility[0] = True  # seed frame is always considered visible

    for t in range(num_frames - 1):
        prev_gray = frames_gray[t]
        next_gray = frames_gray[t + 1]

        next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray, current_pts, None, **LK_PARAMS
        )
        back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            next_gray, prev_gray, next_pts, None, **LK_PARAMS
        )

        fb_error = np.linalg.norm(
            (current_pts - back_pts).reshape(-1, 2), axis=1
        )

        found = (status_fwd.reshape(-1) == 1) & (status_bwd.reshape(-1) == 1)
        good = found & (fb_error <= fb_threshold)

        tracks[t + 1] = next_pts.reshape(-1, 2)
        visibility[t + 1] = good

        # Carry forward last known good position for points that failed this
        # step, so tracking can continue from a sane location next frame.
        bad_idx = ~good
        if np.any(bad_idx):
            next_pts[bad_idx] = current_pts[bad_idx]
            tracks[t + 1][bad_idx] = current_pts.reshape(-1, 2)[bad_idx]

        current_pts = next_pts

    return tracks, visibility


def build_point_trajectories(tracks: np.ndarray, visibility: np.ndarray) -> list[dict]:
    num_frames, num_points, _ = tracks.shape
    points_out: list[dict] = []

    for point_id in range(num_points):
        xs = tracks[:, point_id, 0]
        ys = tracks[:, point_id, 1]
        vis = visibility[:, point_id]

        xs_filled, ys_filled = interpolate_occluded_positions(xs, ys, vis)

        trajectory = [
            {
                "frame": frame_idx,
                "x": float(xs_filled[frame_idx]),
                "y": float(ys_filled[frame_idx]),
                "visible": bool(vis[frame_idx]),
            }
            for frame_idx in range(num_frames)
        ]
        points_out.append({"point_id": point_id, "trajectory": trajectory})

    return points_out


def process_video(
    frames_root: Path,
    seed_points_root: Path,
    output_root: Path,
    label: str,
    video_name: str,
    fb_threshold: float,
) -> None:
    video_dir = frames_root / label / video_name
    seed_path = seed_points_root / label / f"{video_name}.json"

    seed_data = load_seed_points(seed_path)
    frames_gray = load_video_frames_gray(video_dir)
    num_frames = len(frames_gray)

    result: dict[str, dict] = {}

    for garment in GARMENTS:
        points = seed_data.get(garment)
        if not isinstance(points, list) or not points:
            print(f"  WARNING: No {garment} seed points, skipping garment")
            continue

        print(f"  Tracking {garment}: {len(points)} point(s)...")
        started = time.perf_counter()
        tracks, visibility = track_garment_points_lk(frames_gray, points, fb_threshold)
        elapsed = time.perf_counter() - started

        if tracks.shape[0] != num_frames:
            raise RuntimeError(
                f"{garment} track length {tracks.shape[0]} != frame count {num_frames}"
            )

        num_interpolated = int((~visibility).sum())
        total_slots = visibility.size

        result[garment] = {"points": build_point_trajectories(tracks, visibility)}
        print(
            f"    {garment}: {len(points)} point(s), {num_frames} frame(s), "
            f"{elapsed:.1f}s, {num_interpolated}/{total_slots} slots interpolated"
        )

    if not result:
        raise ValueError(f"No garment trajectories produced for [{label}] {video_name}")

    output_path = output_root / label / f"{video_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"  Saved: {output_path}")


def log_failure(failures_path: Path, video_key: str, error: Exception) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with failures_path.open("a", encoding="utf-8") as f:
        f.write(f"{video_key}\t{type(error).__name__}: {error}\n")


def main() -> None:
    args = parse_args()

    frames_root = args.frames.resolve()
    seed_points_root = args.seed_points.resolve()
    output_root = args.output.resolve()
    failures_path = Path(FAILURES_LOG)

    if not frames_root.is_dir():
        raise SystemExit(f"ERROR: Frames folder is not a directory: {frames_root}")
    if not seed_points_root.is_dir():
        raise SystemExit(f"ERROR: Seed-points folder is not a directory: {seed_points_root}")

    videos = discover_videos(seed_points_root)
    if not videos:
        raise SystemExit(f"ERROR: No seed-point JSON files found under {seed_points_root}")

    print(f"Frames root     : {frames_root}")
    print(f"Seed points root: {seed_points_root}")
    print(f"Output root     : {output_root}")
    print(f"FB threshold    : {args.fb_threshold} px")
    print(f"Videos found    : {len(videos)}")
    print("-" * 60)

    successful = 0
    failed = 0

    for label, video_name in videos:
        video_key = f"{label}/{video_name}"
        print(f"\nProcessing: {video_key}")
        started = time.perf_counter()
        try:
            process_video(
                frames_root, seed_points_root, output_root,
                label, video_name, args.fb_threshold,
            )
            elapsed = time.perf_counter() - started
            print(f"  Total time: {elapsed:.1f}s")
            successful += 1
        except Exception as exc:
            elapsed = time.perf_counter() - started
            failed += 1
            print(f"  ERROR after {elapsed:.1f}s: {exc}")
            traceback.print_exc()
            log_failure(failures_path, video_key, exc)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Successful : {successful}/{len(videos)}")
    print(f"  Failed     : {failed}")
    if failed:
        print(f"  Failures logged to: {failures_path.resolve()}")


if __name__ == "__main__":
    main()
