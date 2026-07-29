"""
Phase 3, step 2: Track garment seed points across every frame with CoTracker3 (online).

Online mode processes the video in small sliding windows instead of holding a full
correlation volume across the whole clip in memory at once, which is what offline
mode requires. This project switched to online mode because offline mode did not
fit reliably in 8GB VRAM, even with the smaller baseline checkpoint.

CoTracker3 setup (run once before first use)
--------------------------------------------
Recommended install from the official repo (needed for --checkpoint and custom queries):

  git clone https://github.com/facebookresearch/co-tracker
  cd co-tracker
  pip install -e .

PyTorch with CUDA is strongly recommended. Ensure torch/torchvision are installed first:
  https://pytorch.org/get-started/locally/

Download the online CoTracker3 checkpoint (baseline, lighter weight):

  mkdir -p checkpoints
  cd checkpoints
  curl -L -o baseline_online.pth \\
    https://huggingface.co/facebook/cotracker3/resolve/main/baseline_online.pth
  cd ..

Example invocation from the project root:

  python track_points.py \\
    --frames frames \\
    --seed-points seed_points \\
    --output trajectories \\
    --checkpoint /path/to/co-tracker/checkpoints/baseline_online.pth
"""

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from cotracker.predictor import CoTrackerPredictor

LABELS = ("real", "fake")
GARMENTS = ("upper", "lower")
FRAME_GLOB = "*.jpg"
FAILURES_LOG = "phase3_failures.txt"
ONLINE_WINDOW_LEN = 16  # smaller sliding window than offline mode, bounds VRAM use


def release_video_memory() -> None:
    """Return cached CPU/GPU memory between per-video batch iterations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track garment seed points across frames with CoTracker3 (offline)."
    )
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("frames"),
        help='Path to extracted frames root (default: "frames").',
    )
    parser.add_argument(
        "--seed-points",
        type=Path,
        default=Path("seed_points"),
        help='Root folder for seed-point JSON files (default: "seed_points").',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trajectories"),
        help='Output root for trajectory JSON files (default: "trajectories").',
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to CoTracker3 offline checkpoint (e.g. scaled_offline.pth).",
    )
    return parser.parse_args()


def discover_videos(seed_points_root: Path) -> list[tuple[str, str]]:
    """Return sorted (label, video_name) pairs from seed_points/<label>/*.json."""
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
    frame_files = [path for path in video_dir.glob(FRAME_GLOB) if path.is_file()]
    return sorted(
        frame_files,
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def load_video_frames(video_dir: Path) -> np.ndarray:
    """Load all frames as an RGB uint8 array with shape (T, H, W, 3)."""
    frame_files = list_frame_files(video_dir)
    if not frame_files:
        raise FileNotFoundError(f"No frame images found in {video_dir}")

    frames: list[np.ndarray] = []
    for frame_path in frame_files:
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            raise IOError(f"Could not read frame: {frame_path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    return np.stack(frames, axis=0)


def frames_to_tensor(frames_rgb: np.ndarray, device: str) -> torch.Tensor:
    """Convert (T, H, W, 3) RGB uint8 frames to (1, T, 3, H, W) float tensor."""
    video = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)[None].float()
    return video.to(device)


def load_seed_points(path: Path) -> dict[str, list[list[float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Seed points file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def seed_points_to_queries(
    points: list[list[float]],
    query_frame: int,
    device: str,
) -> torch.Tensor:
    """Build CoTracker queries tensor (B, N, 3) in (t, x, y) format."""
    if not points:
        raise ValueError("Cannot track an empty point list")

    queries = torch.zeros((1, len(points), 3), dtype=torch.float32, device=device)
    queries[:, :, 0] = float(query_frame)
    for idx, (x, y) in enumerate(points):
        queries[0, idx, 1] = float(x)
        queries[0, idx, 2] = float(y)
    return queries


def interpolate_occluded_positions(
    xs: np.ndarray,
    ys: np.ndarray,
    visible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill occluded positions with linear interpolation between visible frames.

    Edge occlusions (no visible frame on one side) hold the nearest visible
    position constant. Original visibility flags are unchanged by the caller.
    """
    xs_out = xs.astype(np.float64, copy=True)
    ys_out = ys.astype(np.float64, copy=True)
    visible = visible.astype(bool, copy=False)

    vis_indices = np.flatnonzero(visible)
    if vis_indices.size == 0:
        return xs_out, ys_out

    for left, right in zip(vis_indices[:-1], vis_indices[1:], strict=False):
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
        xs_out[last_vis + 1 :] = xs[last_vis]
        ys_out[last_vis + 1 :] = ys[last_vis]

    return xs_out, ys_out


def build_point_trajectories(
    tracks: np.ndarray,
    visibility: np.ndarray,
) -> list[dict]:
    """
    Convert CoTracker outputs to the project JSON schema with occlusion fill-in.

    tracks: (T, N, 2) pixel coordinates (x, y)
    visibility: (T, N) bool — CoTracker3 offline threshold at 0.9
    """
    num_frames, num_points, _ = tracks.shape
    points_out: list[dict] = []

    for point_id in range(num_points):
        xs = tracks[:, point_id, 0]
        ys = tracks[:, point_id, 1]
        vis = visibility[:, point_id]

        xs_filled, ys_filled = interpolate_occluded_positions(xs, ys, vis)

        trajectory = []
        for frame_idx in range(num_frames):
            trajectory.append(
                {
                    "frame": frame_idx,
                    "x": float(xs_filled[frame_idx]),
                    "y": float(ys_filled[frame_idx]),
                    "visible": bool(vis[frame_idx]),
                }
            )

        points_out.append({"point_id": point_id, "trajectory": trajectory})

    return points_out


@torch.inference_mode()
def track_garment_points(
    model: CoTrackerPredictor,
    video: torch.Tensor,
    seed_points: list[list[float]],
    query_frame: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one offline CoTracker pass for a single garment point set."""
    queries = seed_points_to_queries(seed_points, query_frame, device=str(video.device))
    tracks, visibility = model(
        video,
        queries=queries,
        grid_size=0,
        grid_query_frame=query_frame,
    )
    tracks_np = tracks[0].detach().cpu().numpy()
    visibility_np = visibility[0].detach().cpu().numpy().squeeze(-1).astype(bool)
    del queries, tracks, visibility
    return tracks_np, visibility_np


def process_video(
    model: CoTrackerPredictor,
    device: str,
    frames_root: Path,
    seed_points_root: Path,
    output_root: Path,
    label: str,
    video_name: str,
) -> dict[str, dict]:
    """Track upper/lower seed points and return the trajectory payload."""
    video_dir = frames_root / label / video_name
    seed_path = seed_points_root / label / f"{video_name}.json"
    seed_data = None
    frames_rgb = None
    video = None
    try:
        seed_data = load_seed_points(seed_path)
        frames_rgb = load_video_frames(video_dir)
        video = frames_to_tensor(frames_rgb, device)

        num_frames = frames_rgb.shape[0]
        result: dict[str, dict] = {}
        stats: dict[str, tuple[int, int]] = {}

        for garment in GARMENTS:
            points = seed_data.get(garment)
            if not isinstance(points, list) or not points:
                print(f"  WARNING: No {garment} seed points, skipping garment")
                continue

            print(f"  Tracking {garment}: {len(points)} point(s)...")
            garment_started = time.perf_counter()
            tracks, visibility = track_garment_points(model, video, points, query_frame=0)
            garment_elapsed = time.perf_counter() - garment_started

            if tracks.shape[0] != num_frames:
                raise RuntimeError(
                    f"{garment} track length {tracks.shape[0]} != frame count {num_frames}"
                )

            result[garment] = {"points": build_point_trajectories(tracks, visibility)}
            stats[garment] = (len(points), num_frames)
            del tracks, visibility
            print(
                f"    {garment}: {len(points)} point(s), {num_frames} frame(s), "
                f"{garment_elapsed:.1f}s"
            )

        if not result:
            raise ValueError(f"No garment trajectories produced for [{label}] {video_name}")

        output_path = output_root / label / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")

        print(f"  Saved: {output_path}")
        return result
    finally:
        del seed_data, frames_rgb, video
        release_video_memory()


def log_failure(failures_path: Path, video_key: str, error: Exception) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with failures_path.open("a", encoding="utf-8") as f:
        f.write(f"{video_key}\t{type(error).__name__}: {error}\n")


def main() -> None:
    args = parse_args()

    frames_root = args.frames.resolve()
    seed_points_root = args.seed_points.resolve()
    output_root = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    failures_path = Path(FAILURES_LOG)

    if not frames_root.is_dir():
        raise SystemExit(f"ERROR: Frames folder is not a directory: {frames_root}")
    if not seed_points_root.is_dir():
        raise SystemExit(f"ERROR: Seed-points folder is not a directory: {seed_points_root}")
    if not checkpoint.is_file():
        raise SystemExit(f"ERROR: Checkpoint not found: {checkpoint}")

    videos = discover_videos(seed_points_root)
    if not videos:
        raise SystemExit(f"ERROR: No seed-point JSON files found under {seed_points_root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available; CoTracker offline inference on CPU will be very slow.")

    print(f"Frames root     : {frames_root}")
    print(f"Seed points root: {seed_points_root}")
    print(f"Output root     : {output_root}")
    print(f"Checkpoint      : {checkpoint}")
    print(f"Device          : {device}")
    print(f"Videos found    : {len(videos)}")
    print("-" * 60)

    model = CoTrackerPredictor(
        checkpoint=str(checkpoint),
        offline=False,
        window_len=ONLINE_WINDOW_LEN,
    ).to(device)
    model.eval()

    successful = 0
    failed = 0

    for label, video_name in videos:
        video_key = f"{label}/{video_name}"
        print(f"\nProcessing: {video_key}")
        started = time.perf_counter()
        try:
            process_video(
                model,
                device,
                frames_root,
                seed_points_root,
                output_root,
                label,
                video_name,
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
        finally:
            release_video_memory()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Successful : {successful}/{len(videos)}")
    print(f"  Failed     : {failed}")
    if failed:
        print(f"  Failures logged to: {failures_path.resolve()}")


if __name__ == "__main__":
    main()
