"""
Phase 3, step 2 (reworked): Track garment points with CoTracker3 (offline),
supporting periodic re-seeding so points lost to rotation/occlusion get
replaced by fresh points on newly visible garment surface.

Each re-seed group is tracked separately from its own seed frame, and only
the segment up to the NEXT re-seed group's start (or end of video, for the
last group) is kept. This avoids stale points dragging on past their
re-seed boundary.

Output structure per video (note: SEGMENTED, different from the old
single-continuous-trajectory format):
{
  "upper": {
    "segments": [
      {"seed_frame": 0,  "points": [{"point_id": 0, "trajectory": [...]}, ...]},
      {"seed_frame": 15, "points": [...]},
      ...
    ]
  },
  "lower": { "segments": [...] }
}
Each trajectory entry has "frame" as the ACTUAL video frame number (not
relative to the segment), so segments can be understood on a shared timeline.

Also includes the adjustable visibility threshold fix (0.6 instead of
CoTracker's hardcoded 0.9, which was found to be overly conservative -
confirmed via debug_cotracker_output.py against real mask-containment data).

Example invocation:
  python track_points.py --frames frames --seed-points seed_points --output trajectories \
    --checkpoint cotracker_checkpoints/scaled_offline.pth
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cotracker.predictor import CoTrackerPredictor
from cotracker.models.core.model_utils import get_points_on_a_grid

LABELS = ("real", "fake")
GARMENTS = ("upper", "lower")
FRAME_GLOB = "*.jpg"
FAILURES_LOG = "phase3_failures.txt"
OFFLINE_WINDOW_LEN = 60
VISIBILITY_THRESHOLD = 0.6  # CoTracker's own predictor.py hardcodes 0.9, confirmed
# overly conservative -- see debug_cotracker_output.py findings.


class CoTrackerPredictorAdjustableThreshold(CoTrackerPredictor):
    """Identical to CoTrackerPredictor, except the hardcoded 0.9 visibility
    threshold inside _compute_sparse_tracks is replaced with VISIBILITY_THRESHOLD.
    Logic copied directly from facebookresearch/co-tracker predictor.py, with
    only the threshold value changed."""

    @torch.no_grad()
    def _compute_sparse_tracks(
        self, video, queries, segm_mask=None, grid_size=0,
        add_support_grid=False, grid_query_frame=0, backward_tracking=False,
    ):
        B, T, C, H, W = video.shape
        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear", align_corners=True)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        if queries is not None:
            B, N, D = queries.shape
            assert D == 3
            queries = queries.clone()
            queries[:, :, 1:] *= queries.new_tensor(
                [(self.interp_shape[1] - 1) / (W - 1), (self.interp_shape[0] - 1) / (H - 1)]
            )
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=video.device)
            if segm_mask is not None:
                segm_mask = F.interpolate(segm_mask, tuple(self.interp_shape), mode="nearest")
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]
            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts], dim=2
            ).repeat(B, 1, 1)

        if add_support_grid:
            grid_pts = get_points_on_a_grid(self.support_grid_size, self.interp_shape, device=video.device)
            grid_pts = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
            grid_pts = grid_pts.repeat(B, 1, 1)
            queries = torch.cat([queries, grid_pts], dim=1)

        tracks, visibilities, *_ = self.model.forward(video=video, queries=queries, iters=6)

        if backward_tracking:
            tracks, visibilities = self._compute_backward_tracks(video, queries, tracks, visibilities)
            if add_support_grid:
                queries[:, -self.support_grid_size**2:, 0] = T - 1

        if add_support_grid:
            tracks = tracks[:, :, : -self.support_grid_size**2]
            visibilities = visibilities[:, :, : -self.support_grid_size**2]

        visibilities = visibilities > VISIBILITY_THRESHOLD  # the one changed line

        for i in range(len(queries)):
            queries_t = queries[i, : tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))
            tracks[i, queries_t, arange] = queries[i, : tracks.size(2), 1:]
            visibilities[i, queries_t, arange] = True

        tracks *= tracks.new_tensor(
            [(W - 1) / (self.interp_shape[1] - 1), (H - 1) / (self.interp_shape[0] - 1)]
        )
        return tracks, visibilities


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track garment points (segmented re-seeding) with CoTracker3.")
    parser.add_argument("--frames", type=Path, default=Path("frames"))
    parser.add_argument("--seed-points", type=Path, default=Path("seed_points"))
    parser.add_argument("--output", type=Path, default=Path("trajectories"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def discover_videos(seed_points_root: Path) -> list[tuple[str, str]]:
    videos = []
    for label in LABELS:
        label_dir = seed_points_root / label
        if not label_dir.is_dir():
            continue
        for json_path in sorted(label_dir.glob("*.json")):
            videos.append((label, json_path.stem))
    return videos


def list_frame_files(video_dir: Path) -> list[Path]:
    if not video_dir.is_dir():
        return []
    frame_files = [p for p in video_dir.glob(FRAME_GLOB) if p.is_file()]
    return sorted(frame_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def load_video_frames(video_dir: Path) -> np.ndarray:
    frame_files = list_frame_files(video_dir)
    if not frame_files:
        raise FileNotFoundError(f"No frame images found in {video_dir}")
    frames = []
    for fp in frame_files:
        bgr = cv2.imread(str(fp))
        if bgr is None:
            raise IOError(f"Could not read frame: {fp}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def frames_to_tensor(frames_rgb: np.ndarray, device: str) -> torch.Tensor:
    video = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)[None].float()
    return video.to(device)


def load_seed_points(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def interpolate_within_segment(xs, ys, visible):
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


def build_segment_trajectories(seg_tracks, seg_visibility, frame_offset: int) -> list[dict]:
    """seg_tracks/seg_visibility: (segment_len, N, ...). frame_offset added to
    produce ACTUAL video frame numbers in the output, not segment-relative ones."""
    seg_len, num_points = seg_tracks.shape[0], seg_tracks.shape[1]
    points_out = []
    for point_id in range(num_points):
        xs = seg_tracks[:, point_id, 0]
        ys = seg_tracks[:, point_id, 1]
        vis = seg_visibility[:, point_id]
        xs_f, ys_f = interpolate_within_segment(xs, ys, vis)
        trajectory = [
            {
                "frame": frame_offset + i,
                "x": float(xs_f[i]),
                "y": float(ys_f[i]),
                "visible": bool(vis[i]),
            }
            for i in range(seg_len)
        ]
        points_out.append({"point_id": point_id, "trajectory": trajectory})
    return points_out


@torch.inference_mode()
def track_from_seed_frame(model, video, seed_points, query_frame: int):
    queries = torch.zeros((1, len(seed_points), 3), dtype=torch.float32, device=video.device)
    queries[:, :, 0] = float(query_frame)
    for idx, (x, y) in enumerate(seed_points):
        queries[0, idx, 1] = float(x)
        queries[0, idx, 2] = float(y)

    tracks, visibility = model(video, queries=queries, grid_size=0, grid_query_frame=query_frame)
    tracks_np = tracks[0].detach().cpu().numpy()
    vis_np = visibility[0].detach().cpu().numpy().astype(bool)
    if vis_np.ndim == 3 and vis_np.shape[-1] == 1:
        vis_np = vis_np.squeeze(-1)
    vis_np = vis_np.reshape(tracks_np.shape[0], tracks_np.shape[1])
    del queries, tracks, visibility
    return tracks_np, vis_np


def process_video(model, device, frames_root, seed_points_root, output_root, label, video_name):
    video_dir = frames_root / label / video_name
    seed_path = seed_points_root / label / f"{video_name}.json"

    seed_data = load_seed_points(seed_path)
    frames_rgb = load_video_frames(video_dir)
    num_frames = frames_rgb.shape[0]
    video = frames_to_tensor(frames_rgb, device)

    result = {}
    try:
        for garment in GARMENTS:
            garment_seeds = seed_data.get(garment)
            if not garment_seeds:
                print(f"  WARNING: No {garment} seed points, skipping garment")
                continue

            seed_frames = sorted(int(k) for k in garment_seeds.keys())
            segments = []

            for i, seed_frame in enumerate(seed_frames):
                points = garment_seeds[str(seed_frame)]
                next_seed = seed_frames[i + 1] if i + 1 < len(seed_frames) else num_frames
                if seed_frame >= num_frames:
                    continue

                print(f"  Tracking {garment} segment: seed_frame={seed_frame}, "
                      f"{len(points)} point(s), covers frames [{seed_frame}, {next_seed})...")
                started = time.perf_counter()
                tracks, visibility = track_from_seed_frame(model, video, points, query_frame=seed_frame)
                elapsed = time.perf_counter() - started

                seg_tracks = tracks[seed_frame:next_seed]
                seg_vis = visibility[seed_frame:next_seed]
                seg_points = build_segment_trajectories(seg_tracks, seg_vis, frame_offset=seed_frame)
                segments.append({"seed_frame": seed_frame, "points": seg_points})

                del tracks, visibility, seg_tracks, seg_vis
                release_memory()
                print(f"    done in {elapsed:.1f}s")

            if segments:
                result[garment] = {"segments": segments}

        if not result:
            raise ValueError(f"No garment trajectories produced for [{label}] {video_name}")

        output_path = output_root / label / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {output_path}")
    finally:
        del seed_data, frames_rgb, video
        release_memory()


def log_failure(failures_path, video_key, error):
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with failures_path.open("a", encoding="utf-8") as f:
        f.write(f"{video_key}\t{type(error).__name__}: {error}\n")


def main():
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
    if args.limit is not None:
        videos = videos[: args.limit]
        print(f"NOTE: --limit {args.limit} applied, processing {len(videos)} video(s) only.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Frames root     : {frames_root}")
    print(f"Seed points root: {seed_points_root}")
    print(f"Output root     : {output_root}")
    print(f"Checkpoint      : {checkpoint}")
    print(f"Device          : {device}")
    print(f"Videos found    : {len(videos)}")
    print("-" * 60)

    model = CoTrackerPredictorAdjustableThreshold(
        checkpoint=str(checkpoint), offline=True, window_len=OFFLINE_WINDOW_LEN,
    ).to(device)
    model.eval()

    successful, failed = 0, 0
    for label, video_name in videos:
        video_key = f"{label}/{video_name}"
        print(f"\nProcessing: {video_key}")
        started = time.perf_counter()
        try:
            process_video(model, device, frames_root, seed_points_root, output_root, label, video_name)
            print(f"  Total time: {time.perf_counter() - started:.1f}s")
            successful += 1
        except Exception as exc:
            failed += 1
            print(f"  ERROR after {time.perf_counter() - started:.1f}s: {exc}")
            traceback.print_exc()
            log_failure(failures_path, video_key, exc)
        finally:
            release_memory()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Successful : {successful}/{len(videos)}")
    print(f"  Failed     : {failed}")
    if failed:
        print(f"  Failures logged to: {failures_path.resolve()}")


if __name__ == "__main__":
    main()