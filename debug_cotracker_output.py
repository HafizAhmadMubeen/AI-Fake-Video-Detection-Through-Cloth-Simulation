"""
Debug script: inspect CoTracker's raw model output for ONE video, before any
of our post-processing (squeeze/reshape/astype) touches it.

Run this to figure out exactly what shape and value range CoTracker is
actually returning, since the QC numbers suggest our visibility extraction
may be misinterpreting the model's output.

Usage:
  python debug_cotracker_output.py --frames frames --seed-points seed_points --checkpoint cotracker_checkpoints/scaled_offline.pth --video real/Man boxing
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse
import sys
import traceback
from pathlib import Path

import cv2
import json
import numpy as np
import torch
from cotracker.predictor import CoTrackerPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("frames"))
    parser.add_argument("--seed-points", type=Path, default=Path("seed_points"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=str, required=True, help='e.g. "real/Man boxing"')
    parser.add_argument("--garment", type=str, default="upper", choices=["upper", "lower"])
    return parser.parse_args()


def list_frame_files(video_dir: Path):
    frame_files = [p for p in video_dir.glob("*.jpg") if p.is_file()]
    return sorted(frame_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def main():
    args = parse_args()
    label, video_name = args.video.split("/", 1)

    video_dir = args.frames / label / video_name
    frame_files = list_frame_files(video_dir)
    print(f"Loading {len(frame_files)} frames from {video_dir}")
    sys.stdout.flush()

    frames = []
    for fp in frame_files:
        bgr = cv2.imread(str(fp))
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    frames_rgb = np.stack(frames, axis=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    sys.stdout.flush()
    video = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)[None].float().to(device)

    seed_path = args.seed_points / label / f"{video_name}.json"
    with seed_path.open("r", encoding="utf-8") as f:
        seed_data = json.load(f)
    points = seed_data[args.garment]
    print(f"Loaded {len(points)} seed points for garment '{args.garment}'")
    sys.stdout.flush()

    print("Building CoTrackerPredictor (loading checkpoint)...")
    sys.stdout.flush()
    model = CoTrackerPredictor(checkpoint=str(args.checkpoint), offline=True, window_len=60).to(device)
    model.eval()
    print("Model loaded successfully.")
    sys.stdout.flush()

    queries = torch.zeros((1, len(points), 3), dtype=torch.float32, device=device)
    for idx, (x, y) in enumerate(points):
        queries[0, idx, 1] = float(x)
        queries[0, idx, 2] = float(y)

    print("Running model inference...")
    sys.stdout.flush()
    with torch.inference_mode():
        output = model(video, queries=queries, grid_size=0, grid_query_frame=0)
    print("Inference complete.")
    sys.stdout.flush()

    print("\n" + "=" * 60)
    print("RAW MODEL OUTPUT INSPECTION")
    print("=" * 60)

    if isinstance(output, tuple):
        print(f"Output is a tuple with {len(output)} element(s).")
        for i, item in enumerate(output):
            if torch.is_tensor(item):
                print(f"\n  Element {i}: tensor, shape={tuple(item.shape)}, dtype={item.dtype}")
                flat = item.detach().cpu().numpy().flatten()
                print(f"    min={flat.min():.4f} max={flat.max():.4f} mean={flat.mean():.4f}")
                print(f"    first 10 values: {flat[:10]}")
                unique_vals = np.unique(flat)
                if len(unique_vals) <= 10:
                    print(f"    unique values: {unique_vals}")
                else:
                    print(f"    {len(unique_vals)} unique values (not listing all)")
            else:
                print(f"\n  Element {i}: type={type(item)}, value={item}")
    else:
        print(f"Output is a single object of type {type(output)}")

    print("\n" + "=" * 60)
    print("If Element 1 (assumed visibility) has values mostly between 0 and 1")
    print("that are NOT just 0.0/1.0, it's a continuous score needing a threshold,")
    print("not something to cast directly with .astype(bool).")
    print("If Element 1's shape has T and N in a different order than Element 0")
    print("(tracks), that's a reshape/transpose bug source.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 60)
        print("EXCEPTION OCCURRED:")
        print("=" * 60)
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)