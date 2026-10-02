"""
Batch version of the residual check: runs across every video, computes the
gap between simulated and observed points TWO ways:

1. RAW gap, in pixels -- same as quick_residual_check.py.
2. NORMALIZED gap -- the raw gap divided by that segment's garment size
   (the bounding-box diagonal of the observed points at the segment's seed
   frame). This accounts for the fact that a loose dress, a tight shirt, a
   close-up shot, and a far-away shot all have very different natural
   scales, so a raw pixel gap isn't a fair comparison across videos.

Prints a per-video table for both metrics, plus a real-vs-fake group
summary, and saves everything to residuals_comparison.csv so you can look
at it in a spreadsheet too.

Usage:
  python batch_residual_check.py --trajectories trajectories --simulated simulated_trajectories --output residuals_comparison.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

LABELS = ("real", "fake")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, default=Path("trajectories"))
    parser.add_argument("--simulated", type=Path, default=Path("simulated_trajectories"))
    parser.add_argument("--output", type=Path, default=Path("residuals_comparison.csv"))
    return parser.parse_args()


def discover_videos(simulated_root: Path):
    videos = []
    for label in LABELS:
        label_dir = simulated_root / label
        if not label_dir.is_dir():
            continue
        for json_path in sorted(label_dir.glob("*.json")):
            videos.append((label, json_path.stem))
    return videos


def build_lookup(garment_data):
    lookup = {}
    for segment in garment_data.get("segments", []):
        for point in segment["points"]:
            for entry in point["trajectory"]:
                lookup.setdefault(entry["frame"], {})[point["point_id"]] = (entry["x"], entry["y"])
    return lookup


def segment_scale(observed_points_in_segment) -> float:
    """Bounding-box diagonal of observed points at their first frame -- a
    simple, robust stand-in for 'how big is this garment on screen'."""
    xs, ys = [], []
    for point in observed_points_in_segment:
        first = point["trajectory"][0]
        xs.append(first["x"])
        ys.append(first["y"])
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    diag = (width ** 2 + height ** 2) ** 0.5
    return max(diag, 1.0)


def compute_video_gaps(obs_data: dict, sim_data: dict):
    raw_gaps = []
    norm_gaps = []

    for garment in obs_data:
        if garment not in sim_data:
            continue

        sim_lookup = build_lookup(sim_data[garment])

        for segment in obs_data[garment]["segments"]:
            scale = segment_scale(segment["points"])

            for point in segment["points"]:
                for entry in point["trajectory"]:
                    frame = entry["frame"]
                    point_id = point["point_id"]
                    if frame in sim_lookup and point_id in sim_lookup[frame]:
                        ox, oy = entry["x"], entry["y"]
                        sx, sy = sim_lookup[frame][point_id]
                        gap = ((ox - sx) ** 2 + (oy - sy) ** 2) ** 0.5
                        raw_gaps.append(gap)
                        norm_gaps.append(gap / scale)

    if not raw_gaps:
        return None, None
    return float(np.mean(raw_gaps)), float(np.mean(norm_gaps))


def main():
    args = parse_args()

    videos = discover_videos(args.simulated)
    if not videos:
        raise SystemExit(f"ERROR: No simulated videos found under {args.simulated}")

    rows = []
    print(f"{'Video':<38} {'Label':<6} {'Raw gap (px)':>14} {'Normalized gap':>16}")
    print("-" * 78)

    for label, video_name in videos:
        obs_path = args.trajectories / label / f"{video_name}.json"
        sim_path = args.simulated / label / f"{video_name}.json"
        if not obs_path.is_file() or not sim_path.is_file():
            continue

        with obs_path.open() as f:
            obs_data = json.load(f)
        with sim_path.open() as f:
            sim_data = json.load(f)

        raw_mean, norm_mean = compute_video_gaps(obs_data, sim_data)
        if raw_mean is None:
            continue

        rows.append({"video": video_name, "label": label, "raw_gap_px": raw_mean, "normalized_gap": norm_mean})
        print(f"{video_name:<38} {label:<6} {raw_mean:>14.2f} {norm_mean:>16.4f}")

    print("-" * 78)

    real_raw = [r["raw_gap_px"] for r in rows if r["label"] == "real"]
    fake_raw = [r["raw_gap_px"] for r in rows if r["label"] == "fake"]
    real_norm = [r["normalized_gap"] for r in rows if r["label"] == "real"]
    fake_norm = [r["normalized_gap"] for r in rows if r["label"] == "fake"]

    print("\nGroup summary:")
    print(f"  RAW gap        -- real: mean={np.mean(real_raw):.2f}, std={np.std(real_raw):.2f}"
          f"   |   fake: mean={np.mean(fake_raw):.2f}, std={np.std(fake_raw):.2f}")
    print(f"  NORMALIZED gap -- real: mean={np.mean(real_norm):.4f}, std={np.std(real_norm):.4f}"
          f"   |   fake: mean={np.mean(fake_norm):.4f}, std={np.std(fake_norm):.4f}")

    def separation(a, b):
        pooled_std = (np.std(a) + np.std(b)) / 2
        return abs(np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

    print(f"\nRough separation score (higher = better real/fake separation):")
    print(f"  Raw gap:        {separation(real_raw, fake_raw):.2f}")
    print(f"  Normalized gap: {separation(real_norm, fake_norm):.2f}")

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "label", "raw_gap_px", "normalized_gap"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved full results to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
