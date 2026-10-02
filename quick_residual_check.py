"""
Quick numeric sanity check: prints the average and max pixel distance between
simulated and observed points, per frame, for one video. Use this instead of
(or alongside) the visual overlay -- overlapping same-size dots are hard to
judge by eye, a number is not.

Usage:
  python quick_residual_check.py --trajectories trajectories --simulated simulated_trajectories --video "real/man doing ghost rope"
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, default=Path("trajectories"))
    parser.add_argument("--simulated", type=Path, default=Path("simulated_trajectories"))
    parser.add_argument("--video", type=str, required=True)
    return parser.parse_args()


def build_lookup(garment_data):
    lookup = {}
    for segment in garment_data.get("segments", []):
        for point in segment["points"]:
            for entry in point["trajectory"]:
                lookup.setdefault(entry["frame"], {})[point["point_id"]] = (entry["x"], entry["y"])
    return lookup


def main():
    args = parse_args()
    label, video_name = args.video.split("/", 1)

    with (args.trajectories / label / f"{video_name}.json").open() as f:
        obs_data = json.load(f)
    with (args.simulated / label / f"{video_name}.json").open() as f:
        sim_data = json.load(f)

    for garment in obs_data:
        if garment not in sim_data:
            print(f"{garment}: not simulated, skipping")
            continue

        obs_lookup = build_lookup(obs_data[garment])
        sim_lookup = build_lookup(sim_data[garment])

        all_frames = sorted(set(obs_lookup) & set(sim_lookup))
        per_frame_avg = []

        for frame in all_frames:
            dists = []
            for point_id, (ox, oy) in obs_lookup[frame].items():
                if point_id in sim_lookup[frame]:
                    sx, sy = sim_lookup[frame][point_id]
                    dists.append(((ox - sx) ** 2 + (oy - sy) ** 2) ** 0.5)
            if dists:
                per_frame_avg.append(np.mean(dists))

        per_frame_avg = np.array(per_frame_avg)
        print(f"\n{garment}:")
        print(f"  Frames compared: {len(per_frame_avg)}")
        print(f"  Mean gap across video: {per_frame_avg.mean():.1f} px")
        print(f"  Min frame gap: {per_frame_avg.min():.1f} px   Max frame gap: {per_frame_avg.max():.1f} px")
        print(f"  Gap at frame 0: {per_frame_avg[0]:.1f} px   Gap at last frame: {per_frame_avg[-1]:.1f} px")


if __name__ == "__main__":
    main()
