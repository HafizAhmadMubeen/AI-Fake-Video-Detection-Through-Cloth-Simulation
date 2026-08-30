"""
Phase 4, Step 2: Run the cloth simulator driven by real tracked skeleton
motion, per re-seed segment, and save the simulated trajectories in the same
format as Phase 3's observed trajectories -- so they can be directly compared.

For each segment:
  1. Take the tracked points' positions at the segment's first frame (seed
     frame) as the simulation's rest state.
  2. Build a spring mesh connecting each point to its ~k nearest neighbors.
  3. Find "anchor" points -- any tracked point within --anchor-threshold
     pixels of a real skeleton joint at the seed frame. These get driven
     directly by that joint's real motion each frame, rather than being
     freely simulated. (Points with no nearby joint are simulated freely.)
  4. Step the simulation forward frame by frame across the segment, at 30fps
     (matching Phase 1's extraction rate), re-pinning each anchor's position
     from the real skeleton before each step.

Output structure matches trajectories/ exactly:
{
  "upper": {"segments": [{"seed_frame": 0, "points": [{"point_id": 0, "trajectory": [...]}]}]},
  "lower": {...}
}
Each trajectory entry has "frame", "x", "y" (no "visible" flag -- these are
simulated, not observed).

Usage:
  python run_simulation.py --trajectories trajectories --pose pose --output simulated_trajectories --video "real/Man boxing"
  python run_simulation.py --trajectories trajectories --pose pose --output simulated_trajectories --all
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cloth_simulator import ClothSimulator, build_knn_springs

LABELS = ("real", "fake")
GARMENTS = ("upper", "lower")
FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cloth simulation driven by real skeleton motion.")
    parser.add_argument("--trajectories", type=Path, default=Path("trajectories"))
    parser.add_argument("--pose", type=Path, default=Path("pose"))
    parser.add_argument("--output", type=Path, default=Path("simulated_trajectories"))
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--k", type=int, default=4, help="Nearest neighbors per point for spring mesh (default: 4).")
    parser.add_argument("--anchor-threshold", type=float, default=60.0,
                         help="Max pixel distance to a skeleton joint to count as an anchor (default: 60).")
    return parser.parse_args()


def discover_videos(trajectories_root: Path) -> list[tuple[str, str]]:
    videos = []
    for label in LABELS:
        label_dir = trajectories_root / label
        if not label_dir.is_dir():
            continue
        for json_path in sorted(label_dir.glob("*.json")):
            videos.append((label, json_path.stem))
    return videos


def load_pose(pose_root: Path, label: str, video_name: str) -> dict | None:
    path = pose_root / label / video_name / "keypoints.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_joint_positions(pose_data: dict, frame_idx: int) -> np.ndarray | None:
    """Returns (num_joints, 2) array of joint pixel positions for a frame, or None."""
    entry = pose_data.get(f"{frame_idx:04d}")
    if not entry:
        return None
    return np.array([[lm["x_px"], lm["y_px"]] for lm in entry], dtype=np.float64)


def find_anchors(rest_positions: np.ndarray, joint_positions: np.ndarray, threshold: float):
    """
    For each tracked point, find the nearest joint. If within threshold,
    mark as anchor and record which joint index drives it and the initial
    offset (point - joint), so the point keeps its relative position as the
    joint moves rather than snapping exactly onto the joint.
    """
    n = len(rest_positions)
    anchor_mask = np.zeros(n, dtype=bool)
    anchor_joint_idx = np.full(n, -1, dtype=int)
    anchor_offset = np.zeros((n, 2), dtype=np.float64)

    for i, pos in enumerate(rest_positions):
        dists = np.linalg.norm(joint_positions - pos, axis=1)
        nearest_joint = int(np.argmin(dists))
        if dists[nearest_joint] <= threshold:
            anchor_mask[i] = True
            anchor_joint_idx[i] = nearest_joint
            anchor_offset[i] = pos - joint_positions[nearest_joint]

    return anchor_mask, anchor_joint_idx, anchor_offset


def simulate_segment(
    points: list[dict],
    pose_data: dict | None,
    seed_frame: int,
    k: int,
    anchor_threshold: float,
) -> list[dict] | None:
    """Returns a new list of points with simulated trajectories, or None if it can't be simulated."""
    num_points = len(points)
    num_frames = len(points[0]["trajectory"])
    frame_numbers = [e["frame"] for e in points[0]["trajectory"]]

    rest_positions = np.array(
        [[points[i]["trajectory"][0]["x"], points[i]["trajectory"][0]["y"]] for i in range(num_points)]
    )

    joint_positions_seed = get_joint_positions(pose_data, seed_frame) if pose_data else None
    if joint_positions_seed is None:
        anchor_mask = np.zeros(num_points, dtype=bool)
        anchor_joint_idx = np.full(num_points, -1, dtype=int)
        anchor_offset = np.zeros((num_points, 2))
    else:
        anchor_mask, anchor_joint_idx, anchor_offset = find_anchors(
            rest_positions, joint_positions_seed, anchor_threshold
        )

    if not anchor_mask.any():
        return None

    springs = build_knn_springs(rest_positions, k=k)
    sim = ClothSimulator(rest_positions, springs, anchor_mask)

    last_known_joint_positions = joint_positions_seed
    simulated = [[rest_positions[i].copy()] for i in range(num_points)]

    dt = 1.0 / FPS
    for frame_num in frame_numbers[1:]:
        joint_positions = get_joint_positions(pose_data, frame_num) if pose_data else None
        if joint_positions is None:
            joint_positions = last_known_joint_positions
        else:
            last_known_joint_positions = joint_positions

        if joint_positions is not None:
            anchor_targets = np.array([
                joint_positions[anchor_joint_idx[i]] + anchor_offset[i]
                for i in range(num_points) if anchor_mask[i]
            ])
            sim.set_anchor_positions(anchor_targets)

        sim.step(dt)

        for i in range(num_points):
            simulated[i].append(sim.positions[i].copy())

    result = []
    for i in range(num_points):
        trajectory = [
            {"frame": frame_numbers[t], "x": float(simulated[i][t][0]), "y": float(simulated[i][t][1])}
            for t in range(num_frames)
        ]
        result.append({"point_id": points[i]["point_id"], "trajectory": trajectory})

    return result


def process_video(trajectories_root: Path, pose_root: Path, output_root: Path, label: str, video_name: str, k: int, anchor_threshold: float) -> None:
    traj_path = trajectories_root / label / f"{video_name}.json"
    with traj_path.open("r", encoding="utf-8") as f:
        traj_data = json.load(f)

    pose_data = load_pose(pose_root, label, video_name)
    if pose_data is None:
        print(f"  WARNING: No pose data found for {label}/{video_name}, skipping")
        return

    result = {}
    for garment in GARMENTS:
        if garment not in traj_data:
            continue
        segments_out = []
        for segment in traj_data[garment]["segments"]:
            seed_frame = segment["seed_frame"]
            sim_points = simulate_segment(segment["points"], pose_data, seed_frame, k, anchor_threshold)
            if sim_points is None:
                print(f"  {garment} segment seed_frame={seed_frame}: no anchors found, skipped")
                continue
            segments_out.append({"seed_frame": seed_frame, "points": sim_points})
        if segments_out:
            result[garment] = {"segments": segments_out}
        print(f"  {garment}: {len(segments_out)}/{len(traj_data[garment]['segments'])} segment(s) simulated")

    if not result:
        print(f"  WARNING: No segments simulated for {label}/{video_name}")
        return

    output_path = output_root / label / f"{video_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {output_path}")


def main() -> None:
    args = parse_args()

    trajectories_root = args.trajectories.resolve()
    pose_root = args.pose.resolve()
    output_root = args.output.resolve()

    if args.video:
        label, video_name = args.video.split("/", 1)
        videos = [(label, video_name)]
    elif args.all:
        videos = discover_videos(trajectories_root)
    else:
        raise SystemExit("ERROR: Specify --video <label/video_name> or --all")

    print(f"Videos to process: {len(videos)}")
    print("-" * 60)

    for label, video_name in videos:
        print(f"\n{label}/{video_name}")
        process_video(trajectories_root, pose_root, output_root, label, video_name, args.k, args.anchor_threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()
