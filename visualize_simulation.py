"""
Phase 4 proof-of-concept check: overlay BOTH the real observed tracked points
(from Phase 3) and the simulated points (from run_simulation.py) on the same
frames, so you can visually judge whether the physics simulation is producing
plausible motion relative to reality -- before trusting it to compute a
residual number.

Coloring:
  - Observed points (real, from CoTracker): green
  - Simulated points (physics prediction): magenta
  - A short line connects each matching pair, so the gap between them (the
    residual) is visually obvious.

Usage:
  python visualize_simulation.py --frames frames --trajectories trajectories --simulated simulated_trajectories --output sim_review --video "real/Man boxing"
"""

import argparse
import json
from pathlib import Path

import cv2

FRAME_GLOB = "*.jpg"
OBSERVED_COLOR = (80, 220, 80)     # BGR: green
SIMULATED_COLOR = (200, 60, 200)   # BGR: magenta
LINE_COLOR = (200, 200, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare simulated vs observed points visually.")
    parser.add_argument("--frames", type=Path, default=Path("frames"))
    parser.add_argument("--trajectories", type=Path, default=Path("trajectories"))
    parser.add_argument("--simulated", type=Path, default=Path("simulated_trajectories"))
    parser.add_argument("--output", type=Path, default=Path("sim_review"))
    parser.add_argument("--video", type=str, required=True, help='e.g. "real/Man boxing"')
    parser.add_argument("--step", type=int, default=1)
    return parser.parse_args()


def list_frame_files(video_dir: Path) -> list[Path]:
    frame_files = [p for p in video_dir.glob(FRAME_GLOB) if p.is_file()]
    return sorted(frame_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def build_frame_lookup(garment_data: dict) -> dict[int, list[dict]]:
    lookup: dict[int, list[dict]] = {}
    for segment in garment_data.get("segments", []):
        for point in segment["points"]:
            for entry in point["trajectory"]:
                lookup.setdefault(entry["frame"], []).append({
                    "point_id": point["point_id"],
                    "x": entry["x"],
                    "y": entry["y"],
                    "visible": entry.get("visible", True),
                })
    return lookup


def main() -> None:
    args = parse_args()
    label, video_name = args.video.split("/", 1)

    frames_root = args.frames.resolve()
    trajectories_root = args.trajectories.resolve()
    simulated_root = args.simulated.resolve()
    output_root = args.output.resolve()

    video_dir = frames_root / label / video_name
    frame_files = list_frame_files(video_dir)
    if not frame_files:
        raise SystemExit(f"ERROR: No frames found in {video_dir}")

    obs_path = trajectories_root / label / f"{video_name}.json"
    sim_path = simulated_root / label / f"{video_name}.json"
    if not obs_path.is_file():
        raise SystemExit(f"ERROR: Observed trajectories not found: {obs_path}")
    if not sim_path.is_file():
        raise SystemExit(f"ERROR: Simulated trajectories not found: {sim_path}")

    with obs_path.open("r", encoding="utf-8") as f:
        obs_data = json.load(f)
    with sim_path.open("r", encoding="utf-8") as f:
        sim_data = json.load(f)

    obs_lookup = {g: build_frame_lookup(obs_data[g]) for g in obs_data}
    sim_lookup = {g: build_frame_lookup(sim_data[g]) for g in sim_data}

    out_dir = output_root / label / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for frame_idx, frame_path in enumerate(frame_files):
        if frame_idx % args.step != 0:
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        for garment in obs_lookup:
            obs_points = {p["point_id"]: p for p in obs_lookup[garment].get(frame_idx, [])}
            sim_points = {p["point_id"]: p for p in sim_lookup.get(garment, {}).get(frame_idx, [])}

            for point_id, sim_p in sim_points.items():
                obs_p = obs_points.get(point_id)
                sx, sy = int(round(sim_p["x"])), int(round(sim_p["y"]))
                cv2.circle(frame, (sx, sy), 4, SIMULATED_COLOR, thickness=-1, lineType=cv2.LINE_AA)
                if obs_p is not None:
                    ox, oy = int(round(obs_p["x"])), int(round(obs_p["y"]))
                    cv2.line(frame, (ox, oy), (sx, sy), LINE_COLOR, 1, cv2.LINE_AA)

            for point_id, obs_p in obs_points.items():
                ox, oy = int(round(obs_p["x"])), int(round(obs_p["y"]))
                cv2.circle(frame, (ox, oy), 4, OBSERVED_COLOR, thickness=-1, lineType=cv2.LINE_AA)

        cv2.putText(frame, f"frame {frame_idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "green=observed  magenta=simulated", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = out_dir / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    print(f"Saved {saved} frame(s) to: {out_dir}")
    print("Green = real observed position, Magenta = physics-simulated position.")
    print("The gray line between them shows the gap (this gap is the physics residual).")


if __name__ == "__main__":
    main()
