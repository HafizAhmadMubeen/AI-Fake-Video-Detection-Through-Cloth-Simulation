"""
Save frames with the MediaPipe pose skeleton overlaid, as individual images,
for visual review of Phase 2 pose extraction quality.

Draws all 33 landmarks as points, connected by MediaPipe's standard skeleton
edges (shoulders-elbows-wrists, hips-knees-ankles, etc.), color-coded by
MediaPipe's own per-landmark visibility score.

Usage:
  python visualize_pose.py --frames frames --pose pose --output pose_review --video "real/Man boxing"
  python visualize_pose.py --frames frames --pose pose --output pose_review --video "real/Man boxing" --step 5
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp

FRAME_GLOB = "*.jpg"
VISIBILITY_THRESHOLD = 0.5  # below this, draw the joint faded/dim

# Reuse MediaPipe's own connection list and landmark name/index mapping so
# the skeleton drawn here always matches whatever extract_pose.py actually
# used, rather than hand-maintaining a duplicate list that could drift.
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
LANDMARK_NAMES = [lm.name for lm in mp.solutions.pose.PoseLandmark]


LABELS = ("real", "fake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MediaPipe pose skeleton overlaid on frames.")
    parser.add_argument("--frames", type=Path, default=Path("frames"))
    parser.add_argument("--pose", type=Path, default=Path("pose"))
    parser.add_argument("--output", type=Path, default=Path("pose_review"))
    parser.add_argument("--video", type=str, default=None, help='e.g. "real/Man boxing". Omit if using --all.')
    parser.add_argument("--all", action="store_true", help="Process every video found under --frames.")
    parser.add_argument("--step", type=int, default=1, help="Save every Nth frame (default: 1, all frames).")
    return parser.parse_args()


def discover_videos(frames_root: Path) -> list[tuple[str, str]]:
    videos = []
    for label in LABELS:
        label_dir = frames_root / label
        if not label_dir.is_dir():
            continue
        for video_dir in sorted(label_dir.iterdir()):
            if video_dir.is_dir():
                videos.append((label, video_dir.name))
    return videos


def list_frame_files(video_dir: Path) -> list[Path]:
    frame_files = [p for p in video_dir.glob(FRAME_GLOB) if p.is_file()]
    return sorted(frame_files, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def draw_skeleton(frame, landmarks_for_frame: list[dict] | None) -> bool:
    """Returns True if a pose was drawn, False if no pose data for this frame."""
    if landmarks_for_frame is None:
        return False

    by_name = {entry["landmark"]: entry for entry in landmarks_for_frame}

    for a_idx, b_idx in POSE_CONNECTIONS:
        a_name, b_name = LANDMARK_NAMES[a_idx], LANDMARK_NAMES[b_idx]
        a, b = by_name.get(a_name), by_name.get(b_name)
        if a is None or b is None:
            continue
        pt_a = (int(round(a["x_px"])), int(round(a["y_px"])))
        pt_b = (int(round(b["x_px"])), int(round(b["y_px"])))
        min_vis = min(a["visibility"], b["visibility"])
        color = (0, 255, 0) if min_vis >= VISIBILITY_THRESHOLD else (0, 140, 140)
        cv2.line(frame, pt_a, pt_b, color, 2, cv2.LINE_AA)

    for entry in landmarks_for_frame:
        x, y = int(round(entry["x_px"])), int(round(entry["y_px"]))
        visible = entry["visibility"] >= VISIBILITY_THRESHOLD
        color = (0, 0, 255) if visible else (100, 100, 200)
        cv2.circle(frame, (x, y), 4, color, thickness=-1, lineType=cv2.LINE_AA)

    return True


def process_video(frames_root: Path, pose_root: Path, output_root: Path, label: str, video_name: str, step: int) -> None:
    video_dir = frames_root / label / video_name
    frame_files = list_frame_files(video_dir)
    if not frame_files:
        print(f"  WARNING: No frames found in {video_dir}, skipping")
        return

    keypoints_path = pose_root / label / video_name / "keypoints.json"
    if not keypoints_path.is_file():
        print(f"  WARNING: Pose keypoints not found: {keypoints_path}, skipping")
        return

    with keypoints_path.open("r", encoding="utf-8") as f:
        pose_data = json.load(f)

    out_dir = output_root / label / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    missing_pose = 0

    for frame_idx, frame_path in enumerate(frame_files):
        if frame_idx % step != 0:
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        frame_key = f"{frame_idx:04d}"
        landmarks_for_frame = pose_data.get(frame_key)

        drawn = draw_skeleton(frame, landmarks_for_frame)
        if not drawn:
            missing_pose += 1
            cv2.putText(frame, "NO POSE DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"frame {frame_idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = out_dir / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    print(f"  Saved {saved} frame(s) to {out_dir}")
    if missing_pose:
        print(f"  WARNING: no pose detected on {missing_pose} reviewed frame(s)")


def main() -> None:
    args = parse_args()

    frames_root = args.frames.resolve()
    pose_root = args.pose.resolve()
    output_root = args.output.resolve()

    if args.video:
        if "/" not in args.video:
            raise SystemExit('ERROR: --video must be "label/video_name", e.g. "real/Man boxing"')
        label, video_name = args.video.split("/", 1)
        videos = [(label, video_name)]
    elif args.all:
        videos = discover_videos(frames_root)
        if not videos:
            raise SystemExit(f"ERROR: No videos found under {frames_root}")
    else:
        raise SystemExit("ERROR: Specify either --video <label/video_name> or --all")

    print(f"Videos to process: {len(videos)}")
    print("-" * 60)

    for label, video_name in videos:
        print(f"\n{label}/{video_name}")
        process_video(frames_root, pose_root, output_root, label, video_name, args.step)

    print("\nDone. Red joints/green lines = confident. Faded joints/teal lines = low visibility.")


if __name__ == "__main__":
    main()