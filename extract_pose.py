"""
Phase 2c: Extract MediaPipe Pose skeleton keypoints for every frame in each video.

Reads frame folders under frames/real/ and frames/fake/, runs pose estimation on each
frame, and writes one keypoints.json per video.
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp

LABELS = ("real", "fake")
FRAME_GLOB = "*.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Pose landmarks for all extracted video frames."
    )
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("frames"),
        help='Path to frames root folder (default: "frames").',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pose"),
        help='Output root for pose JSON files (default: "pose").',
    )
    return parser.parse_args()


def list_video_folders(frames_root: Path, label: str) -> list[Path]:
    label_dir = frames_root / label
    if not label_dir.is_dir():
        print(f"WARNING: Label folder not found, skipping: {label_dir}")
        return []
    folders = sorted(path for path in label_dir.iterdir() if path.is_dir())
    if not folders:
        print(f"No video folders found in {label_dir}")
    return folders


def list_frame_files(video_dir: Path) -> list[Path]:
    if not video_dir.is_dir():
        return []
    frame_files = [path for path in video_dir.glob(FRAME_GLOB) if path.is_file()]
    return sorted(
        frame_files,
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def landmarks_to_json(landmarks, width: int, height: int, pose_landmark_enum) -> list[dict]:
    """Convert MediaPipe pose landmarks to the project JSON schema."""
    output = []
    for landmark_id in pose_landmark_enum:
        lm = landmarks[landmark_id.value]
        output.append(
            {
                "landmark": landmark_id.name,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
                "x_px": lm.x * width,
                "y_px": lm.y * height,
            }
        )
    return output


def process_video(
    pose,
    pose_landmark_enum,
    video_dir: Path,
    output_path: Path,
) -> tuple[int, int]:
    """
    Run pose estimation on every frame in *video_dir*.

    Returns (success_count, failure_count).
    """
    frame_files = list_frame_files(video_dir)
    if not frame_files:
        print(f"  WARNING: No frames found in {video_dir}")
        return 0, 0

    results_by_frame: dict[str, list[dict] | None] = {}
    success_count = 0
    failure_count = 0

    for frame_path in frame_files:
        frame_key = frame_path.stem  # e.g. frame_0000
        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            print(f"  WARNING: Could not read frame, storing null: {frame_path}")
            results_by_frame[frame_key] = None
            failure_count += 1
            continue

        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            results_by_frame[frame_key] = None
            failure_count += 1
            continue

        results_by_frame[frame_key] = landmarks_to_json(
            results.pose_landmarks.landmark,
            width,
            height,
            pose_landmark_enum,
        )
        success_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results_by_frame, f, indent=2)
        f.write("\n")

    return success_count, failure_count


def main() -> None:
    args = parse_args()
    frames_root = args.frames.resolve()
    output_root = args.output.resolve()

    if not frames_root.is_dir():
        raise SystemExit(f"ERROR: Frames folder is not a directory: {frames_root}")

    print(f"Frames root : {frames_root}")
    print(f"Output root : {output_root}")
    print("-" * 60)

    mp_pose = mp.solutions.pose
    total_success = 0
    total_failure = 0
    videos_processed = 0

    # static_image_mode=True because each frame is an independent still image.
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        for label in LABELS:
            print(f"\n=== Label: {label} ===")
            for video_dir in list_video_folders(frames_root, label):
                video_name = video_dir.name
                output_path = output_root / label / video_name / "keypoints.json"

                print(f"Processing [{label}] {video_name}")
                print(f"  Output: {output_path}")

                success, failure = process_video(
                    pose,
                    mp_pose.PoseLandmark,
                    video_dir,
                    output_path,
                )
                total_success += success
                total_failure += failure
                videos_processed += 1

                print(
                    f"  Pose detected: {success}/{success + failure} frame(s)"
                )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Videos processed : {videos_processed}")
    print(f"  Frames succeeded : {total_success}")
    print(f"  Frames failed    : {total_failure}")
    total_frames = total_success + total_failure
    if total_frames:
        rate = 100.0 * total_success / total_frames
        print(f"  Success rate     : {rate:.1f}%")


if __name__ == "__main__":
    main()
