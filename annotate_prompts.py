"""
Phase 2a: Manually annotate SAM2 click prompts on the first frame of each video.

Walks frames/real/<video_name>/frame_0000.jpg and frames/fake/<video_name>/frame_0000.jpg,
collects one upper-garment and one lower-garment click per video, and writes prompts.json.
"""

import argparse
import json
from pathlib import Path

import cv2

FIRST_FRAME_NAME = "frame_0000.jpg"
LABELS = ("real", "fake")

INSTRUCTIONS = (
    "Click upper garment, then lower garment, then press any key to continue. "
    "Press 'r' to redo this frame. Press 's' to skip this video."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate upper/lower garment click prompts for SAM2 segmentation."
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
        default=Path("prompts.json"),
        help='Path to save prompts.json (default: "prompts.json").',
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Re-annotate videos that already have entries in prompts.json.",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> dict:
    """Load existing prompts JSON, or return an empty dict if missing."""
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"WARNING: {path} is not a JSON object; starting fresh.")
            return {}
        return data
    except json.JSONDecodeError as exc:
        print(f"WARNING: Could not parse {path} ({exc}); starting fresh.")
        return {}


def save_prompts(path: Path, prompts: dict) -> None:
    """Write prompts to disk with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2)
        f.write("\n")


def list_video_folders(frames_root: Path, label: str) -> list[Path]:
    """Return sorted video frame folders under frames/<label>/."""
    label_dir = frames_root / label
    if not label_dir.is_dir():
        print(f"WARNING: Label folder not found, skipping: {label_dir}")
        return []
    folders = sorted(path for path in label_dir.iterdir() if path.is_dir())
    if not folders:
        print(f"No video folders found in {label_dir}")
    return folders


def draw_ui(image, clicks: list[tuple[int, int]], status: str) -> None:
    """Draw instructions, click markers, and status text on *image* in place."""
    display = image.copy()
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    cv2.putText(
        display,
        INSTRUCTIONS,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        status,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"Clicks: {len(clicks)}/2",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )

    colors = [(255, 180, 0), (0, 180, 255)]  # upper (blue-ish), lower (orange-ish)
    labels = ("upper", "lower")
    for idx, (x, y) in enumerate(clicks):
        color = colors[idx]
        cv2.circle(display, (x, y), 6, color, -1)
        cv2.circle(display, (x, y), 10, (255, 255, 255), 2)
        cv2.putText(
            display,
            labels[idx],
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    image[:] = display


def annotate_video(
    frame_path: Path,
    video_key: str,
    window_name: str,
) -> dict | None | str:
    """
    Interactively annotate one video's first frame.

    Returns:
        {"upper": [x, y], "lower": [x, y]} on success,
        None if the user skips the video,
        "redo_needed" is not returned — handled internally.
    """
    image = cv2.imread(str(frame_path))
    if image is None:
        print(f"  WARNING: Could not read image, skipping: {frame_path}")
        return None

    clicks: list[tuple[int, int]] = []
    status = "Left-click: upper garment"

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))
            nonlocal status
            if len(clicks) == 1:
                status = "Left-click: lower garment"
            else:
                status = "Press any key to save, 'r' to redo, 's' to skip"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        canvas = image.copy()
        draw_ui(canvas, clicks, status)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            clicks.clear()
            status = "Left-click: upper garment"
            continue

        if key == ord("s"):
            cv2.destroyWindow(window_name)
            print(f"  Skipped by user: {video_key}")
            return None

        if key != 255:  # any key pressed
            if len(clicks) != 2:
                print(
                    f"  WARNING: Expected 2 clicks before continuing, got {len(clicks)}. "
                    "Press 'r' to redo."
                )
                status = "Need exactly 2 clicks — press 'r' to redo"
                continue

            cv2.destroyWindow(window_name)
            return {
                "upper": [clicks[0][0], clicks[0][1]],
                "lower": [clicks[1][0], clicks[1][1]],
            }


def collect_annotation_jobs(frames_root: Path) -> list[tuple[str, Path]]:
    """Build a list of (video_key, frame_0000 path) jobs."""
    jobs: list[tuple[str, Path]] = []
    for label in LABELS:
        for video_dir in list_video_folders(frames_root, label):
            frame_path = video_dir / FIRST_FRAME_NAME
            video_key = f"{label}/{video_dir.name}"
            if not frame_path.is_file():
                print(f"WARNING: Missing {FIRST_FRAME_NAME}, skipping: {video_dir}")
                continue
            jobs.append((video_key, frame_path))
    return jobs


def main() -> None:
    args = parse_args()
    frames_root = args.frames.resolve()
    output_path = args.output.resolve()

    if not frames_root.is_dir():
        raise SystemExit(f"ERROR: Frames folder is not a directory: {frames_root}")

    prompts = load_prompts(output_path)
    jobs = collect_annotation_jobs(frames_root)

    print(f"Frames root : {frames_root}")
    print(f"Output file : {output_path}")
    print(f"Videos found: {len(jobs)}")
    print("-" * 60)

    annotated = 0
    skipped_existing = 0
    skipped_user = 0

    for video_key, frame_path in jobs:
        if video_key in prompts and not args.redo:
            print(f"Skipping (already annotated): {video_key}")
            skipped_existing += 1
            continue

        print(f"Annotating: {video_key}")
        print(f"  Frame: {frame_path}")

        window_name = f"Annotate: {video_key}"
        result = annotate_video(frame_path, video_key, window_name)

        if result is None:
            skipped_user += 1
            continue

        prompts[video_key] = result
        save_prompts(output_path, prompts)
        annotated += 1
        print(f"  Saved prompts for {video_key}")

    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Newly annotated     : {annotated}")
    print(f"  Skipped (existing)  : {skipped_existing}")
    print(f"  Skipped (by user)   : {skipped_user}")
    print(f"  Total in {output_path.name}: {len(prompts)}")


if __name__ == "__main__":
    main()
