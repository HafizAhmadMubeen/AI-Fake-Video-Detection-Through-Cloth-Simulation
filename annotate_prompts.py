"""
Phase 2a (reworked): Draw a bounding BOX around the upper garment and another
around the lower garment, for frame 0 of each video, instead of a single
click point.

Why this changed: single-point prompts proved unreliable -- SAM2 sometimes
grabbed the whole body (point too ambiguous, no boundary information) or
produced an empty mask (point landed somewhere SAM2 couldn't anchor to).
A box gives SAM2 an explicit boundary, which is a much stronger and more
reliable signal for isolating a specific garment region.

Controls, per video:
  - Click and drag to draw a box around the UPPER garment, release to confirm.
  - Click and drag to draw a box around the LOWER garment, release to confirm.
  - Press 'r' to redo both boxes for this video.
  - Press 's' to skip this video.
  - Press any other key once both boxes are drawn to confirm and move on.

Output structure per video in prompts.json:
{
  "real/video_name": {
    "upper": [x1, y1, x2, y2],
    "lower": [x1, y1, x2, y2]
  },
  ...
}

Usage:
  python annotate_prompts.py --frames frames --output prompts.json
  python annotate_prompts.py --frames frames --output prompts.json --redo
"""

import argparse
import json
from pathlib import Path

import cv2

LABELS = ("real", "fake")

drawing = False
start_point = None
current_box = None
boxes_this_video = []
window_name = "Draw box: UPPER garment then LOWER garment (drag with mouse)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw box prompts for upper/lower garments.")
    parser.add_argument("--frames", type=Path, default=Path("frames"))
    parser.add_argument("--output", type=Path, default=Path("prompts.json"))
    parser.add_argument("--redo", action="store_true", help="Re-annotate videos already in the output file.")
    return parser.parse_args()


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, current_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_box = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_box = (start_point[0], start_point[1], x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if start_point is not None:
            x1, y1 = start_point
            x2, y2 = x, y
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            if box[2] - box[0] > 3 and box[3] - box[1] > 3:  # ignore accidental clicks/tiny drags
                boxes_this_video.append(box)
            current_box = None


def draw_overlay(base_frame, label_text: str):
    display = base_frame.copy()

    colors = [(255, 140, 0), (0, 140, 255)]  # blue for upper, orange for lower
    for i, box in enumerate(boxes_this_video):
        color = colors[i] if i < len(colors) else (0, 255, 0)
        cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), color, 2)

    if drawing and current_box is not None:
        cv2.rectangle(display, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (0, 255, 0), 1)

    cv2.putText(display, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        display, "r=redo  s=skip  any other key=confirm once both boxes drawn",
        (10, display.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return display


def annotate_video(frame_path: Path) -> dict | None:
    """Returns {"upper": [x1,y1,x2,y2], "lower": [...]} or None if skipped."""
    global boxes_this_video

    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"  WARNING: Could not load {frame_path}")
        return None

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        boxes_this_video = []
        while True:
            if len(boxes_this_video) == 0:
                label_text = "Draw box around UPPER garment"
            elif len(boxes_this_video) == 1:
                label_text = "Draw box around LOWER garment"
            else:
                label_text = "Both boxes drawn -- press any key to confirm"

            display = draw_overlay(frame, label_text)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s'):
                cv2.destroyWindow(window_name)
                return None
            if key == ord('r'):
                boxes_this_video = []
                continue
            if key != 255 and len(boxes_this_video) >= 2:
                break

        if len(boxes_this_video) >= 2:
            cv2.destroyWindow(window_name)
            return {"upper": boxes_this_video[0], "lower": boxes_this_video[1]}
        # otherwise loop back (e.g. user pressed a key before finishing both boxes)


def main() -> None:
    args = parse_args()

    frames_root = args.frames.resolve()
    output_path = args.output.resolve()

    existing = {}
    if output_path.is_file():
        with output_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)

    videos = []
    for label in LABELS:
        label_dir = frames_root / label
        if not label_dir.is_dir():
            continue
        for video_dir in sorted(label_dir.iterdir()):
            if video_dir.is_dir():
                videos.append((label, video_dir.name))

    if not videos:
        raise SystemExit(f"ERROR: No videos found under {frames_root}")

    print(f"Frames root: {frames_root}")
    print(f"Output     : {output_path}")
    print(f"Videos     : {len(videos)}")
    print("-" * 60)

    for label, video_name in videos:
        video_key = f"{label}/{video_name}"

        if video_key in existing and not args.redo:
            print(f"Skipping (already annotated): {video_key}")
            continue

        video_dir = frames_root / label / video_name
        frame_files = sorted(
            [p for p in video_dir.glob("*.jpg") if p.is_file()],
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )
        if not frame_files:
            print(f"WARNING: No frames found for {video_key}, skipping")
            continue

        print(f"\nAnnotating: {video_key}")
        result = annotate_video(frame_files[0])

        if result is None:
            print(f"  Skipped: {video_key}")
            continue

        existing[video_key] = result
        print(f"  Upper box: {result['upper']}")
        print(f"  Lower box: {result['lower']}")

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

    cv2.destroyAllWindows()
    print(f"\nDone. Saved to: {output_path}")


if __name__ == "__main__":
    main()