"""
Renames frame_0000.jpg -> 0000.jpg across the whole frames/ tree.

SAM2's video predictor requires frame filenames to be pure integers
(e.g. "0000.jpg"), not prefixed names like "frame_0000.jpg". This script
fixes that in-place, non-destructively (skips files that are already renamed).

Usage:
    python rename_frames.py --frames frames
"""

import argparse
import re
from pathlib import Path

PATTERN = re.compile(r"^frame_(\d+)\.jpg$", re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Path to frames root folder")
    args = parser.parse_args()

    root = Path(args.frames)
    if not root.is_dir():
        raise SystemExit(f"ERROR: {root} is not a directory")

    total_renamed = 0
    total_skipped = 0

    # Walk every video subfolder under real/ and fake/
    for label_dir in root.iterdir():
        if not label_dir.is_dir():
            continue
        for video_dir in label_dir.iterdir():
            if not video_dir.is_dir():
                continue

            renamed_here = 0
            for f in sorted(video_dir.iterdir()):
                if not f.is_file():
                    continue
                match = PATTERN.match(f.name)
                if match:
                    new_name = f"{match.group(1)}.jpg"
                    new_path = f.parent / new_name
                    f.rename(new_path)
                    renamed_here += 1
                    total_renamed += 1
                elif f.suffix.lower() == ".jpg":
                    # already numeric-only, or something unexpected — leave alone
                    total_skipped += 1

            if renamed_here:
                print(f"Renamed {renamed_here} frame(s) in {video_dir}")

    print(f"\nDone. Renamed: {total_renamed}, skipped/unchanged: {total_skipped}")


if __name__ == "__main__":
    main()