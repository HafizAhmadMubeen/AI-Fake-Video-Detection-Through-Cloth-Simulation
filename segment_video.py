"""
Phase 2b: Segment upper- and lower-body garments with SAM2's video predictor.

Uses click prompts from prompts.json to track two objects (upper = ID 1, lower = ID 2)
across every frame in each video folder.

SAM2.1 checkpoint setup (run once before first use)
---------------------------------------------------
Recommended for ~8 GB VRAM laptops: "small" or "base_plus" (not "large").

1) Install SAM 2 (requires Python >= 3.10, PyTorch >= 2.5.1, and a CUDA GPU):

   git clone https://github.com/facebookresearch/sam2.git
   cd sam2
   pip install -e .

2) Download a SAM 2.1 checkpoint + note its config path (from the sam2 repo root):

   # Small (~46M params) — best fit for 8 GB VRAM
   mkdir -p checkpoints
   curl -L -o checkpoints/sam2.1_hiera_small.pt \\
     https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

   # Base+ (~81M params) — slightly heavier but still laptop-friendly
   curl -L -o checkpoints/sam2.1_hiera_base_plus.pt \\
     https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

   Config files ship with the repo (no separate download):
     configs/sam2.1/sam2.1_hiera_s.yaml        # pairs with sam2.1_hiera_small.pt
     configs/sam2.1/sam2.1_hiera_b+.yaml       # pairs with sam2.1_hiera_base_plus.pt

3) Example invocation from the sam2 repo root (adjust paths as needed):

   python ../segment_video.py \\
     --frames ../frames \\
     --prompts ../prompts.json \\
     --output ../masks \\
     --checkpoint checkpoints/sam2.1_hiera_small.pt \\
     --config configs/sam2.1/sam2.1_hiera_s.yaml
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

UPPER_OBJECT_ID = 1
LOWER_OBJECT_ID = 2
FRAME_GLOB = "*.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM2 video segmentation for upper/lower garment masks."
    )
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("frames"),
        help='Path to extracted frames root (default: "frames").',
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("prompts.json"),
        help='Path to prompts JSON from annotate_prompts.py (default: "prompts.json").',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("masks"),
        help='Output root for mask PNGs (default: "masks").',
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to SAM2.1 checkpoint .pt file (small or base_plus recommended).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to SAM2 model config YAML (e.g. configs/sam2.1/sam2.1_hiera_s.yaml).",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"ERROR: Prompts file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: Expected a JSON object in {path}")
    return data


def list_frame_files(video_dir: Path) -> list[Path]:
    """Return sorted frame JPG paths for a video folder."""
    if not video_dir.is_dir():
        return []
    frame_files = [path for path in video_dir.glob(FRAME_GLOB) if path.is_file()]
    return sorted(
        frame_files,
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def frame_index_to_name(frame_idx: int) -> str:
    return f"frame_{frame_idx:04d}"


def parse_video_key(video_key: str) -> tuple[str, str]:
    """Split 'real/video_name' into label and video folder name."""
    if "/" not in video_key:
        raise ValueError(f"Invalid video key (expected label/video_name): {video_key}")
    label, video_name = video_key.split("/", 1)
    return label, video_name


def mask_tensor_to_uint8(mask_tensor) -> np.ndarray:
    """Convert one SAM2 mask logit tensor to a binary 0/255 uint8 image."""
    mask = mask_tensor
    if hasattr(mask, "detach"):
        mask = mask.detach()
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    binary = (mask > 0.0).cpu().numpy().astype(np.uint8) * 255
    return binary


def save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), mask):
        raise IOError(f"Failed to write mask: {out_path}")


def overlay_masks(
    frame_bgr: np.ndarray,
    upper_mask: np.ndarray | None,
    lower_mask: np.ndarray | None,
    upper_alpha: float = 0.45,
    lower_alpha: float = 0.45,
) -> np.ndarray:
    """Blend upper (blue) and lower (red) masks onto the original BGR frame."""
    overlay = frame_bgr.astype(np.float32).copy()

    if upper_mask is not None:
        blue = np.zeros_like(overlay)
        blue[:, :, 0] = 255.0
        mask_bool = upper_mask > 0
        overlay[mask_bool] = (
            overlay[mask_bool] * (1.0 - upper_alpha) + blue[mask_bool] * upper_alpha
        )

    if lower_mask is not None:
        red = np.zeros_like(overlay)
        red[:, :, 2] = 255.0
        mask_bool = lower_mask > 0
        overlay[mask_bool] = (
            overlay[mask_bool] * (1.0 - lower_alpha) + red[mask_bool] * lower_alpha
        )

    return overlay.astype(np.uint8)


def process_one_video(
    predictor,
    device: str,
    frames_root: Path,
    output_root: Path,
    video_key: str,
    prompt: dict,
) -> tuple[int, dict[int, dict[int, np.ndarray]]]:
    """
    Segment one video and write upper/lower mask PNGs.

    Returns (num_frames_saved, masks_at_frame_idx) where masks_at_frame_idx maps
    frame_idx -> {object_id: mask_array} for sanity-check overlays.
    """
    label, video_name = parse_video_key(video_key)
    video_dir = frames_root / label / video_name
    frame_files = list_frame_files(video_dir)
    if not frame_files:
        raise FileNotFoundError(f"No frame images found in {video_dir}")

    upper_pt = prompt.get("upper")
    lower_pt = prompt.get("lower")
    if (
        not isinstance(upper_pt, list)
        or len(upper_pt) != 2
        or not isinstance(lower_pt, list)
        or len(lower_pt) != 2
    ):
        raise ValueError(f"Prompt for {video_key} must contain 'upper' and 'lower' [x, y] lists")

    upper_out = output_root / "upper" / label / video_name
    lower_out = output_root / "lower" / label / video_name
    upper_out.mkdir(parents=True, exist_ok=True)
    lower_out.mkdir(parents=True, exist_ok=True)

    masks_by_frame: dict[int, dict[int, np.ndarray]] = {}

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(video_dir))

        for obj_id, point in (
            (UPPER_OBJECT_ID, upper_pt),
            (LOWER_OBJECT_ID, lower_pt),
        ):
            points = np.array([[float(point[0]), float(point[1])]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            frame_name = frame_index_to_name(frame_idx)
            frame_masks: dict[int, np.ndarray] = {}

            for i, obj_id in enumerate(obj_ids):
                mask_u8 = mask_tensor_to_uint8(mask_logits[i])
                frame_masks[obj_id] = mask_u8

                if obj_id == UPPER_OBJECT_ID:
                    save_mask_png(mask_u8, upper_out / f"{frame_name}.png")
                elif obj_id == LOWER_OBJECT_ID:
                    save_mask_png(mask_u8, lower_out / f"{frame_name}.png")

            masks_by_frame[frame_idx] = frame_masks

    return len(frame_files), masks_by_frame


def write_sanity_check(
    frames_root: Path,
    sanity_dir: Path,
    video_key: str,
    masks_by_frame: dict[int, dict[int, np.ndarray]],
    num_frames: int,
) -> bool:
    """Save one overlay image at ~25% through the video for manual inspection."""
    if num_frames <= 0:
        return False

    label, video_name = parse_video_key(video_key)
    target_idx = max(0, int(round(0.25 * (num_frames - 1))))
    frame_path = frames_root / label / video_name / f"{target_idx:04d}.jpg"

    frame_bgr = cv2.imread(str(frame_path))
    if frame_bgr is None:
        print(f"  WARNING: Could not load frame for sanity check: {frame_path}")
        return False

    frame_masks = masks_by_frame.get(target_idx, {})
    overlay = overlay_masks(
        frame_bgr,
        frame_masks.get(UPPER_OBJECT_ID),
        frame_masks.get(LOWER_OBJECT_ID),
    )

    safe_name = video_key.replace("/", "__")
    out_path = sanity_dir / f"{safe_name}_frame_{target_idx:04d}.jpg"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), overlay):
        print(f"  WARNING: Failed to write sanity check image: {out_path}")
        return False
    return True


def log_failure(failures_path: Path, video_key: str, error: Exception) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    with failures_path.open("a", encoding="utf-8") as f:
        f.write(f"{video_key}\t{type(error).__name__}: {error}\n")


def main() -> None:
    args = parse_args()

    frames_root = args.frames.resolve()
    prompts_path = args.prompts.resolve()
    output_root = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    config = str(args.config.resolve())

    if not frames_root.is_dir():
        raise SystemExit(f"ERROR: Frames folder is not a directory: {frames_root}")
    if not checkpoint.is_file():
        raise SystemExit(f"ERROR: Checkpoint not found: {checkpoint}")
    if not Path(config).is_file():
        raise SystemExit(f"ERROR: Config not found: {config}")

    prompts = load_prompts(prompts_path)
    if not prompts:
        raise SystemExit(f"ERROR: No entries found in {prompts_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available; SAM2 video inference on CPU will be very slow.")

    print(f"Frames root : {frames_root}")
    print(f"Prompts     : {prompts_path}")
    print(f"Output      : {output_root}")
    print(f"Checkpoint  : {checkpoint}")
    print(f"Config      : {config}")
    print(f"Device      : {device}")
    print("-" * 60)

    predictor = build_sam2_video_predictor(
        config_file=config,
        ckpt_path=str(checkpoint),
        device=device,
    )

    failures_path = Path("failures.txt")
    sanity_dir = Path("sanity_checks")
    if failures_path.exists():
        failures_path.unlink()

    successful = 0
    failed = 0
    sanity_saved = 0

    for video_key in sorted(prompts.keys()):
        prompt = prompts[video_key]
        print(f"\nProcessing: {video_key}")

        started = time.perf_counter()
        try:
            num_frames, masks_by_frame = process_one_video(
                predictor,
                device,
                frames_root,
                output_root,
                video_key,
                prompt,
            )
            elapsed = time.perf_counter() - started
            print(f"  Frames processed: {num_frames}")
            print(f"  Time taken      : {elapsed:.1f}s")
            successful += 1

            if sanity_saved < 3:
                if write_sanity_check(
                    frames_root,
                    sanity_dir,
                    video_key,
                    masks_by_frame,
                    num_frames,
                ):
                    sanity_saved += 1
                    print(f"  Sanity check saved to {sanity_dir}")

        except Exception as exc:
            elapsed = time.perf_counter() - started
            failed += 1
            print(f"  ERROR after {elapsed:.1f}s: {exc}")
            traceback.print_exc()
            log_failure(failures_path, video_key, exc)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Successful : {successful}/{len(prompts)}")
    print(f"  Failed     : {failed}")
    if failed:
        print(f"  Failures logged to: {failures_path.resolve()}")
    if sanity_saved:
        print(f"  Sanity checks: {sanity_saved} image(s) in {sanity_dir.resolve()}")


if __name__ == "__main__":
    main()
