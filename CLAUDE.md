# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Detects AI-generated videos of humans by checking whether clothing motion obeys real
physics (gravity, stretch, draping, body collision), rather than relying on
generator-specific visual artifacts. The bet: fabric dynamics are governed by physical
principles that generative video models don't explicitly simulate, so a physics-residual
signal should generalize to unseen generators instead of overfitting to one model's
cosmetic fingerprints.

The pipeline is a sequence of standalone CLI scripts, each consuming the previous
stage's output directory and producing its own, keyed by `<label>/<video_name>` where
`label` is `real` or `fake`. There is no orchestrator script — phases are run manually
in order.

**Pipeline status:** Phases 1-3 (frame extraction, segmentation+pose, point tracking) are
done. Phase 4 (physics simulation + residual computation) and Phase 5 (classifier) are not
yet implemented.

## Setup

Full first-time setup (PyTorch CUDA build, SAM2, CoTracker, checkpoints, mediapipe/protobuf
pinning) is documented in [README.md](README.md) — follow it exactly, especially:
- Clone SAM2 into `sam2_repo/` (not `sam2/` — collides with the installed package name).
- `mediapipe==0.10.14` and `protobuf==3.20.3` are pinned for API compatibility; do not
  upgrade them independently.
- `requirements.txt` assumes torch/torchvision are already installed via the CUDA-specific
  command in the README (`pip install -r requirements.txt` alone will not set up CUDA
  correctly).

No test suite, linter, or build step exists in this repo — it's a research pipeline of
independent scripts, validated via the QC script and manual visual review (below).

## Running the Pipeline

Scripts are run in order from the project root; each stage's `--output` becomes the next
stage's input root. See [README.md](README.md) for full command lines and flags. Order:

1. `extract_frames.py` — video files → JPEG frames at fixed fps (`frames/<label>/<video>/0000.jpg`, ...). Frame filenames must stay plain zero-padded integers (SAM2's video loader requires this).
2. `annotate_prompts.py` — interactive, local-display only. Draws upper/lower garment bounding boxes on frame 0 of each video, saved to `prompts.json`.
3. `segment_video.py` — SAM2 video predictor, propagates the two box prompts across all frames → per-garment mask PNGs (`masks/<upper|lower>/<label>/<video>/frame_NNNN.png`).
4. `extract_pose.py` — MediaPipe Pose → `pose/<label>/<video>/keypoints.json`.
5. `sample_seed_points.py` — samples a grid of tracking points from garment masks, re-seeding a fresh grid every N frames (not just at frame 0) → `seed_points/<label>/<video>.json`.
6. `track_points.py` (or `track_points_lk.py`) — tracks each re-seed group's points forward until the next re-seed boundary → `trajectories/<label>/<video>.json`.
7. `qc_trajectories.py` — automated pass/fail check against the trajectories, independent of manual review.

Utilities `visualize_frames.py`, `visualize_masks.py`, `visualize_pose.py`,
`visualize_trajectories.py` render each stage's output back onto frames for manual spot
checks (used whenever `qc_trajectories.py` flags a video, or after any change to an
earlier stage). `fyp_pipeline_colab.ipynb` runs phases 1-3 on Colab's free T4 GPU via
Google Drive, for when local GPU/VRAM is the blocker.

## Architecture Notes

**Two-tracker design (GPU vs CPU fallback).** `track_points.py` uses CoTracker3
(deep-learning, GPU, more accurate) and `track_points_lk.py` uses classical
Lucas-Kanade optical flow (CPU-only via OpenCV, no model/GPU needed, less accurate). Both
read the same `seed_points/` input and both must produce trajectory JSON that
`qc_trajectories.py` can score, but their internal per-video output shape differs:
CoTracker's output segments trajectories by re-seed group (`{"segments": [{"seed_frame":
N, "points": [...]}]}`), while the LK version tracks continuously across the whole video
per garment (`{"points": [...]}`) since LK doesn't need the segment-boundary tracking that
periodic re-seeding introduces. `qc_trajectories.py` currently only understands the
segmented (CoTracker) format.

**CoTracker's visibility threshold is monkey-patched.** `track_points.py` subclasses
`CoTrackerPredictor` and copies `_compute_sparse_tracks` verbatim from upstream
co-tracker, changing only the hardcoded `0.9` confidence threshold to `VISIBILITY_THRESHOLD
= 0.6` (not exposed as a constructor parameter upstream). If upgrading the `cotracker`
package, diff this method against the new upstream source before assuming the patch still
applies cleanly — see the "QC script reports very low visibility" entry in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) for why this threshold matters.

**Periodic re-seeding + per-segment video slicing.** Seeding tracking points only once at
frame 0 loses them permanently once the body rotates the garment surface out of view.
`sample_seed_points.py` re-seeds a fresh point grid every `--reseed-interval` frames (from
that frame's real mask), and `track_points.py` tracks each re-seed group only from its
seed frame to the *next* re-seed boundary (or video end), slicing the video tensor to just
that span before calling CoTracker. Passing the full video on every re-seed group's call
was tried first and multiplied GPU compute ~10x per video — always slice before tracking
when touching this code path.

**SAM2 config paths must stay package-relative, never resolved to absolute paths.**
`segment_video.py` deliberately does NOT call `.resolve()` on `--config` — SAM2's Hydra
config loader expects a name relative to the installed `sam2` package (e.g.
`configs/sam2.1/sam2.1_hiera_s.yaml`), not a filesystem path. The script separately
verifies the file exists inside the installed `sam2` package directory as a sanity check,
decoupled from cwd. Don't "fix" this by resolving the path — see TROUBLESHOOTING.md's
Hydra entry.

**Box prompts, not point prompts.** Both `annotate_prompts.py` and `segment_video.py` use
two bounding boxes per video (upper garment = SAM2 object ID 1, lower = object ID 2)
rather than single-point clicks, because point prompts were unreliable (SAM2 would grab
the whole body or produce empty masks). Don't reintroduce point-based prompting without
re-reading the relevant TROUBLESHOOTING.md entry.

**Video/label discovery convention.** Every stage script independently walks
`<root>/<label>/<video_name>` for `label in ("real", "fake")` and derives its own list of
videos to process from whatever directory exists at that stage (e.g. `track_points.py`
discovers videos from `seed_points/`, `qc_trajectories.py` from `trajectories/`). There is
no shared manifest — if you add a new stage, follow the same `discover_videos()` pattern
already present in each script rather than inventing a new one.

**Failure isolation.** Each per-video processing stage wraps its main loop in a
try/except that logs to `failures.txt` (Phase 2) or `phase3_failures.txt` (Phase 3) and
continues to the next video, rather than aborting a whole batch run over one bad video.
Preserve this pattern in new stages — batch runs over many videos are the normal use case.

## Known Environment Constraints

Developed on an 8GB VRAM laptop GPU. GPU-heavy stages (`segment_video.py`,
`track_points.py`) are sensitive to concurrent GPU usage — see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) for real incidents (Explorer thumbnail view
triggering a driver TDR reset mid-run, external-monitor-dependent GPU visibility on this
hybrid-graphics laptop, PyTorch/driver CUDA version mismatches producing misleading OOM
errors). Check TROUBLESHOOTING.md before assuming a new CUDA/memory error is a fresh bug —
there is a good chance a similar one was already diagnosed there.

## Data Not in the Repository

`dataset/`, `frames/`, `masks/`, `pose/`, `seed_points/`, `trajectories/`,
`checkpoints/`, `cotracker_checkpoints/`, `sam2_repo/`, `cotracker_repo/`, and QC/review
folders are gitignored (regenerated locally or on Colab). Don't assume these exist; check
before writing code that reads them directly.
