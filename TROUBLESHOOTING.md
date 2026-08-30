# Troubleshooting Log

A record of real problems hit during development, organized by phase. For each
one: what the problem looked like, what was tried (including dead ends), and
what actually fixed it. Kept honest on purpose — some entries show an initial
fix that turned out to be wrong, because that's useful context for anyone
hitting a similar symptom later.

---

## Environment Setup

### PowerShell rejects multi-line commands
**Symptom:** `Missing expression after unary operator '--'` when pasting a
multi-line command ending each line with `\`.
**Cause:** PowerShell uses a backtick (`` ` ``) for line continuation, not a
backslash like Bash.
**Fix:** Use a backtick at the end of each line, or put the command on one line.

### `pip install torch` crashes with `MemoryError`
**Symptom:** Download completes (progress bar reaches 100%), then crashes deep
inside `pip`'s caching internals (`msgpack`/`cachecontrol`).
**Fix:** Retry with `--no-cache-dir`. pip was trying to buffer the entire large
wheel file in memory to write its cache; skipping the cache step avoids it.

### `ModuleNotFoundError: No module named 'sam2'`
**Symptom:** After `git clone` + `pip install -e .`, `import sam2` fails, or
`build_sam2_video_predictor` raises a `RuntimeError` about running from the
parent directory of the sam2 repo.
**Cause:** The cloned repo folder was named `sam2`, which is the same name as
the installed Python package — Python couldn't tell them apart.
**Fix:** Clone into a folder named `sam2_repo` instead, then `pip install -e .`
from there.

### Windows "Access denied" renaming the `sam2` folder
**Cause:** Another terminal, editor, or File Explorer window had the folder
open, locking it.
**Fix:** Close every other window referencing that folder, then retry.

---

## Phase 1 — Frame Extraction

### `extract_frames.py` produces 0 frames for one specific video
**Symptom:** Every `cv2.imwrite()` call for one video failed silently
(`Failed to write frame: frame_0000.jpg`, repeated), while other videos worked
fine.
**What was tried first:** Assumed a codec issue (H.265/HEVC) and suggested
re-encoding with ffmpeg — **this was the wrong diagnosis.**
**Actual cause:** The video's filename had a trailing space (`"man dancing
.mp4"`). Windows silently drops trailing spaces from path components at the OS
level, but Python's `mkdir()` and OpenCV's `imwrite()` don't handle that
stripping consistently — the folder that actually got created didn't match the
path OpenCV was writing to.
**Fix:** Strip whitespace from the video filename before building the output
path; renamed the offending source file to remove the trailing space.

### 10fps judged too coarse for physics features
**Cause:** Cloth oscillation/acceleration can happen faster than the 100ms gap
between frames at 10fps. The original rate was chosen for tracking
feasibility, not physics fidelity.
**Fix:** Increased to 30fps (`--fps 30`). Phases 1–3 were re-run at the higher
rate.

---

## Phase 2 — Segmentation and Pose

### SAM2 crashes: `invalid literal for int() with base 10: 'frame_0000'`
**Cause:** SAM2's video loader requires frame filenames to be pure integers
(`0000.jpg`), not prefixed names like `frame_0000.jpg`.
**Fix:** Updated `extract_frames.py` to save frames as plain zero-padded
numeric filenames directly.

### `RuntimeError: CUDA error: out of memory` loading SAM2, despite free VRAM
**Symptom:** `nvidia-smi` showed several GB free, but SAM2 still failed to
allocate a small tensor.
**What this really was:** A stuck/corrupted CUDA driver state left over from
an earlier crash, not genuine memory exhaustion (the error numbers were
internally inconsistent — a hallmark of this kind of driver issue).
**Fix:** A full shutdown-and-restart (not just closing the terminal) reset the
GPU driver state.

### `hydra.errors.MissingConfigException` / "Config not found"
**Symptom:** Passing `--config` as a full/absolute path (including a Google
Drive path on Colab) either got rejected by SAM2's Hydra-based config loader,
or passed the existence check but failed inside `build_sam2_video_predictor`.
**Cause:** SAM2's config loader expects a config **name** relative to the
installed `sam2` package (`pkg://sam2`), not a filesystem path. The script was
calling `.resolve()` on the argument, turning it into an absolute path Hydra
couldn't interpret as a package-relative name.
**Fix:** Stopped resolving `--config` into an absolute path — passed the raw
string through to Hydra unchanged (e.g. `configs/sam2.1/sam2.1_hiera_s.yaml`),
and separately verified the file's existence directly against the installed
`sam2` package's own location rather than the given path.

### Single-point click prompts unreliable
**Symptom:** SAM2 sometimes segmented the entire body instead of just the
garment, or produced an empty mask for the whole video.
**Cause:** A single point gives SAM2 no boundary information — it has to
guess where the object ends, and can over- or under-segment depending on
exactly where the point lands.
**Fix:** Rebuilt `annotate_prompts.py` and `segment_video.py` to use
**bounding-box prompts** instead of single-point clicks. Boxes give SAM2 an
explicit boundary and proved far more reliable on visual spot-checks
afterward.

### `AttributeError: 'FieldDescriptor' object has no attribute 'label'`
**Context:** Raised inside `mediapipe`'s `solution_base.py` when running
`extract_pose.py`.
**Cause:** A `protobuf` version conflict — a newer `protobuf` (pulled in as a
dependency of something else) is incompatible with this version of
`mediapipe`'s expected API.
**What was tried first:** Setting
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` as an environment variable —
did not fix it.
**Actual fix:** `pip install protobuf==3.20.3` to downgrade directly.

---

## Phase 3 — Point Tracking

### CUDA out of memory running CoTracker, inconsistent free-memory numbers
**Symptom:** `torch.OutOfMemoryError` trying to allocate ~100MB while the
error message itself reported multiple GB free.
**What was tried first (in order):** switching to a smaller CoTracker
checkpoint (`baseline_offline.pth`) — didn't fix it; switching from offline to
online tracking mode — reduced but didn't eliminate the problem, and
introduced a worse `CUDA error: unknown error` on retries.
**Actual root cause:** A version mismatch between the installed PyTorch build
(compiled for CUDA 12.1) and the machine's actual GPU driver (reporting CUDA
13.0 support).
**Fix:** Reinstalled PyTorch against a newer CUDA build:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir`.
After this, the **original** offline-mode, `scaled_offline.pth` setup worked
correctly — the earlier checkpoint/mode switching had been solving the wrong
problem.

### Laptop screen flickers black mid-run; CoTracker crashes
**Cause:** Opening Windows File Explorer's **thumbnail view** on an
image-heavy folder while CoTracker was already using most of the GPU's memory
triggered Windows TDR (Timeout Detection and Recovery), which resets the GPU
driver when it appears unresponsive. Desktop compositing and thumbnail
generation both use the GPU too, and there wasn't enough headroom left.
**Fix:** Not a code fix — an operational rule: avoid opening File Explorer
(especially thumbnail/large-icon view) on image-heavy folders while any
GPU-heavy script is running. Confirmed by reproducing the full batch cleanly
with File Explorer left untouched.

### `nvidia-smi` fails with "insufficient permissions" / mentions a non-NVIDIA GPU
**Cause:** On this hybrid-graphics laptop, the external monitor port is wired
directly to the discrete GPU. Running without the monitor connected let
Windows fall back to the integrated GPU, making the NVIDIA GPU appear
inactive/unreachable to `nvidia-smi` and PyTorch.
**Fix:** Reconnecting the external monitor restored GPU access immediately.

### `ValueError: cannot select an axis to squeeze out which has size not equal to one`
**Context:** Extracting CoTracker's visibility output after a successful
model forward pass.
**Cause:** The code assumed CoTracker's visibility tensor always had a
trailing dimension of size 1 to squeeze; this particular
checkpoint/version's output shape didn't match that assumption.
**Fix:** Made the shape handling defensive — only squeeze the trailing
dimension if it's actually size 1, then explicitly reshape to match the
tracks tensor's `(frames, points)` shape rather than assuming a fixed
dimensionality.

### QC script reports very low "visibility" (e.g. 14%) even on well-tracked videos
**Symptom:** A large, inconsistent gap between the visibility rate and the
independently-computed mask-containment rate — e.g. a video passing 85% on
mask-containment but only 14% on visibility.
**Diagnosis:** Ran `debug_cotracker_output.py` to inspect CoTracker's raw
output directly (rather than trusting the processed values). Found that
CoTracker's official predictor code hardcodes a confidence threshold of `0.9`
inside `_compute_sparse_tracks` — not exposed as an adjustable parameter, and
very conservative by default.
**Fix:** Subclassed `CoTrackerPredictor`, copying `_compute_sparse_tracks`
verbatim from the official source with only the threshold changed from `0.9`
to `0.6`. Visibility numbers moved in line with mask-containment afterward,
confirming the diagnosis.

### Re-uploading an updated script to Colab has no effect
**Symptom:** After fixing a bug and re-uploading the script, Colab produced
byte-identical output to the broken version.
**Cause:** Colab's file upload doesn't overwrite an existing file of the same
name — it saves the new one as `filename (1).py`, and the cell was still
executing the old file.
**Fix:** Delete the old file first (`!rm -f track_points.py`) before
re-uploading, or explicitly reference the new `(1)` filename.

### `MessageError: credential propagation was unsuccessful` on `drive.mount()`
**Cause:** Browser blocking third-party cookies, or an ad-blocker/privacy
extension interfering with the Google authentication popup.
**Fix:** Allowed third-party cookies for Google domains and retried
`drive.mount(force_remount=True)`.

### Points tracked well on calm videos but drifted off-body on fast-motion videos
**Symptom:** Confirmed visually (frame-by-frame image review) — tracked
points on a fast-rotating subject stayed frozen in one screen position while
the body visibly moved underneath, or drifted onto the background entirely.
**Cause:** Seeding tracking points only once, at frame 0, means a point is
permanently lost once the body rotates its physical surface out of camera
view — there's no way to recover a point whose surface isn't visible in any
frame, regardless of tracker quality.
**Fix:** Reworked `sample_seed_points.py` and `track_points.py` to re-seed a
fresh grid of points every 15 frames from that frame's real garment mask, so
lost points get replaced by new points on whatever surface is currently
visible.

### Re-seeding accidentally increased GPU memory pressure
**Symptom:** After adding re-seeding, local CoTracker runs became *more*
prone to memory issues, not less.
**Cause:** Each re-seed group's tracking call was passing the **entire video**
to CoTracker (just with a different starting query frame), meaning a
150-frame, 10-re-seed-group video did ~10x the compute of the original
single-pass approach.
**Fix:** Sliced the video tensor down to only the frames each segment actually
needs *before* calling CoTracker, instead of passing the full clip every time.
This cut per-call compute roughly 10x and made local runs both faster and more
memory-stable than before the rework.

---

## Summary Table (quick reference)

| Phase | Issue (short) | Final Fix |
|---|---|---|
| Setup | PowerShell line continuation | Use backtick, not backslash |
| Setup | pip MemoryError | `--no-cache-dir` |
| Setup | sam2 import collision | Clone as `sam2_repo` |
| Phase 1 | Silent 0-frame extraction | Strip trailing whitespace from filenames |
| Phase 1 | 10fps too coarse | Switched to 30fps |
| Phase 2 | SAM2 filename crash | Plain numeric frame filenames |
| Phase 2 | False CUDA OOM | Reboot (driver state) |
| Phase 2 | Hydra config error | Package-relative `--config`, not absolute path |
| Phase 2 | Bad masks from point prompts | Switched to box prompts |
| Phase 2 | protobuf/mediapipe crash | `pip install protobuf==3.20.3` |
| Phase 3 | Real CUDA OOM | Reinstalled PyTorch for cu126 |
| Phase 3 | Driver reset (TDR) | Don't browse image folders during GPU runs |
| Phase 3 | nvidia-smi permission error | Reconnect external monitor |
| Phase 3 | Visibility shape crash | Defensive reshape logic |
| Phase 3 | Overly strict visibility flag | Lowered CoTracker's threshold 0.9 → 0.6 |
| Phase 3 | Colab upload not applying | Delete old file before re-upload |
| Phase 3 | Points lost on rotation | Periodic re-seeding every 15 frames |
| Phase 3 | Re-seeding memory regression | Slice video per-segment before tracking |
