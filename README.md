# AI Fake Video Detection Through Cloth Simulation

This project detects AI-generated videos of humans by analyzing whether clothing
motion obeys real physics, rather than relying on generator-specific visual
artifacts. Fabric dynamics (gravity, stretch, draping, collision with the body)
are governed by well-understood physical principles that generative video
models don't explicitly simulate, so this approach aims to be
generator-agnostic — able to generalize to newer or unseen generators rather
than overfitting to one model's cosmetic fingerprints.

## Pipeline Status

- [x] Phase 1: Frame extraction from video (30fps)
- [x] Phase 2: Cloth segmentation (SAM2 video predictor, box-prompted upper/lower
      garment masks) and body pose extraction (MediaPipe)
- [x] Phase 3: Cloth point tracking with periodic re-seeding (CoTracker3)
- [ ] Phase 4: Physics-based cloth simulation and residual computation
- [ ] Phase 5: Classifier training and evaluation
- [ ] Phase 6: Demo interface

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for a full log of problems hit
during development, what was tried, and what actually fixed each one.

## Requirements

- Python 3.10
- An NVIDIA GPU with CUDA support (developed on an 8GB VRAM laptop GPU; Phase 3
  tracking is also runnable on Google Colab's free T4 GPU — see below)
- Git
- Roughly 5-10GB free disk space for dependencies, plus space for your dataset
  and generated outputs

## Setup (Windows PowerShell)

**1. Create and activate a virtual environment**
```powershell
python -m venv fyp_env
.\fyp_env\Scripts\Activate.ps1
```

**2. Install PyTorch with CUDA support**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
If this fails partway with a `MemoryError` during pip's caching step, retry with:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
```
Verify:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Should print `True`. If your GPU driver reports a newer CUDA version (check via
`nvidia-smi`), use a newer build tag (e.g. `cu124`, `cu126`) instead of `cu121`.

**3. Clone and install SAM2**

Clone into `sam2_repo`, **not** `sam2` — naming it `sam2` causes a Python import
collision with the installed package.
```powershell
git clone https://github.com/facebookresearch/sam2.git sam2_repo
cd sam2_repo
pip install -e .
cd ..
```

**4. Clone and install CoTracker**
```powershell
git clone https://github.com/facebookresearch/co-tracker.git cotracker_repo
cd cotracker_repo
pip install -e .
cd ..
```

**5. Download checkpoints**
```powershell
mkdir checkpoints
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" -OutFile "checkpoints/sam2.1_hiera_small.pt"

mkdir cotracker_checkpoints
Invoke-WebRequest -Uri "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth" -OutFile "cotracker_checkpoints/scaled_offline.pth"
```

**6. Copy SAM2 configs**
```powershell
Copy-Item -Recurse sam2_repo/sam2/configs configs
```

**7. Install remaining dependencies**
```powershell
pip install opencv-python
pip install mediapipe==0.10.14
pip install protobuf==3.20.3
```
`mediapipe` is pinned to `0.10.14` (newer releases have a broken `mp.solutions`
attribute). `protobuf` is pinned to `3.20.3` since newer versions conflict with
this mediapipe version's expected API.

## Preparing Your Dataset

`dataset/` is not included in this repository. Create it yourself:
```
dataset/
  real/    <- real video files (.mp4, .mov, .avi, .mkv, .webm)
  fake/    <- AI-generated video files (same extensions)
```

## Running the Pipeline

**1. Extract frames (30fps)**
```powershell
python extract_frames.py --input dataset --output frames --fps 30 --width 512
```

**2. Annotate garment box prompts** (interactive, needs a display — run locally)
```powershell
python annotate_prompts.py --frames frames --output prompts.json
```
Click-and-drag a box around the upper garment, then the lower garment, per
video. This project uses box prompts rather than single-point clicks — points
proved unreliable (see TROUBLESHOOTING.md). Use `--redo` to re-annotate videos
already in an existing `prompts.json`.

**3. Segment garments with SAM2**
```powershell
python segment_video.py --frames frames --prompts prompts.json --output masks --checkpoint checkpoints/sam2.1_hiera_small.pt --config configs/sam2.1/sam2.1_hiera_s.yaml
```

**4. Extract body pose**
```powershell
python extract_pose.py --frames frames --output pose
```

**5. Sample seed points (with periodic re-seeding)**
```powershell
python sample_seed_points.py --masks masks --output seed_points --grid-size 8 --reseed-interval 15
```
Re-seeds a fresh grid of points every 15 frames so points lost to body
rotation are replaced by fresh points on newly visible garment surface.

**6. Track points with CoTracker**
```powershell
python track_points.py --frames frames --seed-points seed_points --output trajectories --checkpoint cotracker_checkpoints/scaled_offline.pth
```
Use `--limit N` to test on a small batch first (recommended after any
environment change).

**7. Run the QC check**
```powershell
python qc_trajectories.py --trajectories trajectories --masks masks --frames frames
```
Reports visibility rate, mask-containment rate, and jump rate per video/garment
against 90% thresholds. Flagged videos are candidates for closer review with
`visualize_frames.py`.

## Running Phase 3 Tracking on Google Colab

If local GPU issues block `track_points.py`, the same script runs on Colab's
free T4 GPU. Use `fyp_pipeline_colab.ipynb`, which mounts Google Drive so
checkpoints and data persist across session disconnects. See the notebook's
own markdown cells for setup — in short: create `MyDrive/FYP/` with your
`dataset/` and `prompts.json`, upload the pipeline scripts into
`MyDrive/FYP/scripts/`, then run the notebook's cells in order.

## Project Structure

| File | Purpose |
|---|---|
| `extract_frames.py` | Phase 1 — extracts frames from raw videos at a fixed fps |
| `annotate_prompts.py` | Phase 2 — interactive box-prompt annotation tool (local only) |
| `segment_video.py` | Phase 2 — SAM2 video predictor, upper/lower garment masks |
| `extract_pose.py` | Phase 2 — MediaPipe Pose skeleton extraction |
| `sample_seed_points.py` | Phase 3 — samples garment tracking points with periodic re-seeding |
| `track_points.py` | Phase 3 — CoTracker3 tracking (adjusted visibility threshold, per-segment slicing) |
| `qc_trajectories.py` | Phase 3 — automated QC checks on tracked trajectories |
| `visualize_frames.py` | Utility — saves annotated frame images for manual visual review |
| `fyp_pipeline_colab.ipynb` | Consolidated Colab notebook for Phases 1-3 via Google Drive |

**Folders excluded from this repo (generated locally, see `.gitignore`):**
`dataset/`, `frames/`, `masks/`, `pose/`, `seed_points/`, `trajectories/`,
`checkpoints/`, `cotracker_checkpoints/`, `sam2_repo/`, `cotracker_repo/`,
`fyp_env/`, `sanity_checks/`, `frame_review/`, `visualizations/`.

## Note on Data

Generated data folders are not stored in this repository due to size. They are
regenerated locally (or on Colab) by following the steps above.