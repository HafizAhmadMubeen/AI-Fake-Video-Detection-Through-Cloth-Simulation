# AI Fake Video Detection Through Cloth Simulation

This project detects AI-generated videos of humans by analyzing whether clothing
motion obeys real physics, rather than relying on generator-specific visual artifacts.
Since fabric dynamics (gravity, stretch, draping, collision with the body) are governed
by well-understood physical principles that generative video models don't explicitly
simulate, this approach aims to be generator-agnostic — able to generalize to newer or
unseen generators rather than overfitting to one model's cosmetic fingerprints.

## Pipeline Status

- [x] Phase 1: Frame extraction from video
- [x] Phase 2: Cloth segmentation (SAM2 video predictor, upper/lower garment masks) and body pose extraction (MediaPipe)
- [ ] Phase 3: Cloth point tracking across frames
- [ ] Phase 4: Physics-based cloth simulation and residual computation
- [ ] Phase 5: Classifier training and evaluation
- [ ] Phase 6: Demo interface

## Requirements

- Python 3.10
- An NVIDIA GPU with CUDA support (developed and tested on an 8GB VRAM laptop GPU,
  using SAM2's "small" checkpoint to fit that budget)
- Git
- Roughly 5-10GB free disk space for dependencies, plus additional space for your own
  dataset and the outputs the pipeline generates

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
If this fails partway through with a `MemoryError` during pip's caching step (a known
issue with very large wheel files), retry with:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
```
Verify the install:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
This should print `True` for CUDA availability. If it prints `False`, resolve that
before continuing — everything downstream will be extremely slow on CPU.

**3. Clone and install SAM2**

Important: clone it into a folder named `sam2_repo`, **not** `sam2`. Cloning it as
`sam2` causes a Python import collision with the installed `sam2` package (Python gets
confused between the repo folder and the package of the same name).
```powershell
git clone https://github.com/facebookresearch/sam2.git sam2_repo
cd sam2_repo
pip install -e .
cd ..
```

**4. Download the SAM2.1 small checkpoint**

The "small" checkpoint is used specifically to fit an 8GB VRAM budget — if you have a
GPU with significantly more VRAM, you can substitute a larger SAM2.1 checkpoint.
```powershell
mkdir checkpoints
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" -OutFile "checkpoints/sam2.1_hiera_small.pt"
```

**5. Config files**

The `configs/` folder is already included in this repo (originally copied from
`sam2_repo/sam2/configs`), so no extra step is needed here.

**6. Install remaining dependencies**
```powershell
pip install opencv-python
pip install mediapipe==0.10.14
```
Note: this project pins `mediapipe==0.10.14` deliberately. Newer releases (0.10.31+)
have a known open bug where `mp.solutions` is missing or broken
(`AttributeError: module 'mediapipe' has no attribute 'solutions'`).

## Preparing Your Dataset

`dataset/` is not included in this repository (raw video files are excluded via
`.gitignore` due to size and licensing). Create it yourself with this structure before
running anything:

```
dataset/
  real/    <- your real video files (.mp4, .mov, .avi, .mkv, .webm)
  fake/    <- your AI-generated video files (same extensions)
```

## Running the Pipeline

Run these in order:

**1. Extract frames from your videos**
```powershell
python extract_frames.py --input dataset --output frames --fps 10 --width 512
```
Samples frames from each video at a fixed rate, resizes them, and saves them as
`frames/<real_or_fake>/<video_name>/0000.jpg`, `0001.jpg`, etc.

**2. Annotate garment click-points**
```powershell
python annotate_prompts.py --frames frames --output prompts.json
```
Interactive: for the first frame of each video, click once on the upper-body garment
and once on the lower-body garment. Saves the coordinates to `prompts.json`, used to
prompt SAM2 in the next step.

**3. Segment upper and lower garments**
```powershell
python segment_video.py --frames frames --prompts prompts.json --output masks --checkpoint checkpoints/sam2.1_hiera_small.pt --config configs/sam2.1/sam2.1_hiera_s.yaml
```
Uses SAM2's video predictor to track the upper and lower garment masks across every
frame of each video, saved to `masks/upper/` and `masks/lower/`.

**4. Extract body pose**
```powershell
python extract_pose.py --frames frames --output pose
```
Runs MediaPipe Pose on every frame, saving 33 body landmarks per frame to
`pose/<real_or_fake>/<video_name>/keypoints.json`.

## Project Structure

| File | Purpose |
|---|---|
| `extract_frames.py` | Phase 1 — extracts frames from raw videos at a fixed fps |
| `rename_frames.py` | Utility to fix old-format `frame_XXXX.jpg` filenames to plain numeric names (not needed for fresh runs) |
| `annotate_prompts.py` | Interactive click-tool for SAM2 seed points per video |
| `segment_video.py` | Runs SAM2 video predictor to produce upper/lower garment masks |
| `extract_pose.py` | Runs MediaPipe Pose to extract skeleton keypoints per frame |

**Folders included in this repo:** the scripts above, `configs/`, `prompts.json`,
`README.md`, `.gitignore`.

**Folders excluded from this repo (generated locally):** `dataset/`, `frames/`,
`masks/`, `pose/`, `checkpoints/`, `sam2_repo/`, `sanity_checks/`, `fyp_env/`.

## Troubleshooting

- **PowerShell line continuation** uses a backtick `` ` `` at the end of the line, not
  a backslash `\` like Bash — using `\` causes a "Missing expression after unary
  operator" parser error.
- **`RuntimeError` about running Python from the parent directory of the sam2 repo** —
  this means the cloned folder was named `sam2` instead of `sam2_repo` as instructed
  above, causing an import collision.
- **Windows "Access denied" renaming/moving a folder** — another terminal, editor, or
  File Explorer window likely has that folder open; close those first.
- **CUDA out-of-memory error from SAM2 despite free VRAM** (check with `nvidia-smi`) —
  can be a stuck WDDM GPU driver state after a previous crashed process. Rebooting
  typically resolves this.

## Note on Data

The `dataset/`, `frames/`, `masks/`, `pose/`, and `checkpoints/` folders are not stored
in this repository due to size. They are regenerated locally by following the Setup
and Running the Pipeline steps above.
