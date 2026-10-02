# Phase 4 Log: Physics Simulation and Residual Computation

A running log of the design decisions, experiments, and outcomes for this
phase -- kept separate from TROUBLESHOOTING.md (which is for bugs/errors)
because this is about *why* choices were made, not what broke. Append to
this as the phase continues.

---

## Goal

Turn Phase 3's tracked cloth motion into physics-based features a classifier
can learn from. Two categories of output:
1. A **physics residual**: how different is the real observed cloth motion
   from what a physics simulation predicts, given the same real body motion.
2. **Hand-crafted features** (stretch, drape angle, velocity smoothness) --
   not yet built, planned for after the residual is validated.

---

## Design decisions (made before writing any code)

- **Mesh connectivity**: connect each tracked point to its ~4 nearest
  neighbors with a spring, rather than assuming a regular grid. Chosen
  because seed points are filtered to fall inside the (irregular) garment
  mask, so they don't form a perfect lattice.
- **Anchor selection**: a tracked point becomes an "anchor" (driven directly
  by real skeleton motion, not freely simulated) if it falls within a
  distance threshold (60px) of a MediaPipe joint at the segment's seed
  frame. Points with no nearby joint are simulated freely.
- **Simulator scope**: deliberately minimal -- gravity, spring stretch
  resistance, and velocity damping only. No bending stiffness, no
  self-collision, no fabric-specific material properties. This is a known,
  accepted simplification for the MVP (see "Open question" below).
- **Validation approach**: proof-of-concept on 1-2 videos first (one calm,
  one chaotic), with both a visual overlay and a hard numeric check, before
  running the full batch. Chosen specifically to catch a failure mode we'd
  already worried about: a simulation that just "follows the skeleton"
  everywhere and produces a meaningless near-zero residual regardless of
  video content.

---

## Experiment 1: Proof-of-concept on two contrasting videos

**Videos used**: `real/man doing ghost rope` (calm, repetitive motion) vs.
`fake/man dancing` (fast, chaotic motion, also one of Phase 3's
harder-to-track videos).

**Result**:

| Video | Mean gap (raw pixels) |
|---|---|
| real/man doing ghost rope | ~2 px |
| fake/man dancing | ~22-33 px |

**Interpretation**: the ~15x difference confirmed the simulator responds
proportionally to how much real, complex motion is happening -- it isn't
just parroting the skeleton with a fixed offset (which would have produced
a similarly tiny gap on both videos). This was the specific failure mode we
were checking for, and it didn't happen. Treated as a pass to proceed to the
full batch.

**Caveat noted at the time**: `fake/man dancing` is also one of the
worst-tracked videos from Phase 3's QC report, so part of the larger gap
could reflect noisy tracking data rather than purely "more physically
implausible" motion. Not resolved, just flagged as a confound to keep in
mind when interpreting later results.

---

## Open question raised: does one fixed physics setting work across different clothing types?

Raised directly: a loose dress and a tight athletic shirt are physically
very different materials, but the simulator uses identical gravity/
stiffness/damping constants for every video regardless of garment type.

**Reasoning for why this might still be OK**: the simulator doesn't need to
know what fabric it's looking at -- it just needs to be applied
*consistently* across all videos, real and fake alike. The classifier
(Phase 5) learns from labeled data what a "normal" residual looks like
across the range of real footage, rather than us hand-tuning fabric physics
per garment type. Framed as "a consistent ruler," not "a fabric-aware
oracle."

**Agreed compromise**: proceed with one fixed parameter set for now (a
normal, acceptable MVP simplification), but test whether *normalizing* the
raw pixel gap by garment size makes real-vs-fake separation better or
worse, using the full 14-video batch as the test.

---

## Experiment 2: Raw gap vs. size-normalized gap, full 14-video batch

**Normalization tested**: divide each frame's raw pixel gap by that
segment's garment size, estimated as the bounding-box diagonal of the
observed points at the segment's first frame.

**Method**: ran the simulator once across all 14 videos, then computed both
metrics from the same simulation output (no need to re-run the simulator
itself for each metric). Compared using a simple separation score: the gap
between the real-video group's average and the fake-video group's average,
scaled by how spread out each group is (bigger score = cleaner separation
between real and fake).

**Result**:

| Metric | Real mean | Fake mean | Separation score |
|---|---|---|---|
| Raw gap (px) | 6.89 | 16.56 | **1.44** |
| Normalized gap | 0.0988 | 0.0832 | 0.34 |

**Outcome: normalization made separation WORSE, not better.** Raw gap won
clearly.

**Why, diagnosed after the fact**: the chosen denominator (bounding-box
size) isn't a stable "how big is this garment" measurement -- it's
recomputed fresh at every re-seed segment (every 15 frames), based on
wherever the limbs happen to be at that exact instant. Instead of removing
camera-distance/garment-size as a confounding variable, it introduced a
*second* noisy, motion-dependent number into the calculation, which
partially canceled out the real physics signal rather than isolating it.
This was a genuine attempt at a legitimate concern (different clothing
types aren't fairly comparable) that didn't work as implemented -- the
underlying concern is still valid, the specific fix wasn't.

**Decision**: proceed with the RAW pixel gap as the residual feature for
now. Do not use the bounding-box normalization. If normalization is
revisited later, use a more stable denominator -- e.g. the person's
shoulder-to-hip distance measured once at frame 0 of the whole video,
rather than a per-segment, per-instant bounding box.

**Individual video note**: `fake/Man jumping (2)` scored a raw gap of 1.74 --
lower than most REAL videos. This is a genuine miss (a fake whose cloth
motion looked physically plausible to this check). Not resolved; noted as
a known limitation of the residual alone, and a reason the classifier
(Phase 5) should combine this with the hand-crafted features rather than
rely on the residual in isolation.

---

## Experiment 3: Fixing drape and acceleration feature bugs

First run of `compute_features.py` (computing stretch, drape, and
velocity-smoothness features from the observed trajectories) surfaced two
real problems in the output:

**Bug 1 -- `drape_mean_angle_deg` almost entirely empty (~2 non-empty values
out of 140+ rows).**

Cause: the drape feature only measures edges between two points that are
NOT anchors (anchors are rigidly tied to real skeleton motion, so they
can't show independent "hanging" behavior). But the anchor threshold
(60px, same one used for the simulator itself) turned out to be far too
generous for how densely packed the 8x8 tracked-point grid is -- with
shoulders, elbows, wrists, hips, and knees all nearby, nearly every point
on a torso-sized garment ended up counted as an anchor, leaving almost no
qualifying edges.

Fix: introduced a SEPARATE, stricter `DRAPE_ANCHOR_THRESHOLD = 25px`,
decoupled from the simulator's own 60px threshold. The two serve different
purposes -- the simulator wants generous anchoring (keeps the physics
numerically stable, prevents unrealistic free-fall drift), while drape
specifically needs to isolate points that are genuinely NOT rigidly tied to
the body. Also loosened the edge-inclusion rule from "both points must be
non-anchor" to "at least one must be non-anchor," which alone wasn't
sufficient without the threshold change but helped once combined with it.

Result after fix: `drape_mean_angle_deg` now populated across most rows,
with sane, believable values clustering around 30-55 degrees (partial but
inconsistent drape toward vertical -- exactly what's expected from real
cloth on a MOVING body, not a mannequin, where arm swings and torso twists
constantly pull the "hang direction" away from pure vertical).

**Bug 2 -- `accel_mean_magnitude` values huge and inconsistent (ranged
~100 to over 13,000, dataset-wide).**

Cause: acceleration is a SECOND derivative of position (position -> velocity
-> acceleration). Each differentiation step divides by a small time step
(dt = 1/30s), and dividing by a small number twice amplifies any ordinary
tracking jitter (a few pixels of normal CoTracker noise) into a huge,
physically meaningless spike. This is a standard, well-known problem with
naive double-differentiation of noisy discrete position data, not a logic
bug.

Fix: added `smooth_positions()` -- a 3-frame moving average applied to the
visible-only position sequence before computing velocity, and again before
computing acceleration. Standard signal-processing fix: filters
high-frequency jitter while preserving genuine, slower motion trends.

Result after fix: values came down to a consistent ~70-2500 range across
the whole dataset, with the previous extreme outliers (13,000+, all on
poorly-tracked videos) gone.

**Both fixes confirmed in the actual output CSV before proceeding.**

---

## Experiment 4: Full separation analysis and correctness audit

Built `analyze_features.py` to measure every feature properly: per-feature
AUC, **video-level** results (segments from one video are not independent,
so video level is the honest unit), a correlation check, and a
leave-one-video-out classifier preview.

First run (v1 features, saved as `phase4_features_v1.csv`): best video-level
AUC was `accel_std_magnitude` 0.778, residual 0.733. No video-level p-value
below 0.05. Drape was missing on 36% of rows and, where present, sat at
42.6 deg for BOTH classes (AUC 0.508).

A line-by-line audit of `compute_features.py` and `run_simulation.py` then
found seven defects:

| Bug | Problem | Fix |
|---|---|---|
| A | Residual included frames where CoTracker had LOST the point (interpolated guesses) | Visible frames only |
| B | Smoothing window spanned occlusion gaps (e.g. frames 11, 12, 50) | Split into consecutive runs before smoothing |
| C | Drape averaged ALL mesh edges; horizontal (~90 deg) + vertical (~0 deg) = ~45 deg for every video. It measured mesh topology, not drape | Only edges hanging within 45 deg of vertical at rest |
| D | Drape empty whenever every point was anchored | Anchor filter made a preference with a fallback |
| E | Simulated lookup merged segments; re-seeded point IDs overwrote each other | One lookup per segment |
| F | Rest positions not checked for visibility | Checked: every seed frame was fully visible, so this never bit |
| G | Anchors translated with a joint but never rotated with the limb | Two-joint local frame: cloth rotates and scales with the limb |

Diagnostics added so confounds could be TESTED rather than assumed away:
`visible_frac` (tracking quality) and later `pose_dist_*`, `body_scale_px`.

---

## Experiment 5: Anchor audit -- the simulation was not simulating anything

With the 60px anchor threshold, the fraction of points actually left free
for physics was:

| Threshold | Free % (real) | Free % (fake) |
|---|---|---|
| 10 px | 67.4% | 77.5% |
| 20 px | 16.2% | 45.3% |
| 30 px | 0.8% | 26.0% |
| **60 px (used)** | **0.0%** | **8.0%** |

**On every real video, 100% of points were pinned to the skeleton.** Gravity
and springs never acted, so the "physics residual" on real video was
measuring only the kinematic anchor model. And real and fake had different
computations performed on them (0% vs 8% free), so the residual comparison
was not like with like. This also explains the original drape bug.

**Fix:** anchors chosen by RANK -- the closest fixed fraction of points to
joints -- so every segment has the same anchor/free split regardless of
label. Started at 30%; the user chose **20%** (80% freely simulated).

Pose-quality check: tracked points sit at the same distance from detected
joints once normalised by skeleton size (fake/real ratio 0.98-1.02), so
MediaPipe localises poses equally well on both classes.

---

## Experiment 6: Global calibration of gravity / stiffness / damping

`calibrate_physics.py` fitted the constants on **real videos only** (no label
leakage), with a guard rejecting degenerate "cloth welded to the body"
settings. Fake residuals were reported afterwards as a diagnostic only.

| Setting (g / k / d) | Real residual | Fake residual | Ratio | free_dyn |
|---|---|---|---|---|
| 800 / 40 / 6 (guessed) | 13.11 px | 17.33 px | 1.32x | 20.06 |
| 400 / 320 / 16 | 7.51 px | 15.72 px | 2.09x | 10.38 |
| 800 / 1280 / 16 | 6.96 px | 14.60 px | 2.10x | 8.38 |

- **Gravity** was unidentifiable: every value from 25 to 400 fitted within 1%
  (7.42-7.49 px). Leave-one-video-out picked anything from 100 to 400. Not
  instability -- the data simply cannot distinguish them, plausibly because
  athletic clothing is driven by the body far more than by its own weight.
- **Stiffness** had no interior optimum: error kept falling as stiffness rose
  (ran to 1280, the grid maximum). Quadrupling it from 320 to 1280 improved
  real AND fake fit by the same ~7%, leaving the ratio unchanged. This is
  the degeneracy signature: extra stiffness made the cloth track everything
  better, not physics better.
- **Decision:** 400 / 320 / 16, chosen on cloth-plausibility grounds (keeps
  free motion at about half its neutral value), not by fit.

---

## Experiment 7: The scale confound -- the separation was camera framing

After calibration, every PIXEL-valued feature separated real from fake by
almost exactly the same factor (~2x), while every DIMENSIONLESS feature
(strain ratio, drape angle, strain rate) separated by almost nothing. That
pattern is the signature of a scale difference, not four physics findings.

Measured directly: the subject's torso is **62 px tall in real videos versus
134 px in fakes** (`body_scale_px`, video-level AUC 0.867, p = 0.029). The
fake subjects are roughly twice as large in frame.

Dividing by torso length (per-video median shoulder-to-hip distance, the
user's chosen yardstick) removes it:

| Feature | Raw pixels (video AUC) | Normalised (video AUC) |
|---|---|---|
| residual_mean | 0.733 | 0.556 |
| residual_median | 0.756 | 0.533 |
| accel_mean | 0.689 | 0.578 |
| accel_std | 0.689 | 0.511 |

**Correction to Experiment 2:** raw pixels "won" over normalisation there
for the same reason -- raw pixels carried the framing difference. The
conclusion drawn at the time was wrong.

---

## Experiment 8: Per-video material fitting

User's proposal: stiffness and damping are properties of a particular
garment, not universal constants, so fit them **per video**. Fit on the
first 30% of each video's segments, evaluate on the remaining 70% (user's
choice), gravity kept global.

**Attempt 1 -- minimise prediction error:** 100% of fits ran to the edge of
the stiffness range. Minimising error always prefers stiffer, body-hugging
cloth, because hugging the body is a serviceable prediction for any video.

**Attempt 2 -- skill against a rigid baseline** (user's choice):
`skill = error of 'clothes glued to the body' / error of physics simulation`.
Above 1 means physics helps.

| | Skill (held-out 70%) | Fit sharpness (contrast) | At edge |
|---|---|---|---|
| Real | **0.925** | 1.59 | 100% |
| Fake | 0.868 | 1.40 | 100% |

**The physics simulation predicts cloth motion WORSE than simply gluing the
clothes to the body, on real footage.** Per-video skill for real videos
ranged from 0.30 to 1.70. Stiffness still ran to the ceiling -- now because
the best a losing model can do is become the baseline. The fit sharpness
gap (1.59 vs 1.40) was the most promising signal but is small on 14 videos.

Likely contributors: open-loop drift over ~50-frame rollouts (the baseline
is recomputed every frame, the simulation never is), 2D projection of 3D
cloth, tight athletic clothing with little free-hanging fabric, and very
small subjects in the real videos.

---

## Experiment 9: Material-independent gravity law

User's proposal: instead of simulating, check laws that EVERY fabric obeys
regardless of material. Implemented "loose fabric hangs down, not up":

- `gravity_inversion_rate` -- fraction of samples where an edge that hung
  downward at rest now points upward (past 90 deg; not a chosen threshold).
- `gravity_p95_angle_deg` -- how far from vertical hanging fabric gets at its
  most extreme moments. Threshold-free.

Validated on a controlled test: normal sway gives inversion 0.000; flipping
10% of frames reads 0.095; 30% reads 0.286.

Deliberately NOT used: correlating sway angle with angular acceleration. For
any bounded motion that correlation is negative by a mathematical identity
(integration by parts), so it would pass on fake video too.

Result on the dataset (video level): inversion AUC 0.600, p95 angle AUC
0.622. No separation.

---

## Conclusion of Phase 4

**The pipeline is now correct. This dataset cannot answer the research
question.**

Final video-level results, scale-free features only:

| Feature | AUC | p |
|---|---|---|
| strain_rate_mean | 0.689 | 0.30 |
| drape_mean_angle_deg | 0.667 | 0.36 |
| gravity_p95_angle_deg | 0.622 | 0.52 |
| gravity_inversion_rate | 0.600 | 0.59 |

The only statistically significant differences between the classes are
camera framing (`pose_dist_mean_px` 0.889, p = 0.019; `body_scale_px` 0.867,
p = 0.029). No physics feature separates real from fake once framing is
removed.

**Why this is a dataset limit, not evidence against the idea:**

1. **Resolution.** Real-video torsos are ~62 px tall, so tracked points sit
   only a few pixels apart and CoTracker's 1-2 px jitter is 15-25% of every
   measurement. `stretch_mean_abs` reads 12% for BOTH classes -- real fabric
   does not routinely stretch 12%, so that number is tracker noise.
2. **Clothing.** Mostly tight athletic wear, which moves with the body and
   leaves little free-hanging fabric for physics to act on.
3. **Framing mismatch.** Fakes are framed about 2x closer than reals.
4. **Sample size.** 9 real, 5 fake. No video-level result can reach
   significance at this size.
5. **Content mismatch.** Fakes are mostly dancing; reals mostly sport.

**Requirements for a dataset that could test the idea:**

- Loose, free-hanging clothing: dresses, skirts, loose shirts, scarves.
- Subject filling the frame: torso at least ~150 px tall.
- Matched framing and resolution between real and fake.
- 30+ videos per class, fakes from several different generators.
- Similar motions in both classes.

**Settings in the committed feature table:** anchors by rank, 20% for the
feature computations; simulated trajectories generated at 30% anchors with
gravity 400 / stiffness 320 / damping 16, rotation-aware anchors. Scale
normalisation by per-video median torso length.

---

## Files produced in this phase

| File | Purpose |
|---|---|
| `cloth_simulator.py` | Mass-spring engine (spring forces vectorised in v2) |
| `run_simulation.py` | Drives the simulator from the skeleton; rank-based, rotation-aware anchors; per-video materials |
| `compute_features.py` | Per-segment feature table, v2 (bugs A-F fixed, scale-normalised and gravity-law features) |
| `analyze_features.py` | Completeness, AUC, video-level, confound checks, correlation, LOO preview |
| `calibrate_physics.py` | Global calibration on real videos only, with anchor audit and degeneracy guard |
| `fit_material.py` | Per-video material fitting, 30/70 split, skill against a rigid baseline |
| `visualize_simulation.py` | Overlay of simulated vs observed points |
| `quick_residual_check.py`, `batch_residual_check.py` | Early residual checks (Experiments 1-2) |
| `phase4_features.csv` | Final feature table |
| `phase4_features_v1.csv`, `feature_analysis_v1.txt` | Pre-audit results, kept for comparison |
| `feature_analysis.txt`, `feature_plots.png` | Final analysis report and plots |
| `calibration_result.json`, `material_fits.json` | Calibration and material-fit outputs |
