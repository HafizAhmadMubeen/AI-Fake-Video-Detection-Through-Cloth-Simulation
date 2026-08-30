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

## Current status

- Physics residual (raw pixel gap) computed and validated across all 14
  videos, real videos clustering lower (mean 6.89, std 2.93) than fake
  videos (mean 16.56, std 10.47), with at least one known miss.
- Size normalization tried and rejected (see Experiment 2).
- Not yet built: hand-crafted features (stretch, drape angle vs. gravity,
  velocity/acceleration smoothness), and the final per-video feature vector
  aggregation Phase 5 will train on.

## Files produced in this phase so far

| File | Purpose | Status |
|---|---|---|
| `cloth_simulator.py` | Core mass-spring physics engine | Working |
| `run_simulation.py` | Wires simulator to real tracked points + skeleton | Working |
| `visualize_simulation.py` | Visual overlay of simulated vs observed points | Working |
| `quick_residual_check.py` | Single-video numeric gap check | Working |
| `batch_residual_check.py` | Full-dataset raw vs normalized comparison | Working |
| `residuals_comparison.csv` | Saved output of Experiment 2 | Reference data |
