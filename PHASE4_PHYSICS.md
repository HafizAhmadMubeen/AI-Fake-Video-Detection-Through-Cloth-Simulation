# Phase 4 Physics — The Equations Behind Fake-Video Detection

This file explains, in plain language, exactly which physics equations Phase 4 uses,
what each one means, and — most importantly — how it feeds into the actual goal of
the project: telling a real video from an AI-generated one by checking whether the
clothing obeys real-world physics.

The core idea of the whole project lives here: **we don't need to know what "correct
cloth" looks like abstractly — we simulate what correct cloth would do, given the
same body movement, and compare it to what the AI (or camera) actually produced.**
The bigger the gap, the more likely the video is fake.

---

## ⚠ Status after testing (read this first)

The equations below are correct and are what the code implements. But testing
on the current 14-video dataset changed what can be claimed about them. Full
details are in `PHASE4_LOG.md`, Experiments 4–9.

- **The simulation did not beat a no-physics baseline.** Compared against
  "clothes glued to the body", the mass-spring simulation predicted real cloth
  motion slightly *worse* (skill 0.925, where above 1 means physics helps).
- **The early residual separation was mostly camera framing.** Fake subjects
  are about 2× larger in frame, so every pixel-based measure — including the
  residual — came out ~2× larger on fakes. Once divided by torso length, the
  residual no longer separates the classes (video-level AUC 0.556).
- **The "~2 px vs 22–33 px" figure in Section 6 predates these fixes.** At the
  time, 100% of points on real videos were pinned to the skeleton, so no
  physics was running on them at all. Treat that number as historical.
- **Stiffness and damping** turned out to be material properties that vary per
  garment, not constants (Section 3, Section 4). They are now fitted per video.
- **The approach has since shifted** toward material-independent physical laws
  — rules every fabric obeys regardless of stiffness, such as "loose fabric
  hangs down, not up" (Section 7b). These need no simulation at all.

None of this shows the idea is wrong. The current videos are too small in
frame (real torsos ~62 px), mostly tight athletic wear, framed differently
between classes, and too few (9 real, 5 fake) to test it. A dataset built
for the purpose is the next step.

---

## 1. The one equation underneath everything

**Newton's Second Law:**

```
F = m × a
```

Every piece of cloth motion in our simulator — and in real life — comes from this
single equation: force causes acceleration, proportional to mass. We rearrange it as:

```
a = F / m
```

so that once we know all the forces acting on a tracked cloth point, we know exactly
how it should accelerate. Everything else in Phase 4 (gravity, springs, damping) is
just "what goes into F."

**Why it matters for detection:** if an AI generator draws cloth that moves in ways
no force could produce — snapping, floating, or stretching with no physical cause —
our simulator (which *only* moves points using real forces) will diverge from what
was observed. That divergence is our signal.

---

## 2. Force 1 — Gravity

```
F_gravity = (0, g)      where g = 800 px/s²
```

Gravity pulls every non-anchored point straight down at a constant rate. This is the
simplest force and the most obviously "cheatable" one for a generative model — loose
clothing (a shirt hem, a skirt edge) should sag and swing downward under gravity when
not held up by the body.

**Why it matters for detection:** AI-generated cloth often floats, clips through the
body, or hangs at physically implausible angles because the generator has no concept
of gravity — it just learned "what cloth looks like" from pixels, not from physics.

---

## 3. Force 2 — Springs (Hooke's Law)

Cloth is modeled as a mesh of points connected by springs (built via k-nearest-
neighbors, `k = 4`, in `build_knn_springs`). Each spring tries to keep its two
endpoints at their original ("rest") distance apart:

```
stretch = current_distance − rest_length
F_spring = k × stretch × direction
```

where `k = 40` is the stiffness constant and `direction` is the unit vector between
the two points. If the points are farther apart than rest, the spring pulls them back
together; if closer, it pushes them apart.

**Why it matters for detection:** this is what makes the mesh behave like *fabric*
instead of a cloud of independent dots — points move together, stretching only a
little, the way real woven material does. Real cloth doesn't stretch by 300% in one
frame; if our tracked points from the video imply that kind of stretch, either the
tracking failed or the cloth in the video isn't behaving physically (this also
becomes its own hand-crafted feature — see Section 7).

---

## 4. Force 3 — Damping

```
velocity *= (1 − damping × dt)      where damping = 6
```

Damping is friction against motion — it drains energy out of the system every frame
so the cloth doesn't jitter or oscillate forever. Real fabric has internal friction
and air resistance; without damping, a spring-mass system oscillates indefinitely
like an ideal (frictionless) spring, which looks nothing like real cloth.

**Why it matters for detection:** damping is what makes our simulated cloth settle
down naturally after a movement, the way real clothing does. It keeps the simulation
physically believable so that any gap between simulation and the real video is
meaningful (caused by the video), not an artifact of our simulator jittering.

---

## 5. Turning force into motion — numerical integration

Force alone doesn't move anything — we need to update position and velocity frame by
frame. We use **semi-implicit (symplectic) Euler integration**, run over 4 substeps
per frame for stability:

```
v_new = v_old + a × dt
x_new = x_old + v_new × dt
```

Note the key detail: velocity is updated *first*, then the *new* velocity is used to
update position (not the old one) — that's what makes it "semi-implicit," and it's
noticeably more stable than plain (explicit) Euler for spring systems.

**Anchor points are a special case:** any tracked point close enough to a real
MediaPipe body joint (within `ANCHOR_THRESHOLD = 60px`) is treated as "worn on the
body" — it skips the force/integration step entirely and is just moved exactly where
the real skeleton says that joint moved. Only the free-hanging points (hems, sleeves,
loose fabric) are actually simulated with F = ma.

**Why it matters for detection:** this is what connects our physics engine to the
*real* body motion in the video — the cloth reacts to the actual, real skeleton
movement extracted by MediaPipe, not to some generic animation. So the simulation
answers a very specific question: *"given how this person's body really moved, how
should the free parts of their clothing have moved?"*

---

## 6. The Residual — comparing simulation to reality

This is the primary detection signal. For every tracked point, at every frame:

```
residual = sqrt( (x_sim − x_observed)² + (y_sim − y_observed)² )
```

(the standard Euclidean/Pythagorean distance formula.) We then average this over all
points and frames in a segment to get `residual_mean_px`.

**Why it matters for detection:** this is the heart of the whole method. `x_observed,
y_observed` come from CoTracker's tracking of the *actual* video. `x_sim, y_sim` come
from our physics simulator, driven by the same body motion. If the video is real, its
cloth motion was generated by actual physics, so it should closely match what our
simulator predicts — low residual. If the video is AI-generated, the cloth motion
came from a generative model with no physical constraints, so it's very unlikely to
match a genuine physics simulation — high residual.

Our Experiment 1 proof-of-concept confirmed this directly: real videos measured
~2px residual, chaotic/fake-like segments measured ~22–33px — a >10x gap.

---

## 7. Hand-crafted features (built on top of the same equations)

Residual alone is powerful, but Phase 4 also extracts three supporting features from
the same simulated vs. tracked data, each targeting a different way fake cloth
"gives itself away."

### a) Stretch / strain

```
stretch = current_distance − rest_length   (per spring edge, from the tracked/observed points)
```

We compute this directly on the *observed* (CoTracker) trajectories using the same
mesh connectivity as the simulator, then take its mean, std, and max magnitude over a
segment.

**Why it matters:** real fabric barely stretches (it's usually treated as near-
inextensible in cloth simulation). If the observed points imply large or wildly
varying stretch, that's a sign the "cloth" in the video isn't respecting basic
material constraints — a common AI generation artifact (fabric warping, merging,
or morphing between frames).

### b) Drape angle vs. gravity

For each edge between a free (non-anchored) point and its neighbor, we compute the
angle between the edge and straight-down (gravity's direction):

```
cos(θ) = (edge_vector · gravity_vector) / (|edge_vector| × |gravity_vector|)
θ = arccos(cos θ)
```

We use a stricter anchor threshold here (`DRAPE_ANCHOR_THRESHOLD = 25px`, tighter
than the simulator's 60px) so we only measure genuinely free-hanging edges — this was
the fix for the bug where almost every edge was being (wrongly) classified as
anchored and skipped.

**Why it matters:** loose, unsupported cloth (a hanging sleeve, a skirt hem) should
drape roughly toward vertical (small θ) when not being pulled by the body. If the
observed cloth's free edges point sideways or upward with no corresponding body
force explaining it, that violates gravity — again, something a real garment
physically cannot do, but a generative model has no trouble drawing.

### c) Velocity & acceleration smoothness

We take the tracked positions and differentiate twice, using finite differences:

```
velocity[t]     = (position[t] − position[t−1]) / dt
acceleration[t] = (velocity[t] − velocity[t−1]) / dt
```

Before differentiating, we apply a 3-frame moving-average smoothing pass
(`smooth_positions`, window = 3) — this was the fix for the bug where tiny tracking
jitter was being amplified into huge, meaningless acceleration spikes (differentiating
twice roughly *squares* the effect of noise). After smoothing, we report the mean and
std of acceleration magnitude over the segment.

**Why it matters:** real cloth accelerates smoothly — it's a continuous physical
process governed by the same F = ma from Section 1. Sudden, erratic acceleration
spikes in the tracked points (after removing ordinary tracking noise) suggest motion
that isn't being driven by consistent forces frame-to-frame — again more consistent
with a generative model "hallucinating" cloth position frame-by-frame than with an
object obeying Newtonian mechanics continuously through time.

---

## 8. Summary — what each equation contributes

| Equation / concept | Used for | What it detects if violated |
|---|---|---|
| F = m·a (Newton's 2nd law) | Foundation of the whole simulator | Any cloth motion with no physical cause |
| Gravity (F = (0, g)) | Pulls free cloth downward | Cloth that floats, clips, or hangs at impossible angles |
| Hooke's Law (spring force) | Keeps mesh points a realistic distance apart | Cloth that stretches, tears, or morphs unrealistically |
| Damping | Settles motion naturally, stabilizes simulation | N/A (keeps our simulator physically realistic as a baseline) |
| Semi-implicit Euler integration | Turns forces into actual point motion, frame by frame | N/A (the mechanism that produces our simulated trajectory) |
| Anchors (skeleton-driven points) | Ties simulation to the real, observed body motion | Ensures comparison is "same body motion, does the cloth match?" |
| Residual (Euclidean distance) | **Primary fake-detection signal** | Cloth motion that doesn't match physics-driven prediction |
| Stretch / strain | Hand-crafted feature | Cloth stretching beyond what real fabric allows |
| Drape angle vs. gravity | Hand-crafted feature | Free-hanging cloth not obeying gravity |
| Velocity/acceleration smoothness | Hand-crafted feature | Erratic, non-physical frame-to-frame motion |

**In one sentence:** we use Newtonian mechanics to predict how clothing *should* move
given a person's real body motion, and then measure how far the video's actual
clothing motion strays from that prediction — because AI video generators learn to
imitate the *appearance* of cloth, not the physics that produces it.
