"""
Phase 4, Step 1: Core mass-spring cloth simulator.

A minimal, deliberately simple physics engine: point masses connected by
springs, under gravity and damping, integrated with semi-implicit Euler.
This is NOT meant to be a production-grade cloth simulator (no self-collision,
no bending stiffness, no implicit integration) -- it's meant to be just
detailed enough to produce a plausible "how should this cloth move" reference
to compare against real observed motion (the physics residual).

Some points are "anchors": instead of being freely simulated, their position
is driven directly by real tracked skeleton motion each frame (e.g. a point
near the shoulder moves exactly as the shoulder moves). This is what "drives"
the simulation -- everything else responds to gravity and spring forces
relative to the anchors.

Units note: positions are in PIXELS (matching the tracked point coordinates),
not meters. Gravity/stiffness/damping constants are tuned in pixel-space,
not physically "real" SI values -- they only need to produce plausible
RELATIVE motion, since we're comparing shapes of motion, not absolute physical
accuracy. Point mass is taken as 1, so F = m*a reduces to a = F.

---------------------------------------------------------------------------
v2 change: _spring_forces() is vectorised.
---------------------------------------------------------------------------
The original looped over every spring in Python. That is ~128 springs x 4
substeps x ~50 frames x 180 segments = ~4.6 million iterations for ONE pass
over the dataset, which made parameter calibration (dozens of passes)
impractical. The vectorised version computes all spring forces at once with
numpy and accumulates them with bincount. The maths is identical -- only the
order of summation differs, so results match to floating-point rounding.
"""

import numpy as np


class ClothSimulator:
    def __init__(
        self,
        rest_positions: np.ndarray,      # (N, 2) starting positions
        springs: list[tuple[int, int, float]],  # (i, j, rest_length)
        anchor_mask: np.ndarray,         # (N,) bool -- True = driven externally
        gravity: float = 800.0,          # pixels/s^2, downward (+y)
        stiffness: float = 40.0,         # spring stiffness constant
        damping: float = 6.0,            # velocity damping (higher = settles faster)
    ):
        self.positions = rest_positions.astype(np.float64).copy()
        self.velocities = np.zeros_like(self.positions)
        self.springs = springs
        self.anchor_mask = anchor_mask.astype(bool)
        self.gravity = gravity
        self.stiffness = stiffness
        self.damping = damping

        # Pre-split the spring list into flat arrays once, so the per-substep
        # force computation is pure numpy (see module docstring).
        if springs:
            self.spring_i = np.asarray([s[0] for s in springs], dtype=np.intp)
            self.spring_j = np.asarray([s[1] for s in springs], dtype=np.intp)
            self.spring_rest = np.asarray([s[2] for s in springs], dtype=np.float64)
        else:
            self.spring_i = np.empty(0, dtype=np.intp)
            self.spring_j = np.empty(0, dtype=np.intp)
            self.spring_rest = np.empty(0, dtype=np.float64)

    def set_anchor_positions(self, anchor_positions: np.ndarray) -> None:
        """Directly set the position of anchor points (kinematic, not simulated)."""
        self.positions[self.anchor_mask] = anchor_positions
        self.velocities[self.anchor_mask] = 0.0

    def _spring_forces(self) -> np.ndarray:
        """
        Hooke's law over every spring at once.

            stretch = |p_j - p_i| - rest_length
            F       = stiffness * stretch * unit_vector(p_j - p_i)

        applied as +F to i and -F to j.
        """
        n = len(self.positions)
        forces = np.zeros_like(self.positions)
        if self.spring_i.size == 0:
            return forces

        delta = self.positions[self.spring_j] - self.positions[self.spring_i]
        dist = np.linalg.norm(delta, axis=1)

        usable = dist > 1e-6
        if not usable.any():
            return forces

        direction = np.zeros_like(delta)
        direction[usable] = delta[usable] / dist[usable, None]

        stretch = dist - self.spring_rest
        force = (self.stiffness * stretch)[:, None] * direction
        force[~usable] = 0.0

        for axis in (0, 1):
            forces[:, axis] = (
                np.bincount(self.spring_i, weights=force[:, axis], minlength=n)
                - np.bincount(self.spring_j, weights=force[:, axis], minlength=n)
            )
        return forces

    def step(self, dt: float, substeps: int = 4) -> None:
        """
        Advance the simulation by dt seconds, using several smaller substeps
        for stability. Caller must call set_anchor_positions() with the
        correct target BEFORE calling step() for a given frame.

        Semi-implicit (symplectic) Euler: velocity is updated first, then the
        NEW velocity moves the position. More stable than explicit Euler for
        spring systems.
        """
        sub_dt = dt / substeps
        for _ in range(substeps):
            forces = self._spring_forces()
            forces[:, 1] += self.gravity  # gravity pulls +y (down, in image coords)

            self.velocities += forces * sub_dt
            self.velocities *= max(0.0, 1.0 - self.damping * sub_dt)
            self.positions += self.velocities * sub_dt

            self.velocities[self.anchor_mask] = 0.0


def build_knn_springs(positions: np.ndarray, k: int = 4) -> list[tuple[int, int, float]]:
    """
    Connect each point to its k nearest neighbors with a spring, using each
    pair's initial distance as that spring's rest length. Deduplicates
    symmetric pairs. Used instead of assuming a regular grid, since seed
    points are filtered to fall inside the (irregular) garment mask.
    """
    n = len(positions)
    springs = set()
    for i in range(n):
        dists = np.linalg.norm(positions - positions[i], axis=1)
        dists[i] = np.inf
        nearest = np.argsort(dists)[:k]
        for j in nearest:
            pair = (min(i, int(j)), max(i, int(j)))
            springs.add(pair)

    return [
        (i, j, float(np.linalg.norm(positions[i] - positions[j])))
        for i, j in springs
    ]