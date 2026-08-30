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
accuracy.
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

    def set_anchor_positions(self, anchor_positions: np.ndarray) -> None:
        """Directly set the position of anchor points (kinematic, not simulated)."""
        self.positions[self.anchor_mask] = anchor_positions
        self.velocities[self.anchor_mask] = 0.0

    def _spring_forces(self) -> np.ndarray:
        forces = np.zeros_like(self.positions)
        for i, j, rest_length in self.springs:
            delta = self.positions[j] - self.positions[i]
            dist = np.linalg.norm(delta)
            if dist < 1e-6:
                continue
            direction = delta / dist
            stretch = dist - rest_length
            force = self.stiffness * stretch * direction
            forces[i] += force
            forces[j] -= force
        return forces

    def step(self, dt: float, substeps: int = 4) -> None:
        """
        Advance the simulation by dt seconds, using several smaller substeps
        for stability. Caller must call set_anchor_positions() with the
        correct target BEFORE calling step() for a given frame.
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
