"""Spark particles thrown off a plucked string: a small vectorized numpy particle system."""

from __future__ import annotations

import math

import numpy as np

MAX_PARTICLES = 400
MIN_SPARKS = 8
MAX_SPARKS = 40
GRAVITY_PX_S2 = 380.0
DRAG_PER_S = 2.8  # velocity decays as exp(-DRAG_PER_S * t)
LIFETIME_S = (0.5, 1.0)
SPEED_PX_S = (160.0, 620.0)
SPREAD_RAD = math.radians(65.0)  # max angle away from the string normal
STREAK_S = 0.045  # a streak is as long as the distance covered in this time


class ParticleSystem:
    """Fixed-capacity particles kept in arrays, oldest first.

    When an emission would exceed the capacity, the oldest particles make room.
    """

    def __init__(self, capacity: int = MAX_PARTICLES, rng: np.random.Generator | None = None) -> None:
        self._capacity = capacity
        self._rng = rng if rng is not None else np.random.default_rng()
        self._position = np.zeros((capacity, 2))
        self._velocity = np.zeros((capacity, 2))
        self._age = np.zeros(capacity)
        self._lifetime = np.ones(capacity)
        self._tint = np.zeros((capacity, 3))
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def capacity(self) -> int:
        return self._capacity

    @staticmethod
    def spark_count(strength: float) -> int:
        strength = min(1.0, max(0.0, strength))
        return int(round(MIN_SPARKS + (MAX_SPARKS - MIN_SPARKS) * strength))

    def emit(
        self,
        origin: tuple[float, float],
        tangent: tuple[float, float],
        normal: tuple[float, float],
        strength: float,
        tint: tuple[float, float, float],
    ) -> int:
        """Throw sparks from `origin` to both sides of a string; returns how many were spawned."""
        count = min(self.spark_count(strength), self._capacity)
        overflow = self._count + count - self._capacity
        if overflow > 0:
            self._drop_oldest(overflow)

        rng = self._rng
        side = np.where(rng.random(count) < 0.5, -1.0, 1.0)
        angle = rng.uniform(-SPREAD_RAD, SPREAD_RAD, count)
        speed = rng.uniform(*SPEED_PX_S, count) * (0.55 + 0.45 * min(1.0, max(0.0, strength)))
        n = np.asarray(normal, dtype=np.float64)
        t = np.asarray(tangent, dtype=np.float64)
        direction = np.outer(side * np.cos(angle), n) + np.outer(np.sin(angle), t)

        new = slice(self._count, self._count + count)
        self._position[new] = np.asarray(origin, dtype=np.float64) + rng.normal(0.0, 1.5, (count, 2))
        self._velocity[new] = direction * speed[:, None]
        self._age[new] = 0.0
        self._lifetime[new] = rng.uniform(*LIFETIME_S, count)
        self._tint[new] = tint
        self._count += count
        return count

    def update(self, dt: float) -> None:
        """Advance by `dt` seconds: drag, gravity, motion, ageing; dead particles are removed."""
        if self._count == 0 or dt <= 0:
            return
        live = slice(0, self._count)
        velocity = self._velocity[live]
        velocity *= math.exp(-DRAG_PER_S * dt)
        velocity[:, 1] += GRAVITY_PX_S2 * dt
        self._position[live] += velocity * dt
        self._age[live] += dt
        alive = self._age[live] < self._lifetime[live]
        if not alive.all():
            kept = int(alive.sum())
            for array in (self._position, self._velocity, self._age, self._lifetime, self._tint):
                array[:kept] = array[live][alive]
            self._count = kept

    def clear(self) -> None:
        self._count = 0

    def streaks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Line segments to draw: (heads (n, 2), tails (n, 2), colors (n, 3) already faded)."""
        live = slice(0, self._count)
        heads = self._position[live]
        tails = heads - self._velocity[live] * STREAK_S
        fade = 1.0 - self._age[live] / self._lifetime[live]
        colors = self._tint[live] * fade[:, None]
        return heads, tails, colors

    def _drop_oldest(self, count: int) -> None:
        kept = self._count - count
        for array in (self._position, self._velocity, self._age, self._lifetime, self._tint):
            array[:kept] = array[count : self._count]
        self._count = kept
