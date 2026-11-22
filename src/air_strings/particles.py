"""Sparks thrown off a plucked string: a small vectorized numpy particle system.

Sparks spawn along the whole string (denser around the pluck point), burst out to
both sides, then slow down and float slowly upwards while they fade. Each one is
drawn as a short, mostly vertical streak.
"""

from __future__ import annotations

import math

import numpy as np

MAX_PARTICLES = 500
MIN_SPARKS = 12
MAX_SPARKS = 75
NEAR_PLUCK_SHARE = 0.5  # share of the sparks clustered around the pluck point; the rest spread evenly
NEAR_PLUCK_SIGMA = 0.12  # spread of that cluster, as a fraction of the string length
BURST_SPEED_PX_S = (40.0, 220.0)
BURST_SPREAD_RAD = math.radians(55.0)  # max angle away from the string normal
DRAG_PER_S = 3.5  # velocity decays as exp(-DRAG_PER_S * t)
LIFT_PX_S2 = 45.0  # a slow upward drift, so sparks float instead of falling
LIFETIME_S = (0.9, 1.6)
STREAK_LENGTH_PX = (2.5, 10.0)
STREAK_TILT_RAD = math.radians(28.0)  # streaks are vertical, give or take this much


class ParticleSystem:
    """Fixed-capacity particles kept in arrays, oldest first.

    When an emission would exceed the capacity, the oldest particles make room.
    """

    def __init__(self, capacity: int = MAX_PARTICLES, rng: np.random.Generator | None = None) -> None:
        self._capacity = capacity
        self._rng = rng if rng is not None else np.random.default_rng()
        self._position = np.zeros((capacity, 2))
        self._velocity = np.zeros((capacity, 2))
        self._streak = np.zeros((capacity, 2))
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

    @property
    def positions(self) -> np.ndarray:
        """(count, 2) centers of the live sparks."""
        return self._position[: self._count]

    @staticmethod
    def spark_count(strength: float) -> int:
        strength = min(1.0, max(0.0, strength))
        return int(round(MIN_SPARKS + (MAX_SPARKS - MIN_SPARKS) * strength))

    def emit(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        position: float,
        strength: float,
        tint: tuple[float, float, float],
        displacement: float = 0.0,
    ) -> int:
        """Throw sparks off the string from `start` to `end`, plucked at `position` (0..1).

        `displacement` is the pluck's bend in pixels: sparks start anywhere on the
        vibrating string, which reaches that far from the rest line at the pluck
        point and less towards the ends. Returns how many were spawned.
        """
        count = min(self.spark_count(strength), self._capacity)
        overflow = self._count + count - self._capacity
        if overflow > 0:
            self._drop_oldest(overflow)

        rng = self._rng
        s = np.asarray(start, dtype=np.float64)
        span = np.asarray(end, dtype=np.float64) - s
        length = math.hypot(span[0], span[1])
        tangent = span / length if length > 1e-9 else np.array([1.0, 0.0])
        normal = np.array([-tangent[1], tangent[0]])

        p = min(0.98, max(0.02, position))
        along = np.where(
            rng.random(count) < NEAR_PLUCK_SHARE,
            rng.normal(p, NEAR_PLUCK_SIGMA, count),
            rng.uniform(0.0, 1.0, count),
        )
        along = np.clip(along, 0.02, 0.98)
        side = np.where(rng.random(count) < 0.5, -1.0, 1.0)
        reach = abs(displacement) * np.minimum(along / p, (1.0 - along) / (1.0 - p))
        lateral = side * rng.uniform(0.0, 1.0, count) * reach + rng.normal(0.0, 1.5, count)

        angle = rng.uniform(-BURST_SPREAD_RAD, BURST_SPREAD_RAD, count)
        speed = rng.uniform(*BURST_SPEED_PX_S, count) * (0.5 + 0.5 * min(1.0, max(0.0, strength)))
        direction = np.outer(side * np.cos(angle), normal) + np.outer(np.sin(angle), tangent)
        tilt = rng.uniform(-STREAK_TILT_RAD, STREAK_TILT_RAD, count)
        streak_length = rng.uniform(*STREAK_LENGTH_PX, count)

        new = slice(self._count, self._count + count)
        self._position[new] = s + np.outer(along, span) + np.outer(lateral, normal)
        self._velocity[new] = direction * speed[:, None]
        self._streak[new] = np.stack([np.sin(tilt), np.cos(tilt)], axis=1) * streak_length[:, None]
        self._age[new] = 0.0
        self._lifetime[new] = rng.uniform(*LIFETIME_S, count)
        self._tint[new] = tint
        self._count += count
        return count

    def update(self, dt: float) -> None:
        """Advance by `dt` seconds: drag, lift, motion, ageing; dead particles are removed."""
        if self._count == 0 or dt <= 0:
            return
        live = slice(0, self._count)
        velocity = self._velocity[live]
        velocity *= math.exp(-DRAG_PER_S * dt)
        velocity[:, 1] -= LIFT_PX_S2 * dt
        self._position[live] += velocity * dt
        self._age[live] += dt
        alive = self._age[live] < self._lifetime[live]
        if not alive.all():
            kept = int(alive.sum())
            for array in self._arrays():
                array[:kept] = array[live][alive]
            self._count = kept

    def clear(self) -> None:
        self._count = 0

    def streaks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Line segments to draw: (heads (n, 2), tails (n, 2), colors (n, 3) already faded).

        Sparks keep their full brightness for a while, then fade out.
        """
        live = slice(0, self._count)
        half = self._streak[live] * 0.5
        heads = self._position[live] - half
        tails = self._position[live] + half
        fade = 1.0 - (self._age[live] / self._lifetime[live]) ** 2
        colors = self._tint[live] * fade[:, None]
        return heads, tails, colors

    def _arrays(self) -> tuple[np.ndarray, ...]:
        return (self._position, self._velocity, self._streak, self._age, self._lifetime, self._tint)

    def _drop_oldest(self, count: int) -> None:
        kept = self._count - count
        for array in self._arrays():
            array[:kept] = array[count : self._count]
        self._count = kept
