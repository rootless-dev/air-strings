"""The look of a string, as pure geometry: ghost strands, wobble, the pulled curve
and the color gradient along it.

Offsets are in pixels along the string's unit normal, sampled at positions xs
(0..1 from the string's start); render.py turns them into pixels and colors.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from air_strings.geometry import Point, unit_normal
from air_strings.strings import AirString, visual_frequency

# A vibrating string is drawn as a few strands at different phases of its
# oscillation, like a long exposure: the string now (brightest), where it was half
# a period ago (its mirror image), and its two extremes, crest and trough, which
# outline the vibration's envelope (its "eye") the way a string lingering at its
# turning points does on camera. Each extreme wobbles on its own clock.
MIRROR_LAG = 0.5  # in periods of the fundamental
GHOST_WEIGHTS = (1.0, 0.5, 0.4, 0.4)  # now, mirror, crest, trough
EXTREME_WOBBLE_DELAYS_S = (0.37, 0.71)  # crest, trough
# Below this amplitude (px) the ghosts fade out, while the strands merge into the rest line.
GHOST_FADE_PX = 10.0

# Wobble, relative to the vibration amplitude: standing overtones beyond the
# pluck's own harmonics, plus ripples running along the string. Both stay slow
# (about 3 Hz at most) so they read as motion at 30 fps, not as flicker.
WOBBLE_OVERTONES = (3, 4, 5, 6, 7, 8, 9)
WOBBLE_OVERTONE_RATIO = 0.12
WOBBLE_OVERTONE_HZ = (0.8, 2.4)  # for a string vibrating visually at WOBBLE_REFERENCE_RATE
WOBBLE_REFERENCE_RATE = 3.0  # Hz; higher strings wobble a little faster (square root)
WOBBLE_RIPPLES = 4
WOBBLE_RIPPLE_RATIO = 0.09
WOBBLE_RIPPLE_CYCLES = (1.5, 6.5)  # waves per string length
WOBBLE_RIPPLE_SPEED = (0.12, 0.4)  # string lengths per second

# A string held on a fingertip: a smooth curve through the fingertip, with faint
# ghosts trailing behind it (smaller, a little older).
HELD_TRIANGLE_SHARE = 0.35  # the rest of the shape is a sine arch: rounded, not a V
HELD_GHOST_SCALES = (1.0, 0.8, 0.62)
HELD_GHOST_WEIGHTS = (1.0, 0.3, 0.18)
HELD_GHOST_LAG_S = 0.07
HELD_WOBBLE_RATIO = 0.6  # wobble amplitude, as a fraction of the pull depth

# Colors along the string (BGR), from its left end on screen to its right end:
# lime / yellow-green to white-cyan. The glow uses the saturated version.
CORE_LEFT = (80, 255, 165)
CORE_RIGHT = (255, 255, 200)
GLOW_LEFT = (30, 255, 120)
GLOW_RIGHT = (255, 230, 50)


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


@lru_cache(maxsize=64)
def _wobble_params(seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(1_000_003 + seed)
    n = np.asarray(WOBBLE_OVERTONES, dtype=np.float64)
    amplitudes = rng.uniform(0.4, 1.0, n.size) * rng.choice((-1.0, 1.0), n.size) / n**0.6
    amplitudes /= np.abs(amplitudes).sum()
    overtone_hz = rng.uniform(*WOBBLE_OVERTONE_HZ, n.size)
    overtone_phases = rng.uniform(0.0, 2.0 * math.pi, n.size)
    cycles = rng.uniform(*WOBBLE_RIPPLE_CYCLES, WOBBLE_RIPPLES)
    speeds = rng.uniform(*WOBBLE_RIPPLE_SPEED, WOBBLE_RIPPLES) * rng.choice((-1.0, 1.0), WOBBLE_RIPPLES)
    ripple_phases = rng.uniform(0.0, 2.0 * math.pi, WOBBLE_RIPPLES)
    weights = 1.0 / (1.0 + np.arange(WOBBLE_RIPPLES))
    weights /= weights.sum()
    return n, amplitudes, overtone_hz, overtone_phases, cycles, speeds, ripple_phases, weights


def wobble(
    xs: np.ndarray, t: float, amplitude: float, seed: int, rate: float = WOBBLE_REFERENCE_RATE
) -> np.ndarray:
    """Irregular offsets (px) at xs: overtones and travelling ripples, zero at both ends.

    Deterministic for a given `seed` and time `t` (seconds), so the same string
    wobbles the same way frame after frame; proportional to `amplitude`.
    `rate` is the string's visual frequency (Hz), which nudges the overtones' pace.
    """
    if amplitude == 0.0:
        return np.zeros_like(xs)
    n, amplitudes, overtone_hz, overtone_phases, cycles, speeds, ripple_phases, weights = _wobble_params(seed)
    pace = math.sqrt(rate / WOBBLE_REFERENCE_RATE)
    turns = np.cos(2.0 * math.pi * overtone_hz * pace * t + overtone_phases)
    overtones = (amplitudes * turns) @ np.sin(np.pi * np.outer(n, xs))
    waves = np.sin(2.0 * math.pi * (np.outer(xs, cycles) - speeds * t) + ripple_phases)  # (points, ripples)
    ripples = (waves @ weights) * np.sin(np.pi * xs)
    return amplitude * (WOBBLE_OVERTONE_RATIO * overtones + WOBBLE_RIPPLE_RATIO * ripples)


def ghost_fade(amplitude: float) -> float:
    """0..1 brightness factor of the ghost strands for a vibration this wide (px)."""
    return smoothstep(amplitude / GHOST_FADE_PX)


def vibration_strands(string: AirString, xs: np.ndarray, now: float) -> tuple[np.ndarray, np.ndarray]:
    """Offsets (strands, len(xs)) of a vibrating string's strands, and their weights.

    Strands, in order: now, half a period ago, crest, trough. Past shapes are
    scaled to the current amplitude, so the strands stay a symmetric "eye" even
    though the (slowed down) vibration decays noticeably within one period; the
    string's shape before the pluck is the released shape, the crest.
    """
    offsets = np.zeros((len(GHOST_WEIGHTS), xs.size))
    weights = np.asarray(GHOST_WEIGHTS, dtype=np.float64).copy()
    plucked_at = string.last_pluck_at
    released = string.amplitude(plucked_at) if math.isfinite(plucked_at) else 0.0
    if released <= 0.0:
        return offsets, weights
    rate = visual_frequency(string.frequency)
    amplitude = string.amplitude(now)
    elapsed = now - plucked_at
    seed = string.string_id

    def scaled_shape(t: float) -> np.ndarray:
        then = string.amplitude(t)
        return string.displacement(xs, t) * (amplitude / then) if then > 0.0 else np.zeros_like(xs)

    offsets[0] = string.displacement(xs, now) + wobble(xs, elapsed, amplitude, seed, rate)
    mirror_at = max(now - MIRROR_LAG / rate, plucked_at)
    offsets[1] = scaled_shape(mirror_at) + wobble(xs, mirror_at - plucked_at, amplitude, seed, rate)
    crest = scaled_shape(plucked_at)
    for index, (sign, delay) in enumerate(zip((1.0, -1.0), EXTREME_WOBBLE_DELAYS_S), start=2):
        offsets[index] = sign * crest + wobble(xs, elapsed - delay, amplitude, seed, rate)
    weights[1:] *= ghost_fade(amplitude)
    return offsets, weights


def bend_shape(xs: np.ndarray, position: float) -> np.ndarray:
    """0 at both ends, 1 at `position`: a rounded arch with a slight point, not a straight V."""
    u = min(0.98, max(0.02, position))
    ratio = np.where(xs <= u, xs / u, (1.0 - xs) / (1.0 - u))
    return HELD_TRIANGLE_SHARE * ratio + (1.0 - HELD_TRIANGLE_SHARE) * np.sin(0.5 * math.pi * ratio)


def pull_depth(start: Point, end: Point, contact: Point) -> float:
    """Signed distance (px) of the fingertip from the rest line, along the string normal."""
    nx, ny = unit_normal(start, end)
    return (contact[0] - start[0]) * nx + (contact[1] - start[1]) * ny


def held_strands(
    string: AirString, position: float, contact: Point, xs: np.ndarray, now: float
) -> tuple[np.ndarray, np.ndarray]:
    """Offsets (strands, len(xs)) of a string held on a fingertip, and their weights.

    Strand 0 passes exactly through the fingertip (its ripples vanish there); the
    faint ghosts are shallower and a little older.
    """
    depth = pull_depth(string.start, string.end, contact)
    shape = bend_shape(xs, position)
    free = 1.0 - shape  # where the string is free to ripple: everywhere but the fingertip
    rate = visual_frequency(string.frequency)
    offsets = np.empty((len(HELD_GHOST_SCALES), xs.size))
    for index, scale in enumerate(HELD_GHOST_SCALES):
        ripple = wobble(xs, now - index * HELD_GHOST_LAG_S, abs(depth) * HELD_WOBBLE_RATIO, string.string_id, rate)
        offsets[index] = scale * depth * shape + free * ripple
    weights = np.asarray(HELD_GHOST_WEIGHTS, dtype=np.float64).copy()
    weights[1:] *= ghost_fade(abs(depth))
    return offsets, weights


def gradient_positions(start: Point, end: Point, xs: np.ndarray) -> np.ndarray:
    """Where positions xs (from the string's start) sit between its left (0) and right (1) end on screen."""
    return xs if start[0] <= end[0] else 1.0 - xs


def gradient_colors(positions: np.ndarray, left: tuple, right: tuple) -> np.ndarray:
    """(len(positions), 3) BGR colors blended from `left` (0) to `right` (1)."""
    p = np.asarray(positions, dtype=np.float64)[:, None]
    return (1.0 - p) * np.asarray(left, dtype=np.float64) + p * np.asarray(right, dtype=np.float64)
