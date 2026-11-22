"""Air strings: the string model, its vibration, the string collection and the draw gesture."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from air_strings.geometry import Point, crossing_position, distance, unit_normal
from air_strings.scales import PitchMapping, Scale, midi_to_frequency, note_name
from air_strings.synth import note_t60

MAX_STRINGS = 6
MIN_STRING_LENGTH_PX = 80.0
PLUCK_COOLDOWN_S = 0.1
COMMIT_GRACE_S = 0.3
VISUAL_MODES = 6
MAX_VISUAL_AMPLITUDE_PX = 45.0
VISIBLE_THRESHOLD_PX = 0.5

# BGR colors for vibrating strings: greens and cyans.
STRING_COLORS = (
    (80, 255, 60),
    (255, 230, 40),
    (160, 255, 0),
    (230, 255, 120),
    (40, 255, 160),
    (255, 200, 90),
)


def mode_amplitudes(position: float, modes: int = VISUAL_MODES) -> np.ndarray:
    """Fourier sine coefficients of a unit triangular pluck at `position` (0..1).

    b_n = 2 sin(n pi p) / (n^2 pi^2 p (1 - p)); summing b_n sin(n pi x) rebuilds
    the triangle. A pluck in the middle has no even harmonics.
    """
    p = min(0.98, max(0.02, position))
    n = np.arange(1, modes + 1, dtype=np.float64)
    return 2.0 * np.sin(n * math.pi * p) / (n**2 * math.pi**2 * p * (1.0 - p))


def visual_frequency(frequency: float) -> float:
    """A slowed-down, visible vibration rate (a few Hz) that still rises with pitch."""
    return 2.5 * (frequency / 130.81) ** 0.5


@dataclass(eq=False)
class AirString:
    string_id: int
    start: Point
    end: Point
    pitch: float  # continuous MIDI pitch, fixed at commit time
    midi: int
    color: tuple[int, int, int]
    created_at: float
    last_pluck_at: float = -math.inf
    pluck_position: float = 0.5
    pluck_amplitude: float = 0.0
    _modes: np.ndarray = field(default_factory=lambda: np.zeros(VISUAL_MODES))

    @property
    def length(self) -> float:
        return distance(self.start, self.end)

    @property
    def frequency(self) -> float:
        return midi_to_frequency(self.midi)

    @property
    def note_name(self) -> str:
        return note_name(self.midi)

    def can_pluck(self, now: float) -> bool:
        return now - self.created_at >= COMMIT_GRACE_S and now - self.last_pluck_at >= PLUCK_COOLDOWN_S

    def pluck(self, now: float, position: float, velocity: float) -> None:
        self.last_pluck_at = now
        self.pluck_position = position
        self.pluck_amplitude = min(MAX_VISUAL_AMPLITUDE_PX, 0.12 * self.length) * (0.3 + 0.7 * velocity)
        self._modes = mode_amplitudes(position)

    def _envelope(self, now: float) -> float:
        """Current vibration amplitude in pixels (decays like the sound)."""
        elapsed = now - self.last_pluck_at
        if elapsed < 0 or not math.isfinite(elapsed):
            return 0.0
        tau = note_t60(self.frequency) / math.log(1000.0)
        return self.pluck_amplitude * math.exp(-elapsed / tau)

    def is_vibrating(self, now: float) -> bool:
        return self._envelope(now) > VISIBLE_THRESHOLD_PX

    def glow_level(self, now: float) -> float:
        """0..1 glow intensity that fades out as the vibration dies down."""
        return min(1.0, self._envelope(now) / 8.0)

    def displacement(self, xs: np.ndarray, now: float) -> np.ndarray:
        """Offset in pixels, along the string normal, at positions xs (0..1)."""
        envelope = self._envelope(now)
        if envelope <= VISIBLE_THRESHOLD_PX:
            return np.zeros_like(xs)
        elapsed = now - self.last_pluck_at
        n = np.arange(1, VISUAL_MODES + 1, dtype=np.float64)
        phases = np.cos(2.0 * math.pi * n * visual_frequency(self.frequency) * elapsed)
        shapes = np.sin(np.pi * np.outer(n, xs))  # (modes, points)
        return envelope * ((self._modes * phases) @ shapes)

    def points(self, xs: np.ndarray, now: float) -> np.ndarray:
        """Pixel coordinates (len(xs), 2) of the string at positions xs, including vibration."""
        start = np.asarray(self.start, dtype=np.float64)
        end = np.asarray(self.end, dtype=np.float64)
        normal = np.asarray(unit_normal(self.start, self.end), dtype=np.float64)
        base = start + np.outer(xs, end - start)
        return base + np.outer(self.displacement(xs, now), normal)


@dataclass(frozen=True)
class Pluck:
    string: AirString
    position: float


class StringSet:
    """The committed strings (at most MAX_STRINGS; the oldest is dropped)."""

    def __init__(self, mapping: PitchMapping, scale: Scale, max_strings: int = MAX_STRINGS) -> None:
        self._mapping = mapping
        self._scale = scale
        self._max_strings = max_strings
        self._strings: list[AirString] = []
        self._next_id = 0

    @property
    def strings(self) -> list[AirString]:
        return list(self._strings)

    @property
    def scale(self) -> Scale:
        return self._scale

    def add(self, start: Point, end: Point, frame_diagonal: float, now: float) -> AirString:
        pitch = self._mapping.continuous_midi(distance(start, end) / frame_diagonal)
        string = AirString(
            string_id=self._next_id,
            start=start,
            end=end,
            pitch=pitch,
            midi=self._mapping.quantize(pitch, self._scale),
            color=STRING_COLORS[self._next_id % len(STRING_COLORS)],
            created_at=now,
        )
        self._next_id += 1
        self._strings.append(string)
        if len(self._strings) > self._max_strings:
            self._strings.pop(0)
        return string

    def clear(self) -> None:
        self._strings.clear()

    def set_scale(self, scale: Scale) -> None:
        self._scale = scale
        for string in self._strings:
            string.midi = self._mapping.quantize(string.pitch, scale)

    def find_plucks(self, move_start: Point, move_end: Point, now: float) -> list[Pluck]:
        """All strings crossed by a fingertip moving from move_start to move_end."""
        plucks = []
        for string in self._strings:
            if not string.can_pluck(now):
                continue
            position = crossing_position(move_start, move_end, string.start, string.end)
            if position is not None:
                plucks.append(Pluck(string, position))
        return plucks


class DrawGesture:
    """Two pinching hands stretch a pending string; releasing a pinch commits it."""

    def __init__(self, min_length: float = MIN_STRING_LENGTH_PX, release_frames: int = 2) -> None:
        self._min_length = min_length
        self._release_frames = release_frames
        self._missing = 0
        self.pending: tuple[Point, Point] | None = None

    def update(self, pinch_points: list[Point]) -> tuple[Point, Point] | None:
        """Feed the pinch midpoints of the currently pinching hands.

        Returns the segment to commit once a two-hand pinch has ended (for
        `release_frames` consecutive frames, which rides over one-frame
        tracking dropouts) and the string is long enough, otherwise None.
        """
        if len(pinch_points) >= 2:
            a, b = sorted(pinch_points[:2])
            self.pending = (a, b)
            self._missing = 0
            return None
        if self.pending is None:
            return None
        self._missing += 1
        if self._missing < self._release_frames:
            return None
        finished, self.pending = self.pending, None
        self._missing = 0
        if distance(*finished) >= self._min_length:
            return finished
        return None
