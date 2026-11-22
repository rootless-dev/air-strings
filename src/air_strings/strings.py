"""Air strings: the string model, its vibration and the string collection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from air_strings.geometry import Point, crossing_position, distance
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
    pitch: float  # continuous MIDI pitch, set at commit time and while resizing
    midi: int
    color: tuple[int, int, int]
    created_at: float  # commit time; also reset when a resize ends
    resizing: bool = False  # an endpoint is being dragged; the post-commit grace does not apply
    last_pluck_at: float = -math.inf
    pluck_position: float = 0.5
    pluck_amplitude: float = 0.0  # signed: initial displacement, in pixels along the normal
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
        graced = self.resizing or now - self.created_at >= COMMIT_GRACE_S
        return graced and now - self.last_pluck_at >= PLUCK_COOLDOWN_S

    def pluck(self, now: float, position: float, velocity: float, displacement: float | None = None) -> None:
        """Start the visual vibration from a triangular shape peaked at `position` (0..1).

        `displacement` is the signed peak offset in pixels along the string normal
        (the bent shape at release); without it the peak grows with `velocity`.
        """
        self.last_pluck_at = now
        self.pluck_position = position
        if displacement is None:
            displacement = min(MAX_VISUAL_AMPLITUDE_PX, 0.12 * self.length) * (0.3 + 0.7 * velocity)
        self.pluck_amplitude = displacement
        self._modes = mode_amplitudes(position)

    def _decay(self, now: float) -> float:
        """Signed peak displacement now, in pixels (decays like the sound)."""
        elapsed = now - self.last_pluck_at
        if elapsed < 0 or not math.isfinite(elapsed):
            return 0.0
        tau = note_t60(self.frequency) / math.log(1000.0)
        return self.pluck_amplitude * math.exp(-elapsed / tau)

    def amplitude(self, now: float) -> float:
        """Current vibration amplitude in pixels."""
        return abs(self._decay(now))

    def is_vibrating(self, now: float) -> bool:
        return self.amplitude(now) > VISIBLE_THRESHOLD_PX

    def glow_level(self, now: float) -> float:
        """0..1 glow intensity that fades out as the vibration dies down."""
        return min(1.0, self.amplitude(now) / 8.0)

    def displacement(self, xs: np.ndarray, now: float) -> np.ndarray:
        """Offset in pixels, along the string normal, at positions xs (0..1)."""
        peak = self._decay(now)
        if abs(peak) <= VISIBLE_THRESHOLD_PX:
            return np.zeros_like(xs)
        elapsed = now - self.last_pluck_at
        n = np.arange(1, VISUAL_MODES + 1, dtype=np.float64)
        phases = np.cos(2.0 * math.pi * n * visual_frequency(self.frequency) * elapsed)
        shapes = np.sin(np.pi * np.outer(n, xs))  # (modes, points)
        return peak * ((self._modes * phases) @ shapes)


@dataclass(frozen=True)
class Crossing:
    string: AirString
    position: float  # where the movement crossed the string, 0..1 from its start


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

    @property
    def mapping(self) -> PitchMapping:
        return self._mapping

    def __contains__(self, string: AirString) -> bool:
        return any(s is string for s in self._strings)

    def note_for_length(self, length: float, frame_diagonal: float) -> int:
        """Note of the active scale that a string `length` pixels long plays."""
        return self._mapping.note_for_length(length / frame_diagonal, self._scale)

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

    def move_endpoint(self, string: AirString, endpoint: int, point: Point, frame_diagonal: float) -> None:
        """Move the start (endpoint 0) or end (1) of `string` and re-tune it to its new length."""
        if endpoint == 0:
            string.start = point
        else:
            string.end = point
        string.pitch = self._mapping.continuous_midi(string.length / frame_diagonal)
        string.midi = self._mapping.quantize(string.pitch, self._scale)

    def begin_resize(self, string: AirString) -> None:
        """A resize has started: the string can be plucked (by another hand) while it lasts."""
        string.resizing = True

    def end_resize(self, string: AirString, now: float) -> None:
        """A resize has ended: the string gets the same pluck grace period as a new one."""
        string.resizing = False
        string.created_at = now

    def clear(self) -> None:
        self._strings.clear()

    def set_scale(self, scale: Scale) -> None:
        self._scale = scale
        for string in self._strings:
            string.midi = self._mapping.quantize(string.pitch, scale)

    def find_crossings(self, move_start: Point, move_end: Point, now: float) -> list[Crossing]:
        """All pluckable strings crossed by a fingertip moving from move_start to move_end."""
        crossings = []
        for string in self._strings:
            if not string.can_pluck(now):
                continue
            position = crossing_position(move_start, move_end, string.start, string.end)
            if position is not None:
                crossings.append(Crossing(string, position))
        return crossings
