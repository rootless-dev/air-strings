"""Musical scales, note naming and the string-length to pitch mapping."""

from __future__ import annotations

import math
from dataclasses import dataclass

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

LOWEST_MIDI = 48  # C3
RANGE_SEMITONES = 24  # two octaves, up to C5


@dataclass(frozen=True)
class Scale:
    name: str
    intervals: tuple[int, ...]
    short_name: str = ""  # for tight spaces such as a button; defaults to `name`

    @property
    def label(self) -> str:
        return self.short_name or self.name

    def notes(self, lowest: int = LOWEST_MIDI, span: int = RANGE_SEMITONES) -> list[int]:
        """All MIDI notes of this scale from `lowest` to `lowest + span`, inclusive."""
        root = lowest % 12
        return [
            m
            for m in range(lowest, lowest + span + 1)
            if (m - root) % 12 in self.intervals
        ]


SCALES = (
    Scale("C major pentatonic", (0, 2, 4, 7, 9), "Pentatonic"),
    Scale("C major", (0, 2, 4, 5, 7, 9, 11), "Major"),
    Scale("chromatic", tuple(range(12)), "Chromatic"),
)


def next_scale(current: Scale) -> Scale:
    index = SCALES.index(current)
    return SCALES[(index + 1) % len(SCALES)]


def note_name(midi: int) -> str:
    """Scientific pitch name, e.g. 60 -> 'C4', 64 -> 'E4'."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_frequency(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


@dataclass(frozen=True)
class PitchMapping:
    """Maps string length (as a fraction of the frame diagonal) to pitch.

    Like a real string, frequency is inversely proportional to length: a string
    `longest_fraction` long plays the lowest note, one four times shorter plays
    two octaves higher. Lengths outside that range are clamped.
    """

    longest_fraction: float = 0.65
    lowest_midi: int = LOWEST_MIDI
    span: int = RANGE_SEMITONES

    def continuous_midi(self, length_fraction: float) -> float:
        fraction = max(length_fraction, 1e-6)
        semitones = 12.0 * math.log2(self.longest_fraction / fraction)
        semitones = min(float(self.span), max(0.0, semitones))
        return self.lowest_midi + semitones

    def quantize(self, continuous: float, scale: Scale) -> int:
        """Nearest note of `scale` within the mapped range (ties go to the lower note)."""
        candidates = scale.notes(self.lowest_midi, self.span)
        return min(candidates, key=lambda m: (abs(m - continuous), m))

    def note_for_length(self, length_fraction: float, scale: Scale) -> int:
        return self.quantize(self.continuous_midi(length_fraction), scale)
