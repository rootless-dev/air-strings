"""Ruler marks shown while a string is stretched: where each note's string length falls."""

from __future__ import annotations

from dataclasses import dataclass

from air_strings.scales import PitchMapping, Scale


@dataclass(frozen=True)
class RulerMark:
    distance: float  # pixels from the ruler's anchor end, along the string
    midi: int | None  # note played by a string this long; None for a half-step mark
    in_scale: bool  # note of the active scale (a tall tick)


def ruler_marks(mapping: PitchMapping, scale: Scale, max_distance: float, frame_diagonal: float) -> list[RulerMark]:
    """Marks for every semitone of the mapped range, plus one halfway between neighbours.

    A semitone mark sits at the length that plays exactly that note, so a string
    stretched to a mark plays its note. Half-step marks (between two semitones,
    in pitch) only give the ruler its rhythm. Marks farther than `max_distance`
    from the anchor are left out. Sorted by distance, nearest first.
    """
    in_scale = set(scale.notes(mapping.lowest_midi, mapping.span))
    marks = []
    top = mapping.lowest_midi + mapping.span
    for midi in range(top, mapping.lowest_midi - 1, -1):  # shortest (highest) first
        distance = mapping.length_fraction(midi) * frame_diagonal
        if distance <= max_distance:
            marks.append(RulerMark(distance, midi, midi in in_scale))
        if midi > mapping.lowest_midi:
            half = mapping.length_fraction(midi - 0.5) * frame_diagonal
            if half <= max_distance:
                marks.append(RulerMark(half, None, False))
    return marks
