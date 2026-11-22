import math

import pytest

from air_strings.ruler import ruler_marks
from air_strings.scales import SCALES, PitchMapping

DIAGONAL = math.hypot(1280, 720)
MAPPING = PitchMapping()


@pytest.mark.parametrize("scale", SCALES, ids=lambda s: s.name)
def test_a_string_stretched_to_a_note_mark_plays_that_note(scale):
    marks = ruler_marks(MAPPING, scale, 2 * DIAGONAL, DIAGONAL)
    for mark in marks:
        if mark.midi is not None:
            length = mark.distance
            assert MAPPING.note_for_length(length / DIAGONAL, SCALES[2]) == mark.midi
            if mark.in_scale:
                # Round trip: length -> note -> length lands back on the mark.
                note = MAPPING.note_for_length(length / DIAGONAL, scale)
                assert note == mark.midi
                assert MAPPING.length_fraction(note) * DIAGONAL == pytest.approx(length)


def test_tall_marks_are_exactly_the_scale_notes():
    pentatonic = SCALES[0]
    marks = ruler_marks(MAPPING, pentatonic, 2 * DIAGONAL, DIAGONAL)
    assert sorted(m.midi for m in marks if m.in_scale) == pentatonic.notes()
    semitones = [m for m in marks if m.midi is not None]
    assert sorted(m.midi for m in semitones) == list(range(48, 73))
    assert len([m for m in marks if m.midi is None]) == 24  # one half-step mark between neighbours


def test_marks_are_sorted_and_alternate_semitone_and_half_step():
    marks = ruler_marks(MAPPING, SCALES[2], 2 * DIAGONAL, DIAGONAL)
    distances = [m.distance for m in marks]
    assert distances == sorted(distances)
    kinds = [m.midi is None for m in marks]
    assert kinds == [i % 2 == 1 for i in range(len(marks))]
    # Shorter is higher, like a real string.
    notes = [m.midi for m in marks if m.midi is not None]
    assert notes == sorted(notes, reverse=True)


def test_marks_stop_at_the_requested_distance():
    limit = MAPPING.length_fraction(60) * DIAGONAL  # a string playing C4
    marks = ruler_marks(MAPPING, SCALES[0], limit + 1e-6, DIAGONAL)
    assert all(m.distance <= limit + 1e-6 for m in marks)
    assert min(m.midi for m in marks if m.midi is not None) == 60
    assert ruler_marks(MAPPING, SCALES[0], 100.0, DIAGONAL) == []  # shorter than C5's length
