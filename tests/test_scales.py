import numpy as np
import pytest

from air_strings.scales import (
    SCALES,
    PitchMapping,
    midi_to_frequency,
    next_scale,
    note_name,
)


@pytest.mark.parametrize("scale", SCALES, ids=lambda s: s.name)
def test_shorter_string_never_lowers_the_pitch(scale):
    mapping = PitchMapping()
    fractions = np.linspace(0.9, 0.02, 400)  # long to short
    notes = [mapping.note_for_length(f, scale) for f in fractions]
    assert all(b >= a for a, b in zip(notes, notes[1:]))
    assert notes[0] == 48 and notes[-1] == 72  # clamped to C3..C5


@pytest.mark.parametrize("scale", SCALES, ids=lambda s: s.name)
def test_quantized_note_is_always_in_the_scale(scale):
    mapping = PitchMapping()
    allowed = set(scale.notes())
    for fraction in np.linspace(0.01, 1.0, 300):
        assert mapping.note_for_length(fraction, scale) in allowed


def test_pentatonic_notes_cover_two_octaves_of_c():
    assert SCALES[0].notes() == [48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72]


def test_length_maps_to_pitch_like_a_real_string():
    mapping = PitchMapping(longest_fraction=0.6)
    assert mapping.continuous_midi(0.6) == pytest.approx(48.0)
    assert mapping.continuous_midi(0.3) == pytest.approx(60.0)  # half the length, one octave up
    assert mapping.continuous_midi(0.15) == pytest.approx(72.0)


@pytest.mark.parametrize("scale", SCALES, ids=lambda s: s.name)
def test_a_string_exactly_as_long_as_a_scale_note_plays_that_note(scale):
    mapping = PitchMapping()
    for midi in scale.notes():
        fraction = mapping.longest_fraction * 2.0 ** (-(midi - mapping.lowest_midi) / 12.0)
        assert mapping.note_for_length(fraction, scale) == midi


def test_note_names():
    assert note_name(60) == "C4"
    assert note_name(64) == "E4"
    assert note_name(69) == "A4"
    assert note_name(48) == "C3"
    assert note_name(61) == "C#4"
    assert note_name(72) == "C5"


def test_midi_to_frequency():
    assert midi_to_frequency(69) == pytest.approx(440.0)
    assert midi_to_frequency(57) == pytest.approx(220.0)


def test_scale_cycle_wraps_around():
    assert [s.name for s in (SCALES[0], next_scale(SCALES[0]), next_scale(next_scale(SCALES[0])))] == [
        "C major pentatonic",
        "C major",
        "chromatic",
    ]
    assert next_scale(SCALES[-1]) == SCALES[0]
