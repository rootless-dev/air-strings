import math

import numpy as np
import pytest

from air_strings.scales import SCALES, PitchMapping
from air_strings.strings import (
    COMMIT_GRACE_S,
    MAX_STRINGS,
    PLUCK_COOLDOWN_S,
    DrawGesture,
    StringSet,
    mode_amplitudes,
)

DIAGONAL = math.hypot(1280, 720)


def test_middle_pluck_has_no_even_modes():
    amplitudes = mode_amplitudes(0.5, 8)
    assert np.all(np.abs(amplitudes[1::2]) < 1e-12)
    assert np.all(np.abs(amplitudes[0::2]) > 1e-3)


def test_mode_amplitudes_rebuild_the_triangular_pluck():
    position = 0.3
    xs = np.linspace(0, 1, 101)
    n = np.arange(1, 201)
    shape = mode_amplitudes(position, 200) @ np.sin(np.pi * np.outer(n, xs))
    expected = np.minimum(xs / position, (1 - xs) / (1 - position))
    assert np.max(np.abs(shape - expected)) < 0.01


def test_pluck_near_the_end_excites_more_high_modes():
    def high_ratio(position):
        a = np.abs(mode_amplitudes(position, 6))
        return a[3:].sum() / a[0]

    assert high_ratio(0.1) > high_ratio(0.45)


def make_set():
    return StringSet(PitchMapping(), SCALES[0])


def test_string_set_drops_the_oldest_beyond_six():
    strings = make_set()
    for i in range(MAX_STRINGS + 1):
        strings.add((100.0, 100.0 + 50 * i), (600.0, 100.0 + 50 * i), DIAGONAL, now=0.0)
    ids = [s.string_id for s in strings.strings]
    assert ids == list(range(1, MAX_STRINGS + 1))


def test_shorter_string_gets_higher_or_equal_note():
    strings = make_set()
    long = strings.add((100.0, 100.0), (1000.0, 100.0), DIAGONAL, 0.0)
    short = strings.add((100.0, 300.0), (400.0, 300.0), DIAGONAL, 0.0)
    assert short.midi > long.midi


def test_scale_change_requantizes_existing_strings():
    strings = make_set()
    string = strings.add((100.0, 100.0), (100.0 + 0.65 * DIAGONAL / 2 ** (6 / 12), 100.0), DIAGONAL, 0.0)
    assert string.pitch == pytest.approx(54.0)  # F#3: not in pentatonic, in chromatic
    assert string.midi != 54
    strings.set_scale(SCALES[2])
    assert string.midi == 54


def test_strum_crosses_several_strings_at_once():
    strings = make_set()
    for i in range(3):
        strings.add((100.0, 100.0 + 60 * i), (700.0, 100.0 + 60 * i), DIAGONAL, now=0.0)
    plucks = strings.find_plucks((300.0, 50.0), (320.0, 300.0), now=1.0)
    assert len(plucks) == 3
    assert all(0.3 < p.position < 0.4 for p in plucks)


def test_cooldown_and_commit_grace_block_double_triggers():
    strings = make_set()
    strings.add((100.0, 200.0), (700.0, 200.0), DIAGONAL, now=0.0)
    down, up = ((400.0, 150.0), (400.0, 250.0)), ((400.0, 250.0), (400.0, 150.0))
    assert strings.find_plucks(*down, now=COMMIT_GRACE_S / 2) == []
    t = 1.0
    [pluck] = strings.find_plucks(*down, now=t)
    pluck.string.pluck(t, pluck.position, 0.5)
    assert strings.find_plucks(*up, now=t + PLUCK_COOLDOWN_S / 2) == []
    assert len(strings.find_plucks(*up, now=t + PLUCK_COOLDOWN_S * 1.5)) == 1


def test_plucked_string_vibrates_then_settles():
    strings = make_set()
    string = strings.add((100.0, 200.0), (700.0, 200.0), DIAGONAL, now=0.0)
    xs = np.linspace(0, 1, 33)
    assert not string.is_vibrating(1.0)
    string.pluck(1.0, 0.5, 1.0)
    assert string.is_vibrating(1.01)
    assert np.max(np.abs(string.displacement(xs, 1.0))) > 10.0
    assert np.all(string.displacement(xs, 1.0)[[0, -1]] == pytest.approx(0.0, abs=1e-9))
    assert not string.is_vibrating(30.0)


def test_draw_gesture_commits_after_release():
    gesture = DrawGesture(min_length=80.0, release_frames=2)
    assert gesture.update([(500.0, 300.0), (100.0, 320.0)]) is None
    assert gesture.pending == ((100.0, 320.0), (500.0, 300.0))
    assert gesture.update([(100.0, 320.0)]) is None  # one-frame dropout tolerated
    assert gesture.pending is not None
    assert gesture.update([]) == ((100.0, 320.0), (500.0, 300.0))
    assert gesture.pending is None
    assert gesture.update([]) is None


def test_draw_gesture_ignores_short_strings():
    gesture = DrawGesture(min_length=80.0, release_frames=1)
    gesture.update([(100.0, 100.0), (130.0, 110.0)])
    assert gesture.update([]) is None
