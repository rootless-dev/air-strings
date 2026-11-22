import itertools
import math

import numpy as np
import pytest

from air_strings import string_fx
from air_strings.scales import SCALES, PitchMapping
from air_strings.strings import StringSet, visual_frequency

DIAGONAL = math.hypot(1280, 720)
XS = np.linspace(0.0, 1.0, 101)


def make_string(start=(100.0, 300.0), end=(700.0, 300.0)):
    strings = StringSet(PitchMapping(), SCALES[0])
    return strings.add(start, end, DIAGONAL, now=0.0)


def test_a_vibrating_string_is_drawn_as_a_few_strands_the_current_one_brightest():
    string = make_string()
    string.pluck(1.0, 0.5, 1.0, displacement=40.0)
    offsets, weights = string_fx.vibration_strands(string, XS, 1.2)
    assert 3 <= len(weights) <= 5
    assert offsets.shape == (len(weights), XS.size)
    assert weights[0] == 1.0 and np.all(weights[1:] < weights[0])
    assert np.all(offsets[:, [0, -1]] == pytest.approx(0.0, abs=1e-9))  # the ends stay put


@pytest.mark.parametrize("periods", [0.0, 0.2, 0.75, 1.0, 1.3, 1.6, 2.2, 3.0])
def test_mid_pluck_strands_span_an_eye_symmetric_around_the_rest_line(periods):
    string = make_string()
    string.pluck(1.0, 0.5, 1.0, displacement=40.0)
    now = 1.0 + periods / visual_frequency(string.frequency)
    amplitude = string.amplitude(now)
    offsets, _ = string_fx.vibration_strands(string, XS, now)
    middle = offsets[:, 50]
    above, below = middle.max(), -middle.min()
    # Both sides of the rest line are covered, about equally, by most of the envelope.
    assert above > 0.6 * amplitude and below > 0.6 * amplitude
    assert 0.7 < above / below < 1.4
    assert max(above, below) < 1.35 * amplitude


def test_strands_converge_to_the_rest_line_and_the_ghosts_fade_gradually():
    string = make_string()
    string.pluck(1.0, 0.4, 1.0, displacement=40.0)
    times = np.linspace(1.0, 1.0 + 20.0, 400)
    vibrating = [t for t in times if string.is_vibrating(t)]
    ghost_levels = [string_fx.vibration_strands(string, XS, t)[1][1:].max() for t in vibrating]
    widths = [np.abs(string_fx.vibration_strands(string, XS, t)[0]).max() for t in vibrating]
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(ghost_levels))  # never brighter again
    assert ghost_levels[0] == pytest.approx(max(string_fx.GHOST_WEIGHTS[1:]))
    assert ghost_levels[-1] < 0.02  # practically gone just before the vibration ends: no abrupt cut
    assert widths[-1] < 1.0  # merged into the rest line
    steps = np.diff(ghost_levels)
    assert np.abs(steps).max() < 0.05  # a smooth fade, frame after frame


def test_wobble_is_deterministic_per_seed_and_proportional_to_the_amplitude():
    a = string_fx.wobble(XS, 0.37, 10.0, seed=3)
    assert np.array_equal(a, string_fx.wobble(XS, 0.37, 10.0, seed=3))
    assert not np.allclose(a, string_fx.wobble(XS, 0.37, 10.0, seed=4))
    assert np.abs(a).max() > 0.2  # it is there
    assert string_fx.wobble(XS, 0.37, 20.0, seed=3) == pytest.approx(2.0 * a)
    assert np.all(string_fx.wobble(XS, 0.37, 0.0, seed=3) == 0.0)
    assert np.abs(string_fx.wobble(XS, 0.37, 1e-6, seed=3)).max() < 1e-6
    assert a[[0, -1]] == pytest.approx(0.0, abs=1e-9)
    bound = (string_fx.WOBBLE_OVERTONE_RATIO + string_fx.WOBBLE_RIPPLE_RATIO) * 10.0
    assert np.abs(a).max() <= bound


def test_wobble_moves_smoothly_from_frame_to_frame():
    frames = [string_fx.wobble(XS, t, 10.0, seed=1, rate=5.0) for t in np.arange(0.0, 2.0, 1 / 30)]
    peak = max(np.abs(f).max() for f in frames)
    largest_step = max(np.abs(b - a).max() for a, b in itertools.pairwise(frames))
    assert largest_step < 0.6 * peak  # motion, not a new random shape every frame


def test_held_string_is_a_smooth_curve_through_the_fingertip():
    string = make_string()
    position, depth = 0.4, -30.0
    contact = (100.0 + 600.0 * position, 300.0 + depth)  # the normal of a left-to-right string is +y
    offsets, weights = string_fx.held_strands(string, position, contact, XS, now=3.0)
    current = offsets[0]
    assert current[40] == pytest.approx(depth)  # through the fingertip
    assert current[[0, -1]] == pytest.approx(0.0, abs=1e-9)
    assert weights[0] == 1.0 and np.all(weights[1:] < 0.5)  # ghosts are faint
    # Much less of a corner at the fingertip than the straight-segment V.
    slope_jump = abs((current[41] - current[40]) - (current[40] - current[39])) * 100.0
    v_jump = abs(depth) * (1.0 / position + 1.0 / (1.0 - position))
    assert slope_jump < 0.5 * v_jump


def test_bend_shape_is_one_at_the_fingertip_and_zero_at_the_ends():
    shape = string_fx.bend_shape(XS, 0.3)
    assert shape[30] == pytest.approx(1.0)
    assert shape[[0, -1]] == pytest.approx(0.0)
    assert shape.max() == pytest.approx(1.0)
    triangle = np.minimum(XS / 0.3, (1 - XS) / 0.7)
    assert np.all(shape >= triangle - 1e-12)  # rounded outwards, not a V


def test_the_gradient_runs_from_lime_on_the_left_to_cyan_on_the_right_on_screen():
    left_to_right = string_fx.gradient_positions((100.0, 300.0), (700.0, 300.0), XS)
    right_to_left = string_fx.gradient_positions((700.0, 300.0), (100.0, 300.0), XS)
    assert left_to_right[0] == 0.0 and right_to_left[0] == 1.0
    assert right_to_left == pytest.approx(left_to_right[::-1])
    colors = string_fx.gradient_colors(np.array([0.0, 1.0]), string_fx.GLOW_LEFT, string_fx.GLOW_RIGHT)
    (lb, lg, lr), (rb, rg, rr) = colors  # BGR
    assert lg > lb and lr > lb  # yellow-green
    assert rb > rr and rg > rr  # cyan
