"""Renderer checks on a black frame, where the additive layer is all that shows."""

import math

import numpy as np

from air_strings.gestures import Bend, GestureView, Stretch
from air_strings.particles import ParticleSystem
from air_strings.render import Renderer
from air_strings.ruler import RULER_OFFSET_PX, TICK_CENTER_PX, TICK_MINOR_PX
from air_strings.scales import SCALES, PitchMapping
from air_strings.strings import StringSet

DIAGONAL = math.hypot(1280, 720)
EMPTY = GestureView(None, (), ())
LEFT, RIGHT, Y = 200.0, 1000.0, 400.0
BASELINE_Y = int(Y - RULER_OFFSET_PX)


def make_string():
    strings = StringSet(PitchMapping(), SCALES[0])
    return strings.add((LEFT, Y), (RIGHT, Y), DIAGONAL, now=0.0)


def render(strings, view=EMPTY, now=5.0, particles=None):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    particles = particles if particles is not None else ParticleSystem(rng=np.random.default_rng(0))
    Renderer().draw_frame(frame, [], strings, view, particles, now)
    return frame


def lit_columns(frame, y, x0, x1, threshold=30):
    """Share of columns x0..x1 with a lit pixel within a pixel of row y."""
    band = frame[y - 1 : y + 2, int(x0) : int(x1)].max(axis=(0, 2))
    return np.count_nonzero(band > threshold) / band.size


def test_a_committed_string_keeps_a_dim_ruler_over_its_whole_length():
    frame = render([make_string()])
    assert lit_columns(frame, BASELINE_Y, LEFT + 2, RIGHT - 2) > 0.98
    assert frame[BASELINE_Y - 1 : BASELINE_Y + 2, 600].max() < 150  # dim
    # The tallest tick sits over the middle, the shorter ones evenly on both sides.
    center_tip = int(Y - RULER_OFFSET_PX - TICK_CENTER_PX + 2)
    assert frame[center_tip, 599:602].max() > 30
    above_minor_ticks = int(Y - RULER_OFFSET_PX - TICK_MINOR_PX - 3)
    assert frame[above_minor_ticks, 620:640].max() < 30  # between two ticks, only the minor heights
    # Nothing beyond the string's ends.
    assert lit_columns(frame, BASELINE_Y, RIGHT + 15, RIGHT + 200) == 0.0


def test_the_ruler_stays_while_the_string_vibrates():
    string = make_string()
    string.pluck(5.0, 0.4, 1.0, displacement=-30.0)
    frame = render([string], now=5.1)
    assert lit_columns(frame, BASELINE_Y, LEFT + 2, RIGHT - 2) > 0.98


def test_stretching_draws_a_bright_ruler_and_no_dim_one():
    string = make_string()
    stretch = Stretch((LEFT, Y), (RIGHT, Y), string.midi, string.string_id)
    frame = render([string], GestureView(None, (stretch,), ()))
    assert frame[BASELINE_Y - 1 : BASELINE_Y + 2, 600].max() > 150  # bright
    drawing = render([], GestureView(((LEFT, Y), (RIGHT, Y)), (Stretch((LEFT, Y), (RIGHT, Y), 60),), ()))
    assert lit_columns(drawing, BASELINE_Y, LEFT + 2, RIGHT - 2) > 0.98


def test_an_idle_string_is_a_white_line():
    frame = render([make_string()])
    b, g, r = (int(c) for c in frame[int(Y), 600])
    assert min(b, g, r) > 150
    assert max(b, g, r) - min(b, g, r) < 10


def test_a_vibrating_string_goes_from_lime_on_the_left_to_cyan_on_the_right():
    for start, end in (((LEFT, Y), (RIGHT, Y)), ((RIGHT, Y), (LEFT, Y))):  # either drawing direction
        strings = StringSet(PitchMapping(), SCALES[0])
        string = strings.add(start, end, DIAGONAL, now=0.0)
        string.pluck(5.0, 0.5, 1.0, displacement=-40.0)
        frame = render([string], now=5.15).astype(np.int64)
        band = slice(int(Y) - 60, int(Y) + 60)
        left = frame[band, int(LEFT) + 20 : int(LEFT) + 200].reshape(-1, 3).sum(axis=0)
        right = frame[band, int(RIGHT) - 200 : int(RIGHT) - 20].reshape(-1, 3).sum(axis=0)
        assert left[2] > left[0]  # more red than blue: yellow-green
        assert right[0] > right[2]  # more blue than red: cyan


def test_a_held_string_is_drawn_through_the_fingertip():
    string = make_string()
    contact = (520.0, Y - 35.0)
    frame = render([string], GestureView(None, (), (Bend(string, 0.4, contact),)))
    assert frame[int(contact[1]) - 2 : int(contact[1]) + 3, 518:523].max() > 150
    assert frame[int(Y) - 1 : int(Y) + 2, 515:525].max() < 120  # no longer on the rest line there
