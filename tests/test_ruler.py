import numpy as np
import pytest

from air_strings.ruler import (
    MAX_DIVISIONS,
    MIN_DIVISIONS,
    RULER_OFFSET_PX,
    TICK_CENTER_PX,
    TICK_MINOR_PX,
    TICK_QUARTER_PX,
    TICK_SPACING_PX,
    divisions_for,
    ruler_geometry,
    tick_heights,
)


@pytest.mark.parametrize("length", [50.0, 345.0, 700.0, 2000.0])
def test_divisions_are_a_multiple_of_four_within_bounds(length):
    divisions = divisions_for(length)
    assert divisions % 4 == 0
    assert MIN_DIVISIONS <= divisions <= MAX_DIVISIONS


def test_divisions_follow_the_tick_spacing():
    assert divisions_for(12 * TICK_SPACING_PX) == 12
    assert divisions_for(20 * TICK_SPACING_PX) == 20


def test_tick_heights_look_like_a_ruler():
    heights = tick_heights(12)  # ticks 1..11
    assert heights.size == 11
    assert heights[5] == TICK_CENTER_PX == heights.max()  # the middle, tick 6
    assert heights[2] == heights[8] == TICK_QUARTER_PX  # the quarters, ticks 3 and 9
    assert heights[0] == TICK_MINOR_PX == heights.min()
    assert np.all(heights == heights[::-1])  # symmetric around the middle


@pytest.mark.parametrize(
    "a, b",
    [((100.0, 300.0), (800.0, 320.0)), ((800.0, 320.0), (100.0, 300.0)), ((400.0, 100.0), (450.0, 600.0))],
)
def test_ticks_are_evenly_spaced_over_the_whole_string_and_point_away_from_it(a, b):
    geometry = ruler_geometry(a, b)
    start, end = np.asarray(a), np.asarray(b)
    up = np.asarray(geometry.up)
    assert up[1] <= 0  # above the string on screen

    # The baseline runs parallel to the string, end to end, RULER_OFFSET_PX above it.
    assert geometry.baseline == pytest.approx(np.stack([start, end]) + up * RULER_OFFSET_PX)

    bases, tips = geometry.ticks[:, 0], geometry.ticks[:, 1]
    along = (bases - start - up * RULER_OFFSET_PX) @ (end - start) / np.dot(end - start, end - start)
    expected = np.arange(1, geometry.divisions) / geometry.divisions
    assert along == pytest.approx(expected)  # evenly spaced, from one end to the other
    assert (tips - bases) @ up == pytest.approx(tick_heights(geometry.divisions))  # away from the string


def test_endpoint_marks_cross_the_endpoints_from_below_the_string_past_the_ruler():
    a, b = (100.0, 300.0), (800.0, 300.0)
    geometry = ruler_geometry(a, b)
    for mark, endpoint in zip(geometry.endpoint_marks, (a, b)):
        low, high = mark
        assert low[0] == pytest.approx(endpoint[0]) and high[0] == pytest.approx(endpoint[0])  # across the endpoint
        assert low[1] > endpoint[1]  # below the string
        assert high[1] < endpoint[1] - RULER_OFFSET_PX  # past the ruler's baseline


def test_degenerate_string_has_no_ruler():
    assert ruler_geometry((10.0, 10.0), (10.2, 10.0)) is None
