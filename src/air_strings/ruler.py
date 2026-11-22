"""The ruler drawn just above a string: a baseline as long as the string, evenly
spaced ticks with ruler-style heights, and tall marks through both endpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from air_strings.geometry import Point, distance, upper_normal

RULER_OFFSET_PX = 14.0  # from the string to the ruler's baseline, on the upper side
TICK_SPACING_PX = 30.0  # roughly; the tick count is rounded to a multiple of four
MIN_DIVISIONS = 8
MAX_DIVISIONS = 32
TICK_MINOR_PX = 3.0
TICK_EVEN_PX = 5.0  # every second tick
TICK_QUARTER_PX = 8.0  # at a quarter and three quarters of the string
TICK_CENTER_PX = 11.0  # the middle of the string
ENDPOINT_MARK_ABOVE_PX = 12.0  # how far an endpoint mark runs past the ruler's baseline
ENDPOINT_MARK_BELOW_PX = 12.0  # and past the string, on the other side


@dataclass(frozen=True, eq=False)
class RulerGeometry:
    up: Point  # unit normal pointing away from the string, towards the ruler
    divisions: int
    baseline: np.ndarray  # (2, 2): parallel to the string, from above its start to above its end
    ticks: np.ndarray  # (divisions - 1, 2, 2): base on the baseline, tip away from the string
    endpoint_marks: np.ndarray  # (2, 2, 2): across each endpoint, from below the string past the ruler


def divisions_for(length: float) -> int:
    """How many equal parts a ruler `length` pixels long is divided into: a multiple of four."""
    quarters = round(length / (4.0 * TICK_SPACING_PX))
    return int(min(MAX_DIVISIONS, max(MIN_DIVISIONS, 4 * quarters)))


def tick_heights(divisions: int) -> np.ndarray:
    """Heights of the inner ticks 1..divisions-1: minor, every second one taller,
    taller still at the quarters, tallest in the middle."""
    index = np.arange(1, divisions)
    heights = np.where(index % 2 == 0, TICK_EVEN_PX, TICK_MINOR_PX)
    heights = np.where(index % (divisions // 4) == 0, TICK_QUARTER_PX, heights)
    heights[index == divisions // 2] = TICK_CENTER_PX
    return heights


def ruler_geometry(a: Point, b: Point) -> RulerGeometry | None:
    """The ruler of a string from `a` to `b`; None for a degenerate string."""
    length = distance(a, b)
    if length < 1.0:
        return None
    up = upper_normal(a, b)
    up_vector = np.asarray(up, dtype=np.float64)
    start = np.asarray(a, dtype=np.float64)
    end = np.asarray(b, dtype=np.float64)
    shift = up_vector * RULER_OFFSET_PX
    divisions = divisions_for(length)
    bases = start + shift + np.outer(np.arange(1, divisions) / divisions, end - start)
    tips = bases + np.outer(tick_heights(divisions), up_vector)
    below = -up_vector * ENDPOINT_MARK_BELOW_PX
    above = up_vector * (RULER_OFFSET_PX + ENDPOINT_MARK_ABOVE_PX)
    return RulerGeometry(
        up=up,
        divisions=divisions,
        baseline=np.stack([start + shift, end + shift]),
        ticks=np.stack([bases, tips], axis=1),
        endpoint_marks=np.stack([[p + below, p + above] for p in (start, end)]),
    )
