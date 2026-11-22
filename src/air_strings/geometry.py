"""2D geometry helpers: side-of-line tests, segment crossing and projection."""

from __future__ import annotations

import math

Point = tuple[float, float]

_EPSILON = 1e-9


def cross(ax: float, ay: float, bx: float, by: float) -> float:
    """Z component of the cross product of two 2D vectors."""
    return ax * by - ay * bx


def side_of_line(a: Point, b: Point, p: Point) -> float:
    """Signed area telling on which side of the line a->b the point p lies.

    Positive on one side, negative on the other, zero on the line.
    """
    return cross(b[0] - a[0], b[1] - a[1], p[0] - a[0], p[1] - a[1])


def distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def crossing_position(
    move_start: Point, move_end: Point, seg_a: Point, seg_b: Point
) -> float | None:
    """Return where a movement crosses a segment, as a parameter 0..1 along the segment.

    The movement crosses when its start and end lie on opposite sides of the
    segment's line (a sign change; landing exactly on the line counts, starting
    on it does not) and the intersection point lies within both the movement
    and the segment. Returns None when there is no crossing, including when the
    movement is parallel to the segment.
    """
    side_start = side_of_line(seg_a, seg_b, move_start)
    side_end = side_of_line(seg_a, seg_b, move_end)
    if side_start > 0:
        if side_end > 0:
            return None
    elif side_start < 0:
        if side_end < 0:
            return None
    else:
        return None

    rx, ry = move_end[0] - move_start[0], move_end[1] - move_start[1]
    sx, sy = seg_b[0] - seg_a[0], seg_b[1] - seg_a[1]
    denom = cross(rx, ry, sx, sy)
    if abs(denom) < _EPSILON:
        return None
    qpx, qpy = seg_a[0] - move_start[0], seg_a[1] - move_start[1]
    t = cross(qpx, qpy, sx, sy) / denom  # along the movement
    u = cross(qpx, qpy, rx, ry) / denom  # along the segment
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None
    return u


def project_param(a: Point, b: Point, p: Point) -> float:
    """Parameter (clamped to 0..1) of the orthogonal projection of p onto segment a->b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq < _EPSILON:
        return 0.0
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq
    return min(1.0, max(0.0, t))


def unit_normal(a: Point, b: Point) -> Point:
    """Unit vector perpendicular to a->b (zero vector for a degenerate segment)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < _EPSILON:
        return (0.0, 0.0)
    return (-dy / length, dx / length)
