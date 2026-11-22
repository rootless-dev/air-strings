"""All OpenCV drawing: hand landmarks, strings (idle, pending, vibrating with glow) and the HUD."""

from __future__ import annotations

import cv2
import numpy as np

from air_strings.geometry import Point, distance, midpoint, unit_normal
from air_strings.hands import Hand
from air_strings.strings import AirString

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)  # fmt: skip

BONE_COLOR = (0, 220, 0)
JOINT_COLOR = (0, 0, 255)
PINCH_COLOR = (255, 255, 255)
IDLE_COLOR = (235, 235, 235)
PENDING_COLOR = (255, 255, 255)
TEXT_COLOR = (240, 240, 240)

STRING_SAMPLES = 64
TICK_COUNT = 16
GLOW_DOWNSCALE = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _int_point(p) -> tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


class Renderer:
    def __init__(self) -> None:
        self._xs = np.linspace(0.0, 1.0, STRING_SAMPLES)
        self._tick_xs = np.arange(1, TICK_COUNT) / TICK_COUNT

    def draw_frame(
        self,
        frame: np.ndarray,
        hands: list[Hand],
        strings: list[AirString],
        pending: tuple[Point, Point] | None,
        now: float,
        hud_lines: list[str],
    ) -> None:
        self._draw_strings(frame, strings, now)
        if pending is not None:
            self._draw_pending(frame, *pending)
        for hand in hands:
            self._draw_hand(frame, hand)
        self._draw_hud(frame, hud_lines)

    def _draw_hand(self, frame: np.ndarray, hand: Hand) -> None:
        points = [_int_point(p) for p in hand.points]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], BONE_COLOR, 2, cv2.LINE_AA)
        for p in points:
            cv2.circle(frame, p, 4, JOINT_COLOR, -1, cv2.LINE_AA)
        if hand.pinching:
            cv2.circle(frame, _int_point(hand.pinch_point), 9, PINCH_COLOR, 2, cv2.LINE_AA)

    def _draw_pending(self, frame: np.ndarray, a: Point, b: Point) -> None:
        length = distance(a, b)
        dash, gap = 14.0, 9.0
        position = 0.0
        while position < length:
            t0 = position / length
            t1 = min(position + dash, length) / length
            p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
            p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
            cv2.line(frame, _int_point(p0), _int_point(p1), PENDING_COLOR, 2, cv2.LINE_AA)
            position += dash + gap
        for p in (a, b):
            cv2.circle(frame, _int_point(p), 6, PENDING_COLOR, 2, cv2.LINE_AA)

    def _draw_strings(self, frame: np.ndarray, strings: list[AirString], now: float) -> None:
        vibrating = [s for s in strings if s.is_vibrating(now)]
        if vibrating:
            self._add_glow(frame, vibrating, now)
        vibrating_ids = {s.string_id for s in vibrating}
        for string in strings:
            is_vibrating = string.string_id in vibrating_ids
            curve = string.points(self._xs, now).astype(np.int32)
            if is_vibrating:
                color = tuple(min(255, c + 120) for c in string.color)
                cv2.polylines(frame, [curve], False, color, 2, cv2.LINE_AA)
            else:
                color = IDLE_COLOR
                cv2.polylines(frame, [curve], False, color, 1, cv2.LINE_AA)
            self._draw_ticks(frame, string, now, color)
            self._draw_label(frame, string)

    def _add_glow(self, frame: np.ndarray, strings: list[AirString], now: float) -> None:
        height, width = frame.shape[:2]
        small_size = (width // GLOW_DOWNSCALE, height // GLOW_DOWNSCALE)
        glow = np.zeros((small_size[1], small_size[0], 3), dtype=np.uint8)
        for string in strings:
            curve = (string.points(self._xs, now) / GLOW_DOWNSCALE).astype(np.int32)
            level = string.glow_level(now)
            color = tuple(int(c * level) for c in string.color)
            cv2.polylines(glow, [curve], False, color, 3, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 2.5)
        glow = cv2.resize(glow, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.add(frame, glow, dst=frame)
        cv2.add(frame, glow, dst=frame)

    def _draw_ticks(self, frame: np.ndarray, string: AirString, now: float, color) -> None:
        nx, ny = unit_normal(string.start, string.end)
        points = string.points(self._tick_xs, now)
        for index, (x, y) in enumerate(points, start=1):
            half = 7.0 if index % 4 == 0 else 4.0
            p0 = (x - nx * half, y - ny * half)
            p1 = (x + nx * half, y + ny * half)
            cv2.line(frame, _int_point(p0), _int_point(p1), color, 1, cv2.LINE_AA)

    def _draw_label(self, frame: np.ndarray, string: AirString) -> None:
        nx, ny = unit_normal(string.start, string.end)
        if ny > 0:  # put the label above the string
            nx, ny = -nx, -ny
        cx, cy = midpoint(string.start, string.end)
        anchor = (cx + nx * 26.0 - 14.0, cy + ny * 26.0 + 6.0)
        cv2.putText(frame, string.note_name, _int_point(anchor), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, string.note_name, _int_point(anchor), FONT, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, lines: list[str]) -> None:
        line_height = 22
        box_width = 12 + max((cv2.getTextSize(line, FONT, 0.5, 1)[0][0] for line in lines), default=0)
        box_height = 10 + line_height * len(lines)
        region = frame[8 : 8 + box_height, 8 : 8 + box_width]
        region[:] = (region * 0.45).astype(np.uint8)
        for index, line in enumerate(lines):
            origin = (14, 8 + line_height * (index + 1))
            cv2.putText(frame, line, origin, FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
