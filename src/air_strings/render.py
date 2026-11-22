"""All OpenCV drawing: hands, strings (idle, pending, caught, vibrating with glow and trail),
stretch rulers, spark particles and the HUD."""

from __future__ import annotations

import math

import cv2
import numpy as np

from air_strings.geometry import Point, distance, midpoint, unit_normal
from air_strings.gestures import Bend, GestureView, Stretch
from air_strings.hands import Hand
from air_strings.particles import ParticleSystem
from air_strings.ruler import ruler_marks
from air_strings.scales import PitchMapping, Scale, note_name
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
CAUGHT_COLOR = (200, 255, 225)
TEXT_COLOR = (240, 240, 240)
SHADOW_COLOR = (25, 25, 25)

STRING_SAMPLES = 64
TICK_COUNT = 16
GLOW_DOWNSCALE = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX

# The vibration's envelope ("eye") lines: additive, at this fraction of the string color.
TRAIL_INTENSITY = 0.3
# Sparks are also drawn, blurred, into the glow layer at this fraction of their color.
SPARK_GLOW_INTENSITY = 0.45
# Spark colors are quantized to this step per channel so they can be drawn in batches.
STREAK_COLOR_STEP = 16

RULER_OFFSET_PX = 16.0  # from the string to the ruler's baseline, on the upper side
RULER_TALL_PX = 12.0  # notes of the active scale
RULER_SHORT_PX = 6.0  # other semitones
RULER_HALF_PX = 3.5  # half-step marks between semitones
RULER_CURRENT_PX = 17.0  # the note the string plays now
# The ruler runs this far (fraction of the frame diagonal) past the moving end,
# so the next notes are visible before the string gets there.
RULER_LOOKAHEAD_FRACTION = 0.03
RULER_COLOR = (215, 215, 215)
RULER_AHEAD_COLOR = (120, 120, 120)
RULER_CURRENT_COLOR = (150, 255, 205)


def _int_point(p) -> tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


def _upper_normal(a: Point, b: Point) -> tuple[float, float]:
    """Unit normal of a->b pointing up on screen (towards smaller y; left for vertical lines)."""
    nx, ny = unit_normal(a, b)
    if ny > 0 or (ny == 0 and nx > 0):
        nx, ny = -nx, -ny
    return nx, ny


def _bent_points(string: AirString, bend: Bend, xs: np.ndarray) -> np.ndarray:
    """Points at xs (0..1) of a string bent into two straight segments through the contact point."""
    start = np.asarray(string.start, dtype=np.float64)
    end = np.asarray(string.end, dtype=np.float64)
    contact = np.asarray(bend.contact, dtype=np.float64)
    u = min(1.0 - 1e-3, max(1e-3, bend.position))
    left = xs <= u
    out = np.empty((xs.size, 2))
    out[left] = start + np.outer(xs[left] / u, contact - start)
    out[~left] = contact + np.outer((xs[~left] - u) / (1.0 - u), end - contact)
    return out


def _text_with_shadow(frame: np.ndarray, text: str, origin: Point, scale: float, color) -> None:
    x, y = _int_point(origin)
    cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, SHADOW_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


class Renderer:
    def __init__(self, mapping: PitchMapping) -> None:
        self._mapping = mapping
        self._xs = np.linspace(0.0, 1.0, STRING_SAMPLES)
        self._tick_xs = np.arange(1, TICK_COUNT) / TICK_COUNT

    def draw_frame(
        self,
        frame: np.ndarray,
        hands: list[Hand],
        strings: list[AirString],
        gestures: GestureView,
        particles: ParticleSystem,
        scale: Scale,
        now: float,
        hud_lines: list[str],
    ) -> None:
        bends = {bend.string.string_id: bend for bend in gestures.bends}
        resizing = {s.string_id for s in gestures.stretches if s.string_id is not None}
        self._draw_strings(frame, strings, bends, resizing, particles, now)
        if gestures.pending is not None:
            self._draw_pending(frame, *gestures.pending)
        for stretch in gestures.stretches:
            self._draw_ruler(frame, stretch, scale)
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

    def _draw_strings(
        self,
        frame: np.ndarray,
        strings: list[AirString],
        bends: dict[int, Bend],
        resizing: set[int],
        particles: ParticleSystem,
        now: float,
    ) -> None:
        vibrating = [s for s in strings if s.string_id not in bends and s.is_vibrating(now)]
        if vibrating or particles.count:
            self._add_glow(frame, vibrating, particles, now)
        vibrating_ids = {s.string_id for s in vibrating}
        for string in strings:
            bend = bends.get(string.string_id)
            if bend is not None:
                color = CAUGHT_COLOR
                corners = np.array([string.start, bend.contact, string.end])
                cv2.polylines(frame, [np.round(corners).astype(np.int32)], False, color, 2, cv2.LINE_AA)
                cv2.circle(frame, _int_point(bend.contact), 3, color, -1, cv2.LINE_AA)
                tick_points = _bent_points(string, bend, self._tick_xs)
            elif string.string_id in vibrating_ids:
                color = tuple(min(255, c + 120) for c in string.color)
                curve = string.points(self._xs, now).astype(np.int32)
                cv2.polylines(frame, [curve], False, color, 2, cv2.LINE_AA)
                tick_points = string.points(self._tick_xs, now)
            else:
                color = IDLE_COLOR
                curve = string.points(self._xs, now).astype(np.int32)
                cv2.polylines(frame, [curve], False, color, 1, cv2.LINE_AA)
                tick_points = string.points(self._tick_xs, now)
            self._draw_ticks(frame, string, tick_points, color)
            if string.string_id in resizing:
                cv2.circle(frame, _int_point(string.start), 5, PENDING_COLOR, 1, cv2.LINE_AA)
                cv2.circle(frame, _int_point(string.end), 5, PENDING_COLOR, 1, cv2.LINE_AA)
            else:  # while resizing, the ruler shows the live note instead
                self._draw_label(frame, string)

    def _add_glow(self, frame: np.ndarray, strings: list[AirString], particles: ParticleSystem, now: float) -> None:
        """One additive layer: blurred glow (strings, sparks), then crisp trails and sparks."""
        height, width = frame.shape[:2]
        small_size = (width // GLOW_DOWNSCALE, height // GLOW_DOWNSCALE)
        glow = np.zeros((small_size[1], small_size[0], 3), dtype=np.uint8)
        for string in strings:
            curve = (string.points(self._xs, now) / GLOW_DOWNSCALE).astype(np.int32)
            level = string.glow_level(now)
            color = tuple(int(c * level) for c in string.color)
            cv2.polylines(glow, [curve], False, color, 3, cv2.LINE_AA)
        heads, tails, colors = particles.streaks()
        if len(heads):
            small_colors = colors * SPARK_GLOW_INTENSITY
            self._draw_streaks(glow, heads / GLOW_DOWNSCALE, tails / GLOW_DOWNSCALE, small_colors, 2, cv2.LINE_8)
        glow = cv2.GaussianBlur(glow, (0, 0), 2.5)
        layer = cv2.resize(glow, (width, height), interpolation=cv2.INTER_LINEAR)
        for string in strings:
            self._draw_trail(layer, string, now)
        if len(heads):
            self._draw_streaks(layer, heads, tails, colors, 1, cv2.LINE_AA)
        cv2.add(frame, layer, dst=frame)

    def _draw_trail(self, layer: np.ndarray, string: AirString, now: float) -> None:
        """Upper and lower envelope of the vibration: an 'eye' that shrinks as it decays."""
        profile = string.envelope_profile(self._xs, now)
        start = np.asarray(string.start, dtype=np.float64)
        end = np.asarray(string.end, dtype=np.float64)
        normal = np.asarray(unit_normal(string.start, string.end), dtype=np.float64)
        base = start + np.outer(self._xs, end - start)
        offset = np.outer(profile, normal)
        level = string.glow_level(now)
        color = tuple(int(c * TRAIL_INTENSITY * level) for c in string.color)
        lines = [np.round(base + offset).astype(np.int32), np.round(base - offset).astype(np.int32)]
        cv2.polylines(layer, lines, False, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_streaks(
        image: np.ndarray, heads: np.ndarray, tails: np.ndarray, colors: np.ndarray, thickness: int, line_type: int
    ) -> None:
        """Draw segments tail->head, batched into one polylines call per (quantized) color."""
        segments = np.round(np.stack([tails, heads], axis=1)).astype(np.int32)  # (n, 2, 2)
        quantized = (np.clip(colors, 0, 255) // STREAK_COLOR_STEP).astype(np.int32)
        palette, groups = np.unique(quantized, axis=0, return_inverse=True)
        groups = groups.reshape(-1)
        for index, color in enumerate(palette):
            if not color.any():
                continue  # fully faded
            bgr = tuple(int(c) * STREAK_COLOR_STEP + STREAK_COLOR_STEP // 2 for c in color)
            cv2.polylines(image, list(segments[groups == index]), False, bgr, thickness, line_type)

    def _draw_ticks(self, frame: np.ndarray, string: AirString, points: np.ndarray, color) -> None:
        nx, ny = unit_normal(string.start, string.end)
        for index, (x, y) in enumerate(points, start=1):
            half = 7.0 if index % 4 == 0 else 4.0
            p0 = (x - nx * half, y - ny * half)
            p1 = (x + nx * half, y + ny * half)
            cv2.line(frame, _int_point(p0), _int_point(p1), color, 1, cv2.LINE_AA)

    def _draw_label(self, frame: np.ndarray, string: AirString) -> None:
        nx, ny = _upper_normal(string.start, string.end)
        cx, cy = midpoint(string.start, string.end)
        anchor = (cx + nx * 26.0 - 14.0, cy + ny * 26.0 + 6.0)
        cv2.putText(frame, string.note_name, _int_point(anchor), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, string.note_name, _int_point(anchor), FONT, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

    def _draw_ruler(self, frame: np.ndarray, stretch: Stretch, scale: Scale) -> None:
        """A measuring ruler just above the string: one mark per note length, from the anchor."""
        a, b = stretch.anchor, stretch.moving
        length = distance(a, b)
        if length < 1.0:
            return
        height, width = frame.shape[:2]
        diagonal = math.hypot(width, height)
        tx, ty = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        nx, ny = _upper_normal(a, b)
        ox, oy = a[0] + nx * RULER_OFFSET_PX, a[1] + ny * RULER_OFFSET_PX
        marks = ruler_marks(self._mapping, scale, length + RULER_LOOKAHEAD_FRACTION * diagonal, diagonal)

        # Baseline: solid up to the moving end, dim where it runs ahead of it.
        far = max([length] + [m.distance for m in marks])
        moving_end = (ox + tx * length, oy + ty * length)
        cv2.line(frame, _int_point((ox, oy)), _int_point(moving_end), RULER_COLOR, 1, cv2.LINE_AA)
        if far > length:
            ahead = (ox + tx * far, oy + ty * far)
            cv2.line(frame, _int_point(moving_end), _int_point(ahead), RULER_AHEAD_COLOR, 1, cv2.LINE_AA)

        label_at = (ox + tx * length, oy + ty * length)
        label_height = RULER_TALL_PX
        for mark in marks:
            base = (ox + tx * mark.distance, oy + ty * mark.distance)
            if mark.midi == stretch.midi:
                size, color = RULER_CURRENT_PX, RULER_CURRENT_COLOR
                label_at, label_height = base, size
            else:
                size = RULER_TALL_PX if mark.in_scale else RULER_SHORT_PX if mark.midi is not None else RULER_HALF_PX
                color = RULER_AHEAD_COLOR if mark.distance > length else RULER_COLOR
            tip = (base[0] + nx * size, base[1] + ny * size)
            thickness = 2 if mark.midi == stretch.midi else 1
            cv2.line(frame, _int_point(base), _int_point(tip), color, thickness, cv2.LINE_AA)

        text = note_name(stretch.midi)
        (text_width, text_height), _ = cv2.getTextSize(text, FONT, 0.55, 1)
        center = (label_at[0] + nx * (label_height + 12.0), label_at[1] + ny * (label_height + 12.0))
        origin = (center[0] - text_width / 2.0, center[1] + text_height / 2.0)
        _text_with_shadow(frame, text, origin, 0.55, RULER_CURRENT_COLOR)

    def _draw_hud(self, frame: np.ndarray, lines: list[str]) -> None:
        line_height = 22
        box_width = 12 + max((cv2.getTextSize(line, FONT, 0.5, 1)[0][0] for line in lines), default=0)
        box_height = 10 + line_height * len(lines)
        region = frame[8 : 8 + box_height, 8 : 8 + box_width]
        region[:] = (region * 0.45).astype(np.uint8)
        for index, line in enumerate(lines):
            origin = (14, 8 + line_height * (index + 1))
            cv2.putText(frame, line, origin, FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
