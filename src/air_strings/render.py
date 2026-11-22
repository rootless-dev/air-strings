"""All OpenCV drawing: hands, strings, rulers, spark particles and the HUD.

Strings (as glowing ghost strands), their glow, rulers, endpoint marks and sparks
are drawn into one additive layer, which is then added onto the camera frame so
they read as light. The pending string, note labels, hands and the HUD are drawn
opaque on top. The shapes come from string_fx.py and ruler.py.
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

from air_strings import string_fx
from air_strings.geometry import Point, distance, midpoint, unit_normal, upper_normal
from air_strings.gestures import Bend, GestureView, Stretch
from air_strings.hands import Hand
from air_strings.particles import ParticleSystem
from air_strings.ruler import (
    ENDPOINT_MARK_ABOVE_PX,
    RULER_OFFSET_PX,
    TICK_CENTER_PX,
    ruler_geometry,
)
from air_strings.scales import note_name
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
PENDING_COLOR = (255, 255, 255)
TEXT_COLOR = (225, 225, 225)
LIVE_NOTE_COLOR = (150, 255, 205)
SHADOW_COLOR = (25, 25, 25)
FONT = cv2.FONT_HERSHEY_SIMPLEX

STRING_SAMPLES = 97
GRADIENT_CHUNKS = 6  # a strand is drawn in this many pieces, each in its own color of the gradient
GLOW_CHUNKS = 3
SUBPIXEL_BITS = 2  # lines are drawn with 1/4 pixel precision
SUBPIXEL = 1 << SUBPIXEL_BITS

# Strand cores: idle strings are a subtle, nearly white line; vibrating (or held)
# strings brighten and take on more of the gradient's color.
IDLE_BRIGHTNESS = 0.75
IDLE_TINT = 0.0
ACTIVE_TINT = 0.42
ENDPOINT_RADIUS_PX = 5

# Glow: drawn at half resolution and blurred only a little, so the aura stays narrow.
GLOW_DOWNSCALE = 2
GLOW_SIGMA = 1.5  # in downscaled pixels
GLOW_THICKNESS = 2  # in downscaled pixels
GLOW_GAIN = 0.9
ENDPOINT_GLOW_RADIUS = 2  # in downscaled pixels
ENDPOINT_GLOW_GAIN = 0.6
HELD_GLOW_PX = 25.0  # a held string glows fully once pulled this far from its rest line
SPARK_THICKNESS = 1
SPARK_GLOW_INTENSITY = 0.4
# Spark colors are quantized to this step per channel so they can be drawn in batches.
STREAK_COLOR_STEP = 16

RULER_COLOR = (190, 190, 190)  # while a string is drawn or resized
RULER_MARK_COLOR = (235, 235, 235)
RULER_DIM_COLOR = (85, 85, 85)  # on committed strings
RULER_DIM_MARK_COLOR = (175, 175, 175)


def _int_point(p) -> tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


def _fixed(points: np.ndarray) -> np.ndarray:
    """Pixel coordinates as int32 fixed point, for drawing with `shift=SUBPIXEL_BITS`."""
    return np.round(points * SUBPIXEL).astype(np.int32)


def _bgr(color) -> tuple[int, int, int]:
    return tuple(int(min(255.0, max(0.0, c))) for c in color)


def _bgr_table(colors: np.ndarray) -> list:
    """Float colors (..., 3) as nested lists of int BGR triples, clipped to 0..255."""
    return np.clip(colors, 0.0, 255.0).astype(np.int32).tolist()


def _chunk_slices(samples: int, chunks: int) -> list[slice]:
    """Overlapping slices (they share their end points) that cover `samples` points in `chunks` pieces."""
    edges = np.linspace(0, samples - 1, chunks + 1).round().astype(int)
    return [slice(int(a), int(b) + 1) for a, b in itertools.pairwise(edges)]


def _text_with_shadow(frame: np.ndarray, text: str, center: Point, scale: float, color) -> None:
    (width, height), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = _int_point((center[0] - width / 2.0, center[1] + height / 2.0))
    cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, SHADOW_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


class _Box:
    """Bounding box of everything drawn into the glow layer, in full-resolution pixels."""

    def __init__(self) -> None:
        self.x0 = self.y0 = math.inf
        self.x1 = self.y1 = -math.inf

    def add(self, points: np.ndarray) -> None:
        flat = points.reshape(-1, 2)
        low, high = flat.min(axis=0), flat.max(axis=0)
        self.x0, self.y0 = min(self.x0, low[0]), min(self.y0, low[1])
        self.x1, self.y1 = max(self.x1, high[0]), max(self.y1, high[1])

    @property
    def empty(self) -> bool:
        return self.x0 > self.x1


class Renderer:
    def __init__(self) -> None:
        self._xs = np.linspace(0.0, 1.0, STRING_SAMPLES)
        self._chunks = _chunk_slices(STRING_SAMPLES, GRADIENT_CHUNKS)
        self._glow_chunks = _chunk_slices(STRING_SAMPLES, GLOW_CHUNKS)
        self._chunk_mids = np.array([(s.start + s.stop - 1) / 2.0 for s in self._chunks]) / (STRING_SAMPLES - 1)
        self._glow_mids = np.array([(s.start + s.stop - 1) / 2.0 for s in self._glow_chunks]) / (STRING_SAMPLES - 1)
        self._layer = np.zeros((0, 0, 3), dtype=np.uint8)
        self._glow = np.zeros((0, 0, 3), dtype=np.uint8)

    def draw_frame(
        self,
        frame: np.ndarray,
        hands: list[Hand],
        strings: list[AirString],
        gestures: GestureView,
        particles: ParticleSystem,
        now: float,
        hud_lines: list[str],
    ) -> None:
        layer, glow = self._clear_layers(frame.shape)
        box = _Box()
        bends = {bend.string.string_id: bend for bend in gestures.bends}
        stretched = {s.string_id for s in gestures.stretches if s.string_id is not None}
        for string in strings:
            self._draw_string(layer, glow, box, string, bends.get(string.string_id), now)
            if string.string_id not in stretched:  # a stretched string shows the bright ruler instead
                self._draw_ruler(layer, string.start, string.end, RULER_DIM_COLOR, RULER_DIM_MARK_COLOR)
        for stretch in gestures.stretches:
            self._draw_ruler(layer, stretch.anchor, stretch.moving, RULER_COLOR, RULER_MARK_COLOR)
        self._draw_sparks(layer, glow, box, particles)
        self._compose_glow(layer, glow, box)
        cv2.add(frame, layer, dst=frame)

        if gestures.pending is not None:
            self._draw_pending(frame, *gestures.pending)
        for string in strings:
            if string.string_id not in stretched:
                self._draw_label(frame, string)
        for stretch in gestures.stretches:
            self._draw_live_note(frame, stretch)
        for hand in hands:
            self._draw_hand(frame, hand)
        self._draw_hud(frame, hud_lines)

    def _clear_layers(self, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        height, width = shape[:2]
        if self._layer.shape[:2] != (height, width):
            self._layer = np.zeros((height, width, 3), dtype=np.uint8)
            small = (-(-height // GLOW_DOWNSCALE), -(-width // GLOW_DOWNSCALE), 3)
            self._glow = np.zeros(small, dtype=np.uint8)
        else:
            self._layer.fill(0)
            self._glow.fill(0)
        return self._layer, self._glow

    # Strings

    def _draw_string(
        self, layer: np.ndarray, glow: np.ndarray, box: _Box, string: AirString, bend: Bend | None, now: float
    ) -> None:
        xs = self._xs
        if bend is not None:
            offsets, weights = string_fx.held_strands(string, bend.position, bend.contact, xs, now)
            level = min(1.0, abs(string_fx.pull_depth(string.start, string.end, bend.contact)) / HELD_GLOW_PX)
        elif string.is_vibrating(now):
            offsets, weights = string_fx.vibration_strands(string, xs, now)
            level = string.glow_level(now)
        else:
            offsets, weights, level = np.zeros((1, xs.size)), np.ones(1), 0.0

        start = np.asarray(string.start, dtype=np.float64)
        end = np.asarray(string.end, dtype=np.float64)
        normal = np.asarray(unit_normal(string.start, string.end), dtype=np.float64)
        base = start + np.outer(xs, end - start)
        curves = base[None, :, :] + offsets[:, :, None] * normal  # (strands, points, 2)

        # Core colors: from nearly white (idle) towards the gradient, brighter when active.
        tint = IDLE_TINT + (ACTIVE_TINT - IDLE_TINT) * level
        brightness = IDLE_BRIGHTNESS + (1.0 - IDLE_BRIGHTNESS) * level
        mids = string_fx.gradient_positions(string.start, string.end, self._chunk_mids)
        gradient = string_fx.gradient_colors(mids, string_fx.CORE_LEFT, string_fx.CORE_RIGHT)
        cores = brightness * ((1.0 - tint) * 255.0 + tint * gradient)
        strand_colors = _bgr_table(weights[:, None, None] * cores[None, :, :])  # [strand][chunk]
        fixed = _fixed(curves)
        for index, weight in enumerate(weights):
            if weight < 0.02:
                continue
            thickness = 2 if index == 0 else 1
            for chunk, color in zip(self._chunks, strand_colors[index]):
                cv2.polylines(layer, [fixed[index, chunk]], False, color, thickness, cv2.LINE_AA, SUBPIXEL_BITS)

        ends = string_fx.gradient_positions(string.start, string.end, np.array([0.0, 1.0]))
        end_colors = string_fx.gradient_colors(ends, string_fx.CORE_LEFT, string_fx.CORE_RIGHT)
        for point, color in zip((string.start, string.end), end_colors):
            ring = brightness * (0.5 * 255.0 + 0.5 * color)
            cv2.circle(layer, _int_point(point), ENDPOINT_RADIUS_PX, _bgr(ring), 1, cv2.LINE_AA)

        if level > 0.0:
            self._draw_strand_glow(glow, curves, weights, level, ends)
            box.add(curves)

    def _draw_strand_glow(
        self, glow: np.ndarray, curves: np.ndarray, weights: np.ndarray, level: float, ends: np.ndarray
    ) -> None:
        """Strands and endpoints in the saturated gradient, into the (downscaled) glow layer.

        `ends` are the gradient positions (0: left on screen, 1: right) of the string's start and end.
        """
        mids = ends[0] + (ends[1] - ends[0]) * self._glow_mids
        glows = string_fx.gradient_colors(mids, string_fx.GLOW_LEFT, string_fx.GLOW_RIGHT) * (GLOW_GAIN * level)
        glow_colors = _bgr_table(weights[:, None, None] * glows[None, :, :])
        fixed_small = _fixed(curves / GLOW_DOWNSCALE)
        # The glow is blurred afterwards: plain (not anti-aliased) lines are enough, and cheaper.
        for index, weight in enumerate(weights):
            if weight < 0.02:
                continue
            for chunk, color in zip(self._glow_chunks, glow_colors[index]):
                cv2.polylines(
                    glow, [fixed_small[index, chunk]], False, color, GLOW_THICKNESS, cv2.LINE_8, SUBPIXEL_BITS
                )
        end_glows = string_fx.gradient_colors(ends, string_fx.GLOW_LEFT, string_fx.GLOW_RIGHT)
        endpoints = curves[0, [0, -1]]
        for point, color in zip(endpoints, _bgr_table(end_glows * (ENDPOINT_GLOW_GAIN * level))):
            center = tuple(int(v) for v in _fixed(point / GLOW_DOWNSCALE))
            cv2.circle(glow, center, ENDPOINT_GLOW_RADIUS * SUBPIXEL, color, -1, cv2.LINE_8, SUBPIXEL_BITS)

    def _draw_label(self, frame: np.ndarray, string: AirString) -> None:
        """The string's note, small, above the middle of its ruler."""
        up = upper_normal(string.start, string.end)
        center = midpoint(string.start, string.end)
        lift = RULER_OFFSET_PX + TICK_CENTER_PX + 14.0
        anchor = (center[0] + up[0] * lift, center[1] + up[1] * lift)
        _text_with_shadow(frame, string.note_name, anchor, 0.45, TEXT_COLOR)

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

    # Rulers

    @staticmethod
    def _draw_ruler(layer: np.ndarray, a: Point, b: Point, color, mark_color) -> None:
        geometry = ruler_geometry(a, b)
        if geometry is None:
            return
        lines = [_fixed(geometry.baseline)] + list(_fixed(geometry.ticks))
        cv2.polylines(layer, lines, False, color, 1, cv2.LINE_AA, SUBPIXEL_BITS)
        cv2.polylines(layer, list(_fixed(geometry.endpoint_marks)), False, mark_color, 1, cv2.LINE_AA, SUBPIXEL_BITS)

    def _draw_live_note(self, frame: np.ndarray, stretch: Stretch) -> None:
        """The note a string being stretched plays now, above its moving end."""
        up = upper_normal(stretch.anchor, stretch.moving)
        lift = RULER_OFFSET_PX + ENDPOINT_MARK_ABOVE_PX + 12.0
        center = (stretch.moving[0] + up[0] * lift, stretch.moving[1] + up[1] * lift)
        _text_with_shadow(frame, note_name(stretch.midi), center, 0.55, LIVE_NOTE_COLOR)

    # Sparks and glow

    def _draw_sparks(self, layer: np.ndarray, glow: np.ndarray, box: _Box, particles: ParticleSystem) -> None:
        heads, tails, colors = particles.streaks()
        if not len(heads):
            return
        self._draw_streaks(layer, heads, tails, colors, SPARK_THICKNESS, cv2.LINE_AA)
        small_colors = colors * SPARK_GLOW_INTENSITY
        self._draw_streaks(glow, heads / GLOW_DOWNSCALE, tails / GLOW_DOWNSCALE, small_colors, 1, cv2.LINE_AA)
        box.add(heads)
        box.add(tails)

    @staticmethod
    def _draw_streaks(
        image: np.ndarray, heads: np.ndarray, tails: np.ndarray, colors: np.ndarray, thickness: int, line_type: int
    ) -> None:
        """Draw segments tail->head, batched into one polylines call per (quantized) color."""
        segments = _fixed(np.stack([tails, heads], axis=1))  # (n, 2, 2)
        quantized = (np.clip(colors, 0, 255) // STREAK_COLOR_STEP).astype(np.int32)
        palette, groups = np.unique(quantized, axis=0, return_inverse=True)
        groups = groups.reshape(-1)
        for index, color in enumerate(palette):
            if not color.any():
                continue  # fully faded
            bgr = tuple(int(c) * STREAK_COLOR_STEP + STREAK_COLOR_STEP // 2 for c in color)
            cv2.polylines(image, list(segments[groups == index]), False, bgr, thickness, line_type, SUBPIXEL_BITS)

    @staticmethod
    def _compose_glow(layer: np.ndarray, glow: np.ndarray, box: _Box) -> None:
        """Blur the glow layer around what was drawn into it, upscale it and add it to `layer`."""
        if box.empty:
            return
        small_height, small_width = glow.shape[:2]
        pad = math.ceil(3.0 * GLOW_SIGMA) + GLOW_THICKNESS + ENDPOINT_GLOW_RADIUS
        x0 = max(0, int(box.x0 // GLOW_DOWNSCALE) - pad)
        y0 = max(0, int(box.y0 // GLOW_DOWNSCALE) - pad)
        x1 = min(small_width, int(box.x1 // GLOW_DOWNSCALE) + pad + 1)
        y1 = min(small_height, int(box.y1 // GLOW_DOWNSCALE) + pad + 1)
        if x0 >= x1 or y0 >= y1:
            return
        blurred = cv2.GaussianBlur(glow[y0:y1, x0:x1], (0, 0), GLOW_SIGMA)
        size = ((x1 - x0) * GLOW_DOWNSCALE, (y1 - y0) * GLOW_DOWNSCALE)
        upscaled = cv2.resize(blurred, size, interpolation=cv2.INTER_LINEAR)
        target = layer[y0 * GLOW_DOWNSCALE : y1 * GLOW_DOWNSCALE, x0 * GLOW_DOWNSCALE : x1 * GLOW_DOWNSCALE]
        height, width = target.shape[:2]
        target[:] = cv2.add(target, upscaled[:height, :width])

    # Hands and HUD

    def _draw_hand(self, frame: np.ndarray, hand: Hand) -> None:
        points = [_int_point(p) for p in hand.points]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], BONE_COLOR, 2, cv2.LINE_AA)
        for p in points:
            cv2.circle(frame, p, 4, JOINT_COLOR, -1, cv2.LINE_AA)
        if hand.pinching:
            cv2.circle(frame, _int_point(hand.pinch_point), 9, PINCH_COLOR, 2, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, lines: list[str]) -> None:
        line_height = 22
        box_width = 12 + max((cv2.getTextSize(line, FONT, 0.5, 1)[0][0] for line in lines), default=0)
        box_height = 10 + line_height * len(lines)
        region = frame[8 : 8 + box_height, 8 : 8 + box_width]
        region[:] = (region * 0.45).astype(np.uint8)
        for index, line in enumerate(lines):
            origin = (14, 8 + line_height * (index + 1))
            cv2.putText(frame, line, origin, FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
