"""Measure how long one frame takes to draw at 1280x720 in a busy scene.

Six strings (three vibrating as glowing ghost strands, one caught on a fingertip,
all with their rulers), a full particle system, a bright ruler over a string being
drawn and two hands.

Run with: uv run python scripts/bench_render.py
"""

import math
import time

import numpy as np

from air_strings.gestures import Bend, GestureView, Stretch
from air_strings.hands import INDEX_TIP, THUMB_TIP, Hand
from air_strings.particles import MAX_PARTICLES, ParticleSystem
from air_strings.render import Renderer
from air_strings.scales import SCALES, PitchMapping
from air_strings.strings import StringSet

WIDTH, HEIGHT = 1280, 720


def open_hand(hand_id: int, tip: tuple[float, float]) -> Hand:
    rng = np.random.default_rng(hand_id)
    points = np.asarray(tip) + rng.normal(0.0, 40.0, (21, 2))
    points[INDEX_TIP] = tip
    points[THUMB_TIP] = (tip[0] - 60.0, tip[1] + 40.0)
    return Hand(hand_id, points, False, tip)


def main() -> None:
    diagonal = math.hypot(WIDTH, HEIGHT)
    mapping = PitchMapping()
    strings = StringSet(mapping, SCALES[0])
    for i in range(6):
        strings.add((140.0 + 20 * i, 140.0 + 80 * i), (1000.0 - 60 * i, 150.0 + 80 * i), diagonal, now=0.0)
    for string in strings.strings[:3]:
        string.pluck(1.0, 0.35, 1.0, displacement=0.12 * string.length)

    # Sparks of different ages and tints (one emission per 40 ms, from different strings).
    particles = ParticleSystem(rng=np.random.default_rng(0))
    for i in range(14):
        string = strings.strings[i % 6]
        tint = tuple(0.8 * 255.0 + 0.2 * c for c in string.color)
        particles.emit(string.start, string.end, 0.4, 1.0, tint, displacement=40.0)
        particles.update(0.04)
    assert particles.count == MAX_PARTICLES

    caught = strings.strings[4]
    contact = (caught.start[0] * 0.6 + caught.end[0] * 0.4, caught.start[1] * 0.6 + caught.end[1] * 0.4 + 40.0)
    view = GestureView(
        pending=((300.0, 620.0), (1050.0, 640.0)),
        stretches=(Stretch((300.0, 620.0), (1050.0, 640.0), strings.note_for_length(750.0, diagonal)),),
        bends=(Bend(caught, 0.4, contact),),
    )
    hands = [open_hand(0, (500.0, 400.0)), open_hand(1, (900.0, 300.0))]

    renderer = Renderer()
    background = np.random.default_rng(1).integers(0, 120, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame = background.copy()
    timings = []
    for index in range(300):
        np.copyto(frame, background)
        now = 1.05 + index / 30 * 0.01  # keep the strings ringing
        start = time.perf_counter()
        renderer.draw_frame(frame, hands, strings.strings, view, particles, now)
        timings.append(time.perf_counter() - start)
    timings_ms = np.array(timings[20:]) * 1000.0
    vibrating = sum(s.is_vibrating(1.05) for s in strings.strings)
    print(f"scene: {WIDTH}x{HEIGHT}, {len(strings.strings)} strings ({vibrating} vibrating), "
          f"{particles.count} particles, {len(view.stretches)} ruler, {len(view.bends)} caught")
    print(
        f"render time: median {np.median(timings_ms):.2f} ms, "
        f"p99 {np.percentile(timings_ms, 99):.2f} ms, max {timings_ms.max():.2f} ms"
    )


if __name__ == "__main__":
    main()
