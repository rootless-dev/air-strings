"""End-to-end runs of the main loop with a scripted camera, tracker and audio engine."""

import math

import numpy as np
import pytest

from air_strings.app import AirStringsApp
from air_strings.gestures import PULL_THRESHOLD_VELOCITY, DrawGesture, GestureController, GrabGesture, PullGesture
from air_strings.hands import INDEX_TIP, THUMB_TIP, Hand
from air_strings.particles import ParticleSystem
from air_strings.render import Renderer
from air_strings.scales import SCALES, PitchMapping
from air_strings.strings import StringSet

DIAGONAL = math.hypot(1280, 720)


class FakeCapture:
    def read(self):
        return True, np.zeros((720, 1280, 3), dtype=np.uint8)


def hand(hand_id, tip, pinching, previous_tip=None):
    points = np.zeros((21, 2))
    points[:] = tip
    points[INDEX_TIP] = tip
    points[THUMB_TIP] = tip
    return Hand(hand_id, points, pinching, previous_tip)


class ScriptedTracker:
    """Plays a script: one list of (hand_id, tip, pinching) per frame; previous tips are filled in."""

    def __init__(self, script):
        self._script = script
        self._previous = {}
        self.frame = 0

    def process(self, frame, now):
        entries = self._script[min(self.frame, len(self._script) - 1)]
        self.frame += 1
        hands = [hand(hand_id, tip, pinching, self._previous.get(hand_id)) for hand_id, tip, pinching in entries]
        self._previous = {hand_id: tip for hand_id, tip, _ in entries}
        return hands


def draw_string_script(left, right):
    """Frames 0-9: both hands pinch and hold a string from `left` to `right`."""
    return [[(0, left, True), (1, right, True)] for _ in range(10)]


class RecordingAudio:
    def __init__(self):
        self.plucks = []

    def pluck(self, string_id, frequency, velocity, position):
        self.plucks.append((string_id, frequency, velocity, position))


class RecordingRenderer(Renderer):
    def __init__(self, mapping):
        super().__init__(mapping)
        self.views = []

    def draw_frame(self, frame, hands, strings, gestures, particles, scale, now, hud_lines):
        self.views.append(gestures)
        super().draw_frame(frame, hands, strings, gestures, particles, scale, now, hud_lines)


def make_app(monkeypatch, script, frames):
    clock = iter(np.arange(0.0, 20.0, 1 / 30))
    monkeypatch.setattr("air_strings.app.time.monotonic", lambda: float(next(clock)))
    mapping = PitchMapping()
    strings = StringSet(mapping, SCALES[0])
    audio = RecordingAudio()
    particles = ParticleSystem(rng=np.random.default_rng(0))
    renderer = RecordingRenderer(mapping)
    app = AirStringsApp(
        capture=FakeCapture(),
        tracker=ScriptedTracker(script),
        strings=strings,
        gestures=GestureController(strings, DrawGesture(), GrabGesture(), PullGesture()),
        particles=particles,
        renderer=renderer,
        audio=audio,
        max_frames=frames,
        headless=True,
    )
    return app, strings, audio, particles, renderer


def test_draw_commit_and_swipe_reach_the_audio_engine(monkeypatch):
    script = draw_string_script((300.0, 360.0), (1000.0, 360.0))
    script += [[(0, (600.0, 300.0), False)] for _ in range(15)]  # frames 10-24: open hand rests
    script += [[(0, (600.0, 460.0), False)]]  # frame 25: one fast swipe down through the string
    app, strings, audio, particles, _ = make_app(monkeypatch, script, frames=40)

    assert app.run() == 0
    assert app.frames_processed == 40
    [string] = strings.strings
    assert string.start == (300.0, 360.0) and string.end == (1000.0, 360.0)
    [(string_id, frequency, velocity, position)] = audio.plucks
    assert string_id == string.string_id
    assert frequency == string.frequency
    assert 0.0 < velocity <= 1.0
    assert abs(position - 300.0 / 700.0) < 1e-9


def test_grab_resize_then_pull_and_release(monkeypatch):
    script = draw_string_script((300.0, 360.0), (1000.0, 360.0))
    script += [[(0, (600.0, 200.0), False)] for _ in range(10)]  # frames 10-19: commit, rest
    # Frames 20-29: one hand pinches the right end and drags it to x = 800.
    script += [[(0, (float(x), 360.0), True)] for x in np.linspace(1000.0, 800.0, 10)]
    script += [[(0, (500.0, 300.0), False)] for _ in range(15)]  # frames 30-44: release, wait out the grace
    # Frames 45-50: the open fingertip pushes down through the string (500 px: threshold 50 px).
    script += [[(0, (500.0, y), False)] for y in (340.0, 355.0, 370.0, 385.0, 400.0, 415.0)]
    script += [[(0, (500.0, 415.0), False)]]
    app, strings, audio, particles, renderer = make_app(monkeypatch, script, frames=60)

    assert app.run() == 0
    [string] = strings.strings
    assert string.start == (300.0, 360.0)
    assert string.end == pytest.approx((800.0, 360.0))
    assert string.midi == strings.note_for_length(500.0, DIAGONAL)

    resize_views = [v for v in renderer.views[20:30] if v.stretches]
    assert len(resize_views) == 10
    assert all(v.stretches[0].string_id == string.string_id for v in resize_views)
    assert resize_views[-1].stretches[0].anchor == (300.0, 360.0)
    assert all(not v.stretches for v in renderer.views[32:])  # the ruler is gone after release

    bent_frames = [i for i, v in enumerate(renderer.views) if v.bends]
    assert bent_frames == [47, 48, 49]  # caught at y = 370, released at y = 415
    [(string_id, frequency, velocity, position)] = audio.plucks
    assert string_id == string.string_id
    assert frequency == string.frequency
    assert velocity == pytest.approx(PULL_THRESHOLD_VELOCITY * 55.0 / 50.0)
    assert position == pytest.approx(200.0 / 500.0)
    assert particles.count > 0
