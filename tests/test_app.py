"""End-to-end run of the main loop with a scripted camera, tracker and audio engine."""

import numpy as np

from air_strings.app import AirStringsApp
from air_strings.hands import INDEX_TIP, THUMB_TIP, Hand
from air_strings.render import Renderer
from air_strings.scales import SCALES, PitchMapping
from air_strings.strings import DrawGesture, StringSet


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
    """Frames 0-9: both hands pinch and stretch a string from x=300 to x=1000 at y=360.
    Frames 10-24: hands open and rest. Frame 25: hand 0 flicks down across the string."""

    def __init__(self):
        self.frame = 0

    def process(self, frame, now):
        f = self.frame
        self.frame += 1
        if f < 10:
            return [hand(0, (300.0, 360.0), True), hand(1, (1000.0, 360.0), True)]
        if f < 25:
            return [hand(0, (600.0, 300.0), False, (600.0, 300.0))]
        if f == 25:
            return [hand(0, (600.0, 420.0), False, (600.0, 300.0))]
        return [hand(0, (600.0, 420.0), False, (600.0, 420.0))]


class RecordingAudio:
    def __init__(self):
        self.plucks = []

    def pluck(self, string_id, frequency, velocity, position):
        self.plucks.append((string_id, frequency, velocity, position))


def test_draw_commit_and_pluck_reach_the_audio_engine(monkeypatch):
    clock = iter(np.arange(0.0, 10.0, 1 / 30))
    monkeypatch.setattr("air_strings.app.time.monotonic", lambda: float(next(clock)))
    strings = StringSet(PitchMapping(), SCALES[0])
    audio = RecordingAudio()
    app = AirStringsApp(
        capture=FakeCapture(),
        tracker=ScriptedTracker(),
        strings=strings,
        gesture=DrawGesture(),
        renderer=Renderer(),
        audio=audio,
        max_frames=40,
        headless=True,
    )

    assert app.run() == 0
    assert app.frames_processed == 40
    [string] = strings.strings
    assert string.start == (300.0, 360.0) and string.end == (1000.0, 360.0)
    [(string_id, frequency, velocity, position)] = audio.plucks
    assert string_id == string.string_id
    assert frequency == string.frequency
    assert 0.0 < velocity <= 1.0
    assert abs(position - 300.0 / 700.0) < 1e-9
