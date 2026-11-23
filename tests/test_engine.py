"""The shared per-frame step's bookkeeping: rates, timings and the metrics it reports."""

import math

import numpy as np
import pytest

from air_strings.engine import Engine, FrameMeter
from air_strings.gestures import DrawGesture, GestureController, GrabGesture, PullGesture
from air_strings.particles import ParticleSystem
from air_strings.pipeline import Detection
from air_strings.render import Renderer
from air_strings.scales import SCALES, PitchMapping
from air_strings.strings import StringSet


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class SilentAudio:
    active_voices = 4

    def pluck(self, string_id, frequency, velocity, position):
        pass


def make_engine(meter=None):
    strings = StringSet(PitchMapping(), SCALES[0])
    engine = Engine(
        strings=strings,
        gestures=GestureController(strings, DrawGesture(), GrabGesture(), PullGesture()),
        particles=ParticleSystem(rng=np.random.default_rng(0)),
        renderer=Renderer(),
        audio=SilentAudio(),
        meter=meter,
    )
    return engine, strings


def test_meter_rates_and_timings_from_frames_and_sequence_numbers():
    clock = FakeClock()
    meter = FrameMeter(clock=clock, smoothing=1.0)  # no smoothing: the last sample
    for frame, sequence in enumerate((1, 3, 5)):  # two detections per displayed frame
        clock.now = frame * 0.05
        meter.record(sequence, detection_time=0.02, render_time=0.004)

    assert meter.fps == pytest.approx(20.0)
    assert meter.detection_fps == pytest.approx(40.0)
    assert meter.detection_ms == pytest.approx(20.0)
    assert meter.render_ms == pytest.approx(4.0)
    assert meter.summary() == (
        "air-strings: frames 3 in 0.10 s = 20.0 fps, detections 40.0 fps, detect 20.0 ms, render 4.0 ms"
    )


def test_meter_counts_a_restarted_sequence_after_a_camera_switch():
    clock = FakeClock()
    meter = FrameMeter(clock=clock, smoothing=1.0)
    for frame, sequence in enumerate((9, 10, 1, 2)):  # a new worker starts again at 1
        clock.now = frame * 0.1
        meter.record(sequence, 0.0, 0.0)

    assert meter.detection_fps == pytest.approx(10.0)
    assert "detections 10.0 fps" in meter.summary()


def test_meter_without_frames():
    assert FrameMeter().summary() == "air-strings: no frames displayed"


def test_engine_steps_draw_and_report_metrics():
    engine, strings = make_engine()
    strings.add((100.0, 100.0), (600.0, 100.0), math.hypot(1280, 720), now=0.0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    engine.step(Detection(frame, [], 1.0, 0.0, 1, detection_time=0.015))

    assert engine.frames == 1
    assert frame.any()  # the string was drawn onto the frame
    metrics = engine.metrics()
    assert (metrics.strings, metrics.max_strings, metrics.scale, metrics.voices) == (1, 6, "C major pentatonic", 4)
    assert metrics.detection_ms == pytest.approx(15.0)
    assert metrics.render_ms > 0.0


def test_engine_clear_and_cycle_scale():
    engine, strings = make_engine()
    strings.add((100.0, 100.0), (600.0, 100.0), math.hypot(1280, 720), now=0.0)

    engine.clear()
    assert strings.strings == []
    assert engine.cycle_scale() == SCALES[1]
    assert engine.scale == SCALES[1] == strings.scale
