"""The per-frame step shared by the window and the headless loop, and its metrics.

`Engine.step` takes one detection and does everything a displayed frame needs:
replays the hands through the gestures, triggers plucks (audio and sparks),
advances the particles and draws onto the frame in place. How the frame is then
shown (a Qt window, or nothing) is up to the caller.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from air_strings.gestures import GestureController, Pluck
from air_strings.particles import ParticleSystem
from air_strings.pipeline import Detection
from air_strings.render import Renderer
from air_strings.scales import Scale, next_scale
from air_strings.strings import StringSet

# Weight of the newest sample in the smoothed rates and timings.
RATE_SMOOTHING = 0.1
# Longest step the particles are advanced by, so a stalled frame does not fling them away.
MAX_PARTICLE_DT_S = 0.1
# Sparks are pale: mostly white, slightly tinted with the string's color.
SPARK_WHITENESS = 0.8


class PluckSink(Protocol):
    """The audio side as the engine sees it."""

    @property
    def active_voices(self) -> int: ...

    def pluck(self, string_id: int, frequency: float, velocity: float, position: float) -> None: ...


@dataclass(frozen=True)
class Metrics:
    """What the window's metrics bar shows."""

    fps: float  # displayed frames per second
    detection_fps: float  # detections per second (displayed or not)
    detection_ms: float  # hand tracker time per detection
    render_ms: float  # engine step time per displayed frame (gestures + drawing)
    strings: int
    max_strings: int
    scale: str
    voices: int  # audio voices sounding


class FrameMeter:
    """Smoothed display and detection rates and per-frame timings, plus a run summary.

    `record` is called once per displayed frame. Rates use `clock` (wall time on
    the consumer's side); the detection rate counts detection sequence numbers,
    so detections the display skipped count too. A sequence number that does not
    grow means a new detection worker (another camera) started counting from 1.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter, smoothing: float = RATE_SMOOTHING) -> None:
        self._clock = clock
        self._smoothing = smoothing
        self.fps = 0.0
        self.detection_fps = 0.0
        self.detection_ms = 0.0
        self.render_ms = 0.0
        self._last_at: float | None = None
        self._last_sequence = 0
        self._first_at: float | None = None
        self._frames = 0
        self._detections = 0  # detections since the first displayed frame
        self._detection_total_s = 0.0
        self._render_total_s = 0.0

    def record(self, sequence: int, detection_time: float, render_time: float) -> None:
        now = self._clock()
        if self._first_at is None:
            self._first_at = now
        elif self._last_at is not None:
            # A restarted worker counts from 1 again: its sequence number is its detection count.
            new = sequence - self._last_sequence if sequence > self._last_sequence else sequence
            self._detections += new
            elapsed = now - self._last_at
            if elapsed > 0:
                self.fps = self._smooth(self.fps, 1.0 / elapsed)
                self.detection_fps = self._smooth(self.detection_fps, new / elapsed)
        self.detection_ms = self._smooth(self.detection_ms, detection_time * 1000.0)
        self.render_ms = self._smooth(self.render_ms, render_time * 1000.0)
        self._last_at = now
        self._last_sequence = sequence
        self._frames += 1
        self._detection_total_s += detection_time
        self._render_total_s += render_time

    def _smooth(self, current: float, sample: float) -> float:
        if current == 0.0:
            return sample
        return current + self._smoothing * (sample - current)

    def summary(self) -> str:
        """End-of-run rates for bounded runs, measured from the first displayed frame."""
        if self._first_at is None or self._last_at is None:
            return "air-strings: no frames displayed"
        elapsed = self._last_at - self._first_at
        if elapsed <= 0:
            return f"air-strings: frames {self._frames}"
        frames = self._frames - 1
        return (
            f"air-strings: frames {self._frames} in {elapsed:.2f} s = {frames / elapsed:.1f} fps, "
            f"detections {self._detections / elapsed:.1f} fps, "
            f"detect {1000.0 * self._detection_total_s / self._frames:.1f} ms, "
            f"render {1000.0 * self._render_total_s / self._frames:.1f} ms"
        )


class Engine:
    """Gestures, plucks, particles and drawing for each displayed frame."""

    def __init__(
        self,
        strings: StringSet,
        gestures: GestureController,
        particles: ParticleSystem,
        renderer: Renderer,
        audio: PluckSink,
        meter: FrameMeter | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._strings = strings
        self._gestures = gestures
        self._particles = particles
        self._renderer = renderer
        self._audio = audio
        self._meter = meter or FrameMeter()
        self._clock = clock
        self._last_timestamp: float | None = None
        self.frames = 0  # displayed frames stepped so far

    @property
    def scale(self) -> Scale:
        return self._strings.scale

    @property
    def meter(self) -> FrameMeter:
        return self._meter

    def step(self, detection: Detection) -> None:
        """Update strings and plucks from the detected hands, and draw onto the frame in place."""
        started = self._clock()
        frame, now = detection.frame, detection.timestamp
        height, width = frame.shape[:2]
        diagonal = math.hypot(width, height)
        # Time since the last displayed frame; it differs from `detection.dt` (time since the
        # previous detection, which the fingertip speed is measured over) when one was skipped.
        dt = now - self._last_timestamp if self._last_timestamp is not None else 0.0
        self._last_timestamp = now

        # Detections the display skipped still go through the gestures, in order: each
        # hand's previous fingertip is from the detection before it, so a swipe that
        # crossed a string in a skipped one would otherwise never be seen.
        for sample in (*detection.skipped, detection):
            for pluck in self._gestures.update(sample.hands, sample.timestamp, sample.dt, diagonal):
                self._on_pluck(pluck)
        self._particles.update(min(dt, MAX_PARTICLE_DT_S))

        self._renderer.draw_frame(
            frame,
            detection.hands,
            self._strings.strings,
            self._gestures.view(),
            self._particles,
            now,
        )
        self.frames += 1
        self._meter.record(detection.sequence, detection.detection_time, self._clock() - started)

    def clear(self) -> None:
        """Remove every string and spark."""
        self._strings.clear()
        self._particles.clear()

    def cycle_scale(self) -> Scale:
        """Switch to the next scale (retuning the strings) and return it."""
        scale = next_scale(self._strings.scale)
        self._strings.set_scale(scale)
        return scale

    def metrics(self) -> Metrics:
        meter = self._meter
        return Metrics(
            fps=meter.fps,
            detection_fps=meter.detection_fps,
            detection_ms=meter.detection_ms,
            render_ms=meter.render_ms,
            strings=len(self._strings.strings),
            max_strings=self._strings.max_strings,
            scale=self._strings.scale.name,
            voices=self._audio.active_voices,
        )

    def _on_pluck(self, pluck: Pluck) -> None:
        string = pluck.string
        self._audio.pluck(string.string_id, string.frequency, pluck.velocity, pluck.position)
        tint = tuple(SPARK_WHITENESS * 255.0 + (1.0 - SPARK_WHITENESS) * c for c in string.color)
        self._particles.emit(string.start, string.end, pluck.position, pluck.velocity, tint, pluck.displacement)
