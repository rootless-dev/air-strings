"""Application composition, main loop and lifecycle."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2

from air_strings.demo import render_demo
from air_strings.gestures import DrawGesture, GestureController, GrabGesture, Pluck, PullGesture
from air_strings.hands import DEFAULT_MODEL_PATH, HandTracker, ensure_model
from air_strings.particles import ParticleSystem
from air_strings.pipeline import CameraStoppedError, Detection, DetectionWorker, LatestDetection
from air_strings.render import Renderer
from air_strings.scales import SCALES, PitchMapping, next_scale
from air_strings.strings import MAX_STRINGS, StringSet
from air_strings.synth import AudioEngine, VoicePool

WINDOW_NAME = "air-strings"
# Longest wait for a new detection before pumping the window events anyway, so keys
# and window closing still work when the camera is slow or stalls.
DETECTION_WAIT_S = 0.05
# Weight of the newest sample in the smoothed HUD rates.
RATE_SMOOTHING = 0.1
# Longest step the particles are advanced by, so a stalled frame does not fling them away.
MAX_PARTICLE_DT_S = 0.1
# Sparks are pale: mostly white, slightly tinted with the string's color.
SPARK_WHITENESS = 0.8


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="air-strings", description="Draw guitar strings in the air and pluck them.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--width", type=int, default=1280, help="capture width (default 1280)")
    parser.add_argument("--height", type=int, default=720, help="capture height (default 720)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="hand_landmarker.task path")
    parser.add_argument("--render-demo", type=Path, metavar="OUT.wav", help="render an audio demo to a WAV file and exit")
    # Hidden flags for bounded, non-interactive smoke runs.
    parser.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


class AirStringsApp:
    """Main thread: gestures, drawing and the window. Capture and detection run in `worker`."""

    def __init__(
        self,
        worker: DetectionWorker,
        strings: StringSet,
        gestures: GestureController,
        particles: ParticleSystem,
        renderer: Renderer,
        audio: AudioEngine,
        max_frames: int = 0,
        headless: bool = False,
    ) -> None:
        self._worker = worker
        self._strings = strings
        self._gestures = gestures
        self._particles = particles
        self._renderer = renderer
        self._audio = audio
        self._max_frames = max_frames
        self._headless = headless
        self._window_shown = False
        self._last_timestamp: float | None = None
        # Displayed-frame and detection rates, measured on the main thread's wall clock.
        self._fps = 0.0
        self._detection_fps = 0.0
        self._last_display_at: float | None = None
        self._last_sequence = 0
        self._first_display: tuple[float, int] | None = None  # (wall time, sequence)
        self.frames_processed = 0

    def run(self) -> int:
        self._worker.start()
        try:
            status = self._loop()
        finally:
            self._worker.stop()
        if self._max_frames > 0:
            self._print_summary()
        return status

    def _loop(self) -> int:
        mailbox = self._worker.mailbox
        while self._max_frames <= 0 or self.frames_processed < self._max_frames:
            detection = mailbox.take(DETECTION_WAIT_S)
            if detection is not None:
                self._step(detection)
                self.frames_processed += 1
            elif mailbox.closed:
                return self._worker_ended()
            if not self._headless:
                if detection is not None:
                    cv2.imshow(WINDOW_NAME, detection.frame)
                    self._window_shown = True
                if not self._handle_key(cv2.waitKey(1) & 0xFF):
                    break
                if self._window_shown and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        return 0

    def _worker_ended(self) -> int:
        """The detection thread ended on its own: report why."""
        error = self._worker.error
        if isinstance(error, CameraStoppedError):
            print("air-strings: camera stopped delivering frames", file=sys.stderr)
            return 1
        if error is not None:
            raise error
        return 0

    def _step(self, detection: Detection) -> None:
        """Update strings and plucks from the detected hands, and draw onto the frame in place."""
        frame, hands, now = detection.frame, detection.hands, detection.timestamp
        height, width = frame.shape[:2]
        diagonal = math.hypot(width, height)
        # Time since the last displayed frame; it differs from `detection.dt` (time since the
        # previous detection, which the fingertip speed is measured over) when one was skipped.
        dt = now - self._last_timestamp if self._last_timestamp is not None else 0.0
        self._last_timestamp = now
        self._update_rates(detection.sequence)

        # Detections the display skipped still go through the gestures, in order: each
        # hand's previous fingertip is from the detection before it, so a swipe that
        # crossed a string in a skipped one would otherwise never be seen.
        for sample in (*detection.skipped, detection):
            for pluck in self._gestures.update(sample.hands, sample.timestamp, sample.dt, diagonal):
                self._on_pluck(pluck)
        self._particles.update(min(dt, MAX_PARTICLE_DT_S))

        hud = [
            f"Scale: {self._strings.scale.name}",
            f"FPS: {self._fps:4.1f} | det {self._detection_fps:4.1f}   "
            f"strings: {len(self._strings.strings)}/{MAX_STRINGS}",
            "Pinch with both hands, pull apart, release: new string",
            "Pinch one end of a string with one hand and drag: resize",
            "Push your index finger through a string until it slips: pluck",
            "c: clear   s: scale   q/Esc: quit",
        ]
        self._renderer.draw_frame(
            frame,
            hands,
            self._strings.strings,
            self._gestures.view(),
            self._particles,
            now,
            hud,
        )

    def _update_rates(self, sequence: int) -> None:
        now = time.perf_counter()
        if self._first_display is None:
            self._first_display = (now, sequence)
        if self._last_display_at is not None:
            elapsed = now - self._last_display_at
            if elapsed > 0:
                fps = 1.0 / elapsed
                detection_fps = (sequence - self._last_sequence) / elapsed
                if self._fps == 0:
                    self._fps, self._detection_fps = fps, detection_fps
                else:
                    self._fps += RATE_SMOOTHING * (fps - self._fps)
                    self._detection_fps += RATE_SMOOTHING * (detection_fps - self._detection_fps)
        self._last_display_at = now
        self._last_sequence = sequence

    def _print_summary(self) -> None:
        """End-of-run rates for bounded runs (`--frames`), measured from the first displayed frame."""
        if self._first_display is None or self._last_display_at is None:
            print("air-strings: no frames displayed", file=sys.stderr)
            return
        start, first_sequence = self._first_display
        elapsed = self._last_display_at - start
        frames = self.frames_processed - 1
        detections = self._last_sequence - first_sequence
        if elapsed <= 0:
            print(f"air-strings: frames {self.frames_processed}", file=sys.stderr)
            return
        print(
            f"air-strings: frames {self.frames_processed} in {elapsed:.2f} s = {frames / elapsed:.1f} fps, "
            f"detections {detections / elapsed:.1f} fps",
            file=sys.stderr,
        )

    def _on_pluck(self, pluck: Pluck) -> None:
        string = pluck.string
        self._audio.pluck(string.string_id, string.frequency, pluck.velocity, pluck.position)
        tint = tuple(SPARK_WHITENESS * 255.0 + (1.0 - SPARK_WHITENESS) * c for c in string.color)
        self._particles.emit(string.start, string.end, pluck.position, pluck.velocity, tint, pluck.displacement)

    def _handle_key(self, key: int) -> bool:
        """Returns False when the app should quit."""
        if key in (ord("q"), 27):
            return False
        if key == ord("c"):
            self._strings.clear()
            self._particles.clear()
        elif key == ord("s"):
            self._strings.set_scale(next_scale(self._strings.scale))
        return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.render_demo is not None:
        path = render_demo(args.render_demo)
        print(f"air-strings: demo written to {path}")
        return 0

    model_path = ensure_model(args.model)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(
            f"air-strings: cannot open camera {args.camera} "
            "(on macOS, allow camera access for your terminal app)",
            file=sys.stderr,
        )
        return 1
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    tracker = HandTracker(model_path)
    worker = DetectionWorker(capture, tracker, LatestDetection())
    audio = AudioEngine(VoicePool())
    mapping = PitchMapping()
    strings = StringSet(mapping, SCALES[0])
    app = AirStringsApp(
        worker=worker,
        strings=strings,
        gestures=GestureController(strings, DrawGesture(), GrabGesture(), PullGesture()),
        particles=ParticleSystem(),
        renderer=Renderer(),
        audio=audio,
        max_frames=args.frames,
        headless=args.headless,
    )
    audio.start()
    try:
        return app.run()
    except KeyboardInterrupt:
        return 0
    finally:
        audio.stop()
        # `run` has joined the worker; if it is somehow still mid-read, leave the camera and
        # the tracker to it rather than freeing them under its feet.
        if not worker.alive:
            tracker.close()
            capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
