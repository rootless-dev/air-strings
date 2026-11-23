"""Application composition and lifecycle, and the headless loop."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import cv2

from air_strings.camera import CameraSession
from air_strings.demo import render_demo
from air_strings.engine import Engine
from air_strings.gestures import DrawGesture, GestureController, GrabGesture, PullGesture
from air_strings.hands import DEFAULT_MODEL_PATH, HandTracker, ensure_model
from air_strings.particles import ParticleSystem
from air_strings.pipeline import CameraStoppedError
from air_strings.render import Renderer
from air_strings.scales import SCALES, PitchMapping
from air_strings.strings import StringSet
from air_strings.synth import AudioEngine, VoicePool

# Longest wait of the headless loop for a new detection, so it notices a closed mailbox.
DETECTION_WAIT_S = 0.05

# Runs the user interface until it is closed: (engine, camera session, max frames).
UserInterface = Callable[[Engine, CameraSession, int], None]


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
    """Starts the camera session, runs the UI (or the headless loop) and shuts everything down.

    With `ui` None the app runs headless: no window, frames are stepped and dropped.
    `max_frames` > 0 ends the run after that many displayed frames.
    """

    def __init__(
        self,
        session: CameraSession,
        engine: Engine,
        max_frames: int = 0,
        ui: UserInterface | None = None,
    ) -> None:
        self._session = session
        self._engine = engine
        self._max_frames = max_frames
        self._ui = ui

    @property
    def frames_processed(self) -> int:
        return self._engine.frames

    def run(self) -> int:
        self._session.start()
        try:
            if self._ui is None:
                self._run_headless()
            else:
                self._ui(self._engine, self._session, self._max_frames)
        finally:
            self._session.stop()
        if self._max_frames > 0:
            print(self._engine.meter.summary(), file=sys.stderr)
        return self._exit_status()

    def _run_headless(self) -> None:
        while self._max_frames <= 0 or self._engine.frames < self._max_frames:
            mailbox = self._session.mailbox
            detection = mailbox.take(DETECTION_WAIT_S)
            if detection is not None:
                self._engine.step(detection)
            elif mailbox.closed:
                return

    def _exit_status(self) -> int:
        """0, or why the detection thread ended on its own: 1 for a lost camera, else its exception."""
        error = self._session.worker.error
        if isinstance(error, CameraStoppedError):
            print("air-strings: camera stopped delivering frames", file=sys.stderr)
            return 1
        if error is not None:
            raise error
        return 0


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture | None:
    """Camera `index` at the requested size, or None if it cannot be opened."""
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.render_demo is not None:
        path = render_demo(args.render_demo)
        print(f"air-strings: demo written to {path}")
        return 0

    model_path = ensure_model(args.model)

    def capture_factory(index: int) -> cv2.VideoCapture | None:
        return open_camera(index, args.width, args.height)

    capture = capture_factory(args.camera)
    if capture is None:
        print(
            f"air-strings: cannot open camera {args.camera} "
            "(on macOS, allow camera access for your terminal app)",
            file=sys.stderr,
        )
        return 1

    session = CameraSession(args.camera, capture, capture_factory, lambda: HandTracker(model_path))
    audio = AudioEngine(VoicePool())
    strings = StringSet(PitchMapping(), SCALES[0])
    engine = Engine(
        strings=strings,
        gestures=GestureController(strings, DrawGesture(), GrabGesture(), PullGesture()),
        particles=ParticleSystem(),
        renderer=Renderer(),
        audio=audio,
    )
    ui: UserInterface | None = None
    if not args.headless:
        from air_strings.ui.main_window import run_window  # Qt is only loaded with a window

        ui = run_window
    app = AirStringsApp(session, engine, max_frames=args.frames, ui=ui)
    audio.start()
    try:
        return app.run()
    except KeyboardInterrupt:
        return 0
    finally:
        audio.stop()
