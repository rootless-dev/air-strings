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
from air_strings.render import Renderer
from air_strings.scales import SCALES, PitchMapping, next_scale
from air_strings.strings import MAX_STRINGS, StringSet
from air_strings.synth import AudioEngine, VoicePool

WINDOW_NAME = "air-strings"
MAX_FAILED_READS = 30
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
    def __init__(
        self,
        capture: cv2.VideoCapture,
        tracker: HandTracker,
        strings: StringSet,
        gestures: GestureController,
        particles: ParticleSystem,
        renderer: Renderer,
        audio: AudioEngine,
        max_frames: int = 0,
        headless: bool = False,
    ) -> None:
        self._capture = capture
        self._tracker = tracker
        self._strings = strings
        self._gestures = gestures
        self._particles = particles
        self._renderer = renderer
        self._audio = audio
        self._max_frames = max_frames
        self._headless = headless
        self._fps = 0.0
        self._last_frame_at: float | None = None
        self.frames_processed = 0

    def run(self) -> int:
        failed_reads = 0
        while self._max_frames <= 0 or self.frames_processed < self._max_frames:
            ok, frame = self._capture.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= MAX_FAILED_READS:
                    print("air-strings: camera stopped delivering frames", file=sys.stderr)
                    return 1
                continue
            failed_reads = 0
            frame = cv2.flip(frame, 1)  # mirror, like a selfie view
            self._step(frame, time.monotonic())
            self.frames_processed += 1
            if not self._headless:
                cv2.imshow(WINDOW_NAME, frame)
                if not self._handle_key(cv2.waitKey(1) & 0xFF):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        return 0

    def _step(self, frame, now: float) -> None:
        """Track hands, update strings and plucks, and draw onto `frame` in place."""
        height, width = frame.shape[:2]
        diagonal = math.hypot(width, height)
        dt = now - self._last_frame_at if self._last_frame_at is not None else 0.0
        self._last_frame_at = now
        if dt > 0:
            self._fps = 1.0 / dt if self._fps == 0 else 0.9 * self._fps + 0.1 / dt

        hands = self._tracker.process(frame, now)
        for pluck in self._gestures.update(hands, now, dt, diagonal):
            self._on_pluck(pluck)
        self._particles.update(min(dt, MAX_PARTICLE_DT_S))

        hud = [
            f"Scale: {self._strings.scale.name}",
            f"FPS: {self._fps:4.1f}   strings: {len(self._strings.strings)}/{MAX_STRINGS}",
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
    audio = AudioEngine(VoicePool())
    mapping = PitchMapping()
    strings = StringSet(mapping, SCALES[0])
    app = AirStringsApp(
        capture=capture,
        tracker=tracker,
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
        tracker.close()
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
