"""Application composition, main loop and lifecycle."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2

from air_strings.demo import render_demo
from air_strings.geometry import distance
from air_strings.hands import DEFAULT_MODEL_PATH, HandTracker, ensure_model
from air_strings.render import Renderer
from air_strings.scales import SCALES, PitchMapping, next_scale
from air_strings.strings import MAX_STRINGS, DrawGesture, StringSet
from air_strings.synth import AudioEngine, VoicePool

WINDOW_NAME = "air-strings"
# Fingertip speed, in frame diagonals per second, that counts as a full-force pluck.
FULL_VELOCITY_DIAGONALS_PER_S = 2.0
MAX_FAILED_READS = 30


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
        gesture: DrawGesture,
        renderer: Renderer,
        audio: AudioEngine,
        max_frames: int = 0,
        headless: bool = False,
    ) -> None:
        self._capture = capture
        self._tracker = tracker
        self._strings = strings
        self._gesture = gesture
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

        committed = self._gesture.update([h.pinch_point for h in hands if h.pinching])
        if committed is not None:
            self._strings.add(committed[0], committed[1], diagonal, now)

        if dt > 0:
            for hand in hands:
                if hand.pinching or hand.previous_tip is None:
                    continue
                plucks = self._strings.find_plucks(hand.previous_tip, hand.tip, now)
                if not plucks:
                    continue
                speed = distance(hand.previous_tip, hand.tip) / dt / diagonal
                velocity = min(1.0, speed / FULL_VELOCITY_DIAGONALS_PER_S)
                for pluck in plucks:
                    pluck.string.pluck(now, pluck.position, velocity)
                    self._audio.pluck(pluck.string.string_id, pluck.string.frequency, velocity, pluck.position)

        hud = [
            f"Scale: {self._strings.scale.name}",
            f"FPS: {self._fps:4.1f}   strings: {len(self._strings.strings)}/{MAX_STRINGS}",
            "Pinch with both hands, pull apart, release: new string",
            "Flick your index finger across a string to pluck",
            "c: clear   s: scale   q/Esc: quit",
        ]
        self._renderer.draw_frame(frame, hands, self._strings.strings, self._gesture.pending, now, hud)

    def _handle_key(self, key: int) -> bool:
        """Returns False when the app should quit."""
        if key in (ord("q"), 27):
            return False
        if key == ord("c"):
            self._strings.clear()
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
    app = AirStringsApp(
        capture=capture,
        tracker=tracker,
        strings=StringSet(PitchMapping(), SCALES[0]),
        gesture=DrawGesture(),
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
