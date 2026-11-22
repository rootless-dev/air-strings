"""Hand tracking: MediaPipe HandLandmarker wrapper, landmark smoothing and pinch detection."""

from __future__ import annotations

import math
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9

# Max wrist jump (as a fraction of the frame diagonal) to keep a hand's identity.
IDENTITY_MATCH_FRACTION = 0.25


def ensure_model(path: Path = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Download the hand landmarker model on first use."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"air-strings: downloading hand model to {path} ...", file=sys.stderr)
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)
    return path


class OneEuroFilter:
    """One-Euro filter over arrays of 2D points (shape (..., 2)).

    Low cutoff when a point is still (kills jitter), higher cutoff as it moves
    fast (keeps latency low). The cutoff adapts per point, from its speed.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.008, d_cutoff: float = 1.0) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._value: np.ndarray | None = None
        self._derivative: np.ndarray | None = None
        self._time: float | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: np.ndarray, t: float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self._value is None or self._time is None or t <= self._time:
            self._value = value.copy()
            self._derivative = np.zeros_like(value)
            self._time = t
            return self._value.copy()
        dt = t - self._time
        raw_derivative = (value - self._value) / dt
        a_d = self._alpha(self._d_cutoff, dt)
        self._derivative = a_d * raw_derivative + (1.0 - a_d) * self._derivative
        speed = np.linalg.norm(self._derivative, axis=-1, keepdims=True)
        a = self._alpha(self._min_cutoff + self._beta * speed, dt)
        self._value = a * value + (1.0 - a) * self._value
        self._time = t
        return self._value.copy()


class PinchDetector:
    """Thumb-tip to index-tip distance, normalized by hand size, with hysteresis."""

    def __init__(self, on_ratio: float = 0.28, off_ratio: float = 0.42) -> None:
        self._on_ratio = on_ratio
        self._off_ratio = off_ratio
        self.pinching = False

    @staticmethod
    def ratio(points: np.ndarray) -> float:
        hand_size = float(np.linalg.norm(points[MIDDLE_MCP] - points[WRIST]))
        gap = float(np.linalg.norm(points[INDEX_TIP] - points[THUMB_TIP]))
        return gap / max(hand_size, 1e-6)

    def update(self, points: np.ndarray) -> bool:
        ratio = self.ratio(points)
        if self.pinching:
            self.pinching = ratio < self._off_ratio
        else:
            self.pinching = ratio < self._on_ratio
        return self.pinching


@dataclass(frozen=True)
class Hand:
    hand_id: int
    points: np.ndarray  # (21, 2) smoothed pixel coordinates
    pinching: bool
    previous_tip: tuple[float, float] | None  # index fingertip in the previous frame

    @property
    def tip(self) -> tuple[float, float]:
        return (float(self.points[INDEX_TIP, 0]), float(self.points[INDEX_TIP, 1]))

    @property
    def pinch_point(self) -> tuple[float, float]:
        mid = (self.points[THUMB_TIP] + self.points[INDEX_TIP]) * 0.5
        return (float(mid[0]), float(mid[1]))


class _TrackedHand:
    def __init__(self, hand_id: int) -> None:
        self.hand_id = hand_id
        self.filter = OneEuroFilter()
        self.pinch = PinchDetector()
        self.points: np.ndarray | None = None


class HandTracker:
    """Runs the HandLandmarker (VIDEO mode) and keeps per-hand state across frames."""

    def __init__(self, model_path: Path, num_hands: int = 2, detect_width: int = 640) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._detect_width = detect_width
        self._last_timestamp_ms = -1
        self._tracked: list[_TrackedHand] = []
        self._next_id = 0

    def close(self) -> None:
        self._landmarker.close()

    def process(self, frame_bgr: np.ndarray, now: float) -> list[Hand]:
        """Detect hands in a BGR frame; `now` is a monotonic time in seconds."""
        import mediapipe as mp

        height, width = frame_bgr.shape[:2]
        small = frame_bgr
        if width > self._detect_width:
            scale = self._detect_width / width
            small = cv2.resize(frame_bgr, (self._detect_width, int(round(height * scale))))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = max(int(now * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        detections = [
            np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float64)
            for landmarks in result.hand_landmarks
        ]
        return self._update_tracks(detections, now, math.hypot(width, height))

    def _update_tracks(self, detections: list[np.ndarray], now: float, diagonal: float) -> list[Hand]:
        unmatched = list(self._tracked)
        tracked: list[_TrackedHand] = []
        hands: list[Hand] = []
        max_jump = IDENTITY_MATCH_FRACTION * diagonal
        for raw in detections:
            match = None
            best = max_jump
            for candidate in unmatched:
                jump = float(np.linalg.norm(candidate.points[WRIST] - raw[WRIST]))
                if jump < best:
                    best, match = jump, candidate
            if match is None:
                match = _TrackedHand(self._next_id)
                self._next_id += 1
            else:
                unmatched.remove(match)
            previous_tip = None
            if match.points is not None:
                previous_tip = (float(match.points[INDEX_TIP, 0]), float(match.points[INDEX_TIP, 1]))
            points = match.filter(raw, now)
            match.points = points
            pinching = match.pinch.update(points)
            tracked.append(match)
            hands.append(Hand(match.hand_id, points, pinching, previous_tip))
        self._tracked = tracked
        return hands
