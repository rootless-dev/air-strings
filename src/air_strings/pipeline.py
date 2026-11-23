"""Camera capture and hand detection on a background thread.

The worker reads a frame, mirrors it, runs the hand tracker on it and publishes
the bundle to a single-slot mailbox. A new bundle overwrites one the main thread
has not taken yet, so the display always gets the freshest frame and never
falls behind a queue of stale ones. The overwritten bundle's hands are kept,
though: gestures need every detection, because each hand's previous fingertip
is the one from the tracker's previous detection, displayed or not. OpenCV and
MediaPipe release the GIL in their native calls, so capture and detection run
in parallel with rendering.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from air_strings.hands import Hand

# Consecutive failed reads after which the camera is considered gone.
MAX_FAILED_READS = 30
# How long stop() waits for the worker to finish its current read and detection.
STOP_TIMEOUT_S = 2.0
# Most skipped detections whose hands are kept for the gestures (half a second at
# 30 fps); after a longer stall the oldest are dropped, rather than replaying stale plucks.
MAX_SKIPPED = 15


class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...


class HandDetector(Protocol):
    def process(self, frame_bgr: np.ndarray, now: float) -> list[Hand]: ...


class CameraStoppedError(RuntimeError):
    """The camera stopped delivering frames."""


@dataclass(frozen=True)
class HandSample:
    """The hands of a detection whose frame was never displayed."""

    hands: list[Hand]
    timestamp: float
    dt: float


@dataclass(frozen=True)
class Detection:
    frame: np.ndarray  # mirrored BGR frame, owned by the consumer once taken
    hands: list[Hand]  # hands detected on this very frame
    timestamp: float  # monotonic capture time, in seconds
    dt: float  # time since the previous detection (0 for the first one)
    sequence: int  # 1 for the first detection, then +1 per detection
    # Hands of the detections this one overwrote before they were taken, oldest first.
    skipped: tuple[HandSample, ...] = ()


class LatestDetection:
    """Single-slot mailbox: publishing replaces any detection not taken yet.

    The replaced detection's frame is dropped, but its hands are carried over
    in the new detection's `skipped` (up to `max_skipped` of them).
    """

    def __init__(self, max_skipped: int = MAX_SKIPPED) -> None:
        self._condition = threading.Condition()
        self._item: Detection | None = None
        self._closed = False
        self._max_skipped = max_skipped

    def publish(self, item: Detection) -> None:
        with self._condition:
            old = self._item
            if old is not None:
                skipped = (*old.skipped, HandSample(old.hands, old.timestamp, old.dt))
                item = dataclasses.replace(item, skipped=skipped[max(0, len(skipped) - self._max_skipped) :])
            self._item = item
            self._condition.notify_all()

    def close(self) -> None:
        """No more detections will come; wakes up a waiting `take`."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def take(self, timeout: float) -> Detection | None:
        """The newest detection not taken yet, or None after `timeout` seconds or once closed and empty."""
        with self._condition:
            self._condition.wait_for(lambda: self._item is not None or self._closed, timeout)
            item, self._item = self._item, None
            return item


class DetectionWorker:
    """Owns the capture-and-detect thread; results arrive through `mailbox`.

    The thread ends when `stop` is called, when the camera fails
    `max_failed_reads` times in a row (`error` is then a `CameraStoppedError`)
    or when the tracker raises (`error` is that exception). In every case the
    mailbox is closed, so the consumer never waits forever.
    """

    def __init__(
        self,
        capture: FrameSource,
        tracker: HandDetector,
        mailbox: LatestDetection,
        clock: Callable[[], float] = time.monotonic,
        max_failed_reads: int = MAX_FAILED_READS,
    ) -> None:
        self._capture = capture
        self._tracker = tracker
        self.mailbox = mailbox
        self._clock = clock
        self._max_failed_reads = max_failed_reads
        self._stop = threading.Event()
        # Daemon: if `stop` times out (a read stuck on a hung camera), the thread is
        # abandoned and must not keep the process alive at interpreter exit.
        self._thread = threading.Thread(target=self._run, name="air-strings-detection", daemon=True)
        self.error: BaseException | None = None

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = STOP_TIMEOUT_S) -> bool:
        """Signal the thread to stop and join it; returns False if it did not finish in time."""
        self._stop.set()
        if self._thread.ident is None:
            return True  # never started
        self._thread.join(timeout)
        if self._thread.is_alive():
            print("air-strings: detection thread did not stop in time", file=sys.stderr)
            return False
        return True

    def _run(self) -> None:
        try:
            self._loop()
        except BaseException as error:  # surfaced to the main thread through `error`
            self.error = error
        finally:
            self.mailbox.close()

    def _loop(self) -> None:
        failed_reads = 0
        sequence = 0
        last_timestamp: float | None = None
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= self._max_failed_reads:
                    raise CameraStoppedError("camera stopped delivering frames")
                continue
            failed_reads = 0
            timestamp = self._clock()
            frame = cv2.flip(frame, 1)  # mirror, like a selfie view
            hands = self._tracker.process(frame, timestamp)
            dt = timestamp - last_timestamp if last_timestamp is not None else 0.0
            last_timestamp = timestamp
            sequence += 1
            self.mailbox.publish(Detection(frame, hands, timestamp, dt, sequence))
