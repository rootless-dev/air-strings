"""The camera in use: its capture, hand tracker and detection worker, and switching cameras.

A `CameraSession` owns one capture, one tracker and the `DetectionWorker` that
runs them. Switching to another camera opens the new capture first (possibly
on another thread, since opening a camera can take a second), and only then
stops the old worker, frees the old capture and tracker, and starts a fresh
worker on the new camera with a new tracker. If the new camera cannot be
opened, the old one keeps running untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from air_strings.hands import Hand
from air_strings.pipeline import DetectionWorker, LatestDetection


class Camera(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


class Tracker(Protocol):
    def process(self, frame_bgr: np.ndarray, now: float) -> list[Hand]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CameraInfo:
    index: int  # the OpenCV capture index
    name: str


# Opens camera `index`, or returns None if it cannot be opened.
CaptureFactory = Callable[[int], Camera | None]
TrackerFactory = Callable[[], Tracker]
WorkerFactory = Callable[[Camera, Tracker], DetectionWorker]


def default_worker(capture: Camera, tracker: Tracker) -> DetectionWorker:
    return DetectionWorker(capture, tracker, LatestDetection())


class CameraSession:
    """The current camera's capture, tracker and worker; `switch_to` replaces all three.

    Every method except `open` must be called from one thread (the UI thread).
    """

    def __init__(
        self,
        index: int,
        capture: Camera,
        capture_factory: CaptureFactory,
        tracker_factory: TrackerFactory,
        worker_factory: WorkerFactory = default_worker,
    ) -> None:
        self._capture_factory = capture_factory
        self._tracker_factory = tracker_factory
        self._worker_factory = worker_factory
        self._listener: Callable[[], None] | None = None
        self._stopped = False
        self._index = index
        self._capture = capture
        self._tracker = tracker_factory()
        self._worker = worker_factory(capture, self._tracker)

    @property
    def index(self) -> int:
        return self._index

    @property
    def worker(self) -> DetectionWorker:
        return self._worker

    @property
    def mailbox(self) -> LatestDetection:
        return self._worker.mailbox

    def set_listener(self, listener: Callable[[], None] | None) -> None:
        """Wake-up callback for the current worker's mailbox, kept across camera switches."""
        self._listener = listener
        self._worker.mailbox.set_listener(listener)

    def start(self) -> None:
        self._worker.start()

    def open(self, index: int) -> Camera | None:
        """Open camera `index` without touching the running one; safe to call from any thread."""
        return self._capture_factory(index)

    def switch_to(self, index: int, capture: Camera) -> None:
        """Replace the running camera with `capture` (already opened by `open`), with a new tracker."""
        if self._stopped:
            capture.release()
            return
        self._shut_down_current()
        self._index = index
        self._capture = capture
        self._tracker = self._tracker_factory()
        self._worker = self._worker_factory(capture, self._tracker)
        self._worker.mailbox.set_listener(self._listener)
        self._worker.start()

    def switch(self, index: int) -> bool:
        """Open camera `index` and switch to it; False (and nothing changes) if it cannot be opened."""
        if index == self._index:
            return True
        capture = self.open(index)
        if capture is None:
            return False
        self.switch_to(index, capture)
        return True

    def stop(self) -> None:
        """Stop the worker and free the camera and the tracker; later calls do nothing."""
        if self._stopped:
            return
        self._stopped = True
        self._shut_down_current()

    def _shut_down_current(self) -> None:
        self._worker.mailbox.set_listener(None)
        # If the worker is somehow still mid-read after the timeout, leave the camera and
        # the tracker to it rather than freeing them under its feet.
        if self._worker.stop():
            self._tracker.close()
            self._capture.release()
