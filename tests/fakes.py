"""Cameras, trackers and audio stand-ins shared by the camera-session and window tests."""

import threading

import numpy as np


def detection_threads():
    return [t for t in threading.enumerate() if t.name == "air-strings-detection"]


class FakeCamera:
    """Delivers black frames (or none, when `dead`) and records its release."""

    def __init__(self, index=0, dead=False, size=(72, 128)):
        self.index = index
        self.dead = dead
        self.size = size
        self.released = False

    def read(self):
        if self.dead:
            return False, None
        return True, np.zeros((*self.size, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class CameraFactory:
    """Opens FakeCameras by index; indices in `broken` cannot be opened."""

    def __init__(self, broken=()):
        self.broken = set(broken)
        self.opened = []

    def __call__(self, index):
        if index in self.broken:
            return None
        camera = FakeCamera(index)
        self.opened.append(camera)
        return camera


class FakeTracker:
    def __init__(self, error=None):
        self.error = error
        self.closed = False

    def process(self, frame, now):
        if self.error is not None:
            raise self.error
        return []

    def close(self):
        self.closed = True


class TrackerFactory:
    def __init__(self):
        self.made = []

    def __call__(self):
        tracker = FakeTracker()
        self.made.append(tracker)
        return tracker


class SilentAudio:
    active_voices = 0

    def pluck(self, string_id, frequency, velocity, position):
        pass
