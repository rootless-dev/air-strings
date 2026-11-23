"""The cameras the user can pick, as OpenCV capture indices with readable names.

OpenCV numbers cameras but does not name them, so the names come from Qt
Multimedia (`QMediaDevices.videoInputs()`). The two lists are in different
orders, and Qt's order is not even stable from run to run. On macOS, OpenCV's
AVFoundation backend sorts the devices by their unique ID before numbering
them, and Qt's `QCameraDevice.id()` is that same unique ID, so sorting Qt's
devices by ID gives OpenCV's indices. This assumes both see the same set of
devices (true for built-in, USB and virtual cameras on the machine it was
checked on). Elsewhere, or when Qt lists nothing, the indices are probed and
named "Camera N".
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import cv2
from PySide6.QtCore import QLoggingCategory

from air_strings.camera import CameraInfo

PROBE_LIMIT = 5  # indices 0..4 are tried when names are not available


def list_cameras(in_use: int | None = None) -> list[CameraInfo]:
    """The available cameras, by OpenCV index. `in_use` is not probed (it is open already)."""
    named = qt_cameras()
    if named:
        return named
    return probe_cameras(in_use)


def qt_cameras() -> list[CameraInfo]:
    """Named cameras from Qt Multimedia, numbered like OpenCV's; empty if that is not known here."""
    if sys.platform != "darwin":
        return []
    QLoggingCategory.setFilterRules("qt.multimedia.ffmpeg=false")  # skip its version banner
    try:
        from PySide6.QtMultimedia import QMediaDevices
    except ImportError:
        return []
    devices = [(bytes(device.id()).decode("utf-8", "replace"), device.description()) for device in QMediaDevices.videoInputs()]
    return number_like_opencv(devices)


def number_like_opencv(devices: list[tuple[str, str]]) -> list[CameraInfo]:
    """(unique ID, name) pairs, numbered the way OpenCV's AVFoundation backend numbers them."""
    return [CameraInfo(index, name) for index, (_, name) in enumerate(sorted(devices))]


def probe_cameras(
    in_use: int | None = None,
    limit: int = PROBE_LIMIT,
    opener: Callable[[int], cv2.VideoCapture] = cv2.VideoCapture,
) -> list[CameraInfo]:
    """Indices 0..limit-1 that open, each released straight away."""
    found = []
    for index in range(limit):
        if index != in_use:
            capture = opener(index)
            opened = capture.isOpened()
            capture.release()
            if not opened:
                continue
        found.append(CameraInfo(index, f"Camera {index}"))
    return found
