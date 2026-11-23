"""The camera view: the rendered frame, letterboxed to its aspect ratio, in a rounded frame."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from air_strings.ui import theme

DEFAULT_ASPECT = 16 / 9
CORNER_RADIUS = 14.0
BORDER_WIDTH = 1.5
PLACEHOLDER_TEXT = "Waiting for the camera..."


def letterbox(width: int, height: int, aspect: float) -> tuple[int, int, int, int]:
    """The largest `aspect` (width / height) rectangle centered in `width` x `height`: (x, y, w, h)."""
    if width <= 0 or height <= 0:
        return 0, 0, 0, 0
    if width / height > aspect:  # too wide: bars left and right
        fitted_height = height
        fitted_width = round(height * aspect)
    else:  # too tall: bars above and below
        fitted_width = width
        fitted_height = round(width / aspect)
    return (width - fitted_width) // 2, (height - fitted_height) // 2, fitted_width, fitted_height


def frame_to_image(frame_bgr: np.ndarray, width: int, height: int) -> tuple[QImage, np.ndarray]:
    """The frame scaled to `width` x `height` as a QImage, and the pixel buffer it points into.

    The image does not own its pixels: keep the returned buffer alive as long as the image.
    OpenCV scales (area averaging when shrinking) and adds the alpha channel, so the result
    is Format_RGB32, which Qt draws without converting.
    """
    if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
        shrinking = width < frame_bgr.shape[1]
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=interpolation)
    buffer = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)  # little-endian 0xffRRGGBB
    image = QImage(buffer.data, width, height, buffer.strides[0], QImage.Format.Format_RGB32)
    return image, buffer


class VideoView(QWidget):
    """Shows the latest frame, scaled smoothly to fit, with rounded corners and a light border."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoView")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 180)
        self._frame: np.ndarray | None = None
        self._image: QImage | None = None
        self._buffer: np.ndarray | None = None  # keeps `_image`'s pixels alive

    @property
    def aspect(self) -> float:
        if self._frame is None:
            return DEFAULT_ASPECT
        height, width = self._frame.shape[:2]
        return width / height

    @property
    def image(self) -> QImage | None:
        return self._image

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """Show `frame_bgr` (BGR, uint8); the view keeps a reference to it until the next one."""
        self._frame = frame_bgr
        self._rescale()
        self.update()

    def video_rect(self) -> QRect:
        """Where the frame is drawn, in widget coordinates: letterboxed inside the border.

        Centered horizontally, but kept at the top, so the video stays under the logo
        and level with the controls next to it when the window is taller than the frame.
        """
        inset = int(np.ceil(BORDER_WIDTH))
        x, _, w, h = letterbox(self.width() - 2 * inset, self.height() - 2 * inset, self.aspect)
        return QRect(x + inset, inset, w, h)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._frame is None:
            return
        rect = self.video_rect()
        ratio = self.devicePixelRatioF()
        width, height = max(1, round(rect.width() * ratio)), max(1, round(rect.height() * ratio))
        self._image, self._buffer = frame_to_image(self._frame, width, height)
        self._image.setDevicePixelRatio(ratio)

    def paintEvent(self, event) -> None:  # noqa: N802, ARG002 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.video_rect())
        path = QPainterPath()
        path.addRoundedRect(rect, CORNER_RADIUS, CORNER_RADIUS)
        if self._image is not None:
            painter.save()
            painter.setClipPath(path)
            painter.drawImage(rect.topLeft(), self._image)
            painter.restore()
        else:
            painter.fillPath(path, QColor(theme.SURFACE))
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, PLACEHOLDER_TEXT)
        # The border is drawn over the clipped edge, which hides its (aliased) stair steps.
        painter.setPen(QPen(QColor(theme.FRAME_BORDER), BORDER_WIDTH))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.end()
