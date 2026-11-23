"""The logo, drawn in code: a plucked-string glyph in the strings' gradient, and the wordmark."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QWidget

from air_strings.ui import theme

WORDMARK = "air-strings"
GLYPH_ASPECT = 1.6  # width / height
WAVE_CYCLES = 2.5
WAVE_SAMPLES = 64


def wave_path(rect: QRectF) -> QPainterPath:
    """A string between two ends, caught mid-vibration: a sine under a sin(pi x) envelope."""
    path = QPainterPath()
    middle = rect.center().y()
    amplitude = rect.height() * 0.34
    for i in range(WAVE_SAMPLES + 1):
        x = i / WAVE_SAMPLES
        y = middle - amplitude * math.sin(math.pi * x) * math.sin(2.0 * math.pi * WAVE_CYCLES * x)
        point = QPointF(rect.left() + x * rect.width(), y)
        if i == 0:
            path.moveTo(point)
        else:
            path.lineTo(point)
    return path


def paint_glyph(painter: QPainter, rect: QRectF) -> None:
    """The glyph inside `rect`: a soft glow, the bright string and its two end rings."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    unit = rect.height()
    ring = unit * 0.09
    inner = rect.adjusted(ring * 2.0, 0.0, -ring * 2.0, 0.0)
    path = wave_path(inner)

    glow = QLinearGradient(inner.topLeft(), inner.topRight())
    glow.setColorAt(0.0, _with_alpha(theme.GLOW_LEFT, 70))
    glow.setColorAt(1.0, _with_alpha(theme.GLOW_RIGHT, 70))
    painter.setPen(QPen(glow, unit * 0.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPath(path)

    core = QLinearGradient(inner.topLeft(), inner.topRight())
    core.setColorAt(0.0, theme.ACCENT_LEFT)
    core.setColorAt(1.0, theme.ACCENT_RIGHT)
    painter.setPen(QPen(core, unit * 0.075, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawPath(path)

    painter.setBrush(Qt.BrushStyle.NoBrush)
    for point, color in ((path.pointAtPercent(0.0), theme.ACCENT_LEFT), (path.pointAtPercent(1.0), theme.ACCENT_RIGHT)):
        painter.setPen(QPen(color, unit * 0.05))
        painter.drawEllipse(point, ring, ring)
    painter.restore()


def _with_alpha(color: QColor, alpha: int) -> QColor:
    result = QColor(color)
    result.setAlpha(alpha)
    return result


def app_icon(size: int = 256) -> QIcon:
    """The glyph on a dark rounded square, for the window and the dock."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    margin = size * 0.06
    tile = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    painter.setPen(QPen(QColor(theme.FRAME_BORDER), size * 0.012))
    painter.setBrush(QColor(theme.BACKGROUND))
    painter.drawRoundedRect(tile, size * 0.2, size * 0.2)
    glyph_width = tile.width() * 0.78
    glyph_height = glyph_width / GLYPH_ASPECT
    paint_glyph(painter, QRectF(tile.center().x() - glyph_width / 2, tile.center().y() - glyph_height / 2,
                                glyph_width, glyph_height))
    painter.end()
    return QIcon(pixmap)


class LogoWidget(QWidget):
    """Glyph and wordmark, side by side."""

    def __init__(self, height: int = 30, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("logo")
        self._glyph_height = height
        self._font = QFont(self.font())
        self._font.setPixelSize(round(height * 0.7))
        self._font.setWeight(QFont.Weight.DemiBold)
        self._font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 102)
        self._gap = round(height * 0.4)
        self.setFixedSize(self.sizeHint())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        text_width = QFontMetrics(self._font).horizontalAdvance(WORDMARK)
        return QSize(round(self._glyph_height * GLYPH_ASPECT) + self._gap + text_width + 2, self._glyph_height + 4)

    def paintEvent(self, event) -> None:  # noqa: N802, ARG002 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        glyph = QRectF(0.0, 2.0, self._glyph_height * GLYPH_ASPECT, self._glyph_height)
        paint_glyph(painter, glyph)
        painter.setFont(self._font)
        painter.setPen(QColor(theme.TEXT))
        text_rect = QRectF(glyph.right() + self._gap, 0.0, self.width() - glyph.right() - self._gap, self.height())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, WORDMARK)
        painter.end()
