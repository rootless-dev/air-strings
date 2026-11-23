"""The bottom bar: a status message on the left, live metrics on the right."""

from __future__ import annotations

from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from air_strings.engine import Metrics

# (key, caption, widest expected value): the width keeps the bar from jittering as numbers change.
METRICS = (
    ("fps", "Display", "00.0 fps"),
    ("detection_fps", "Detection", "00.0 fps"),
    ("detection_ms", "Detect time", "00.0 ms"),
    ("render_ms", "Render time", "00.0 ms"),
    ("strings", "Strings", "0/0"),
    ("scale", "Scale", "C major pentatonic"),
    ("voices", "Voices", "00"),
)


def format_metrics(metrics: Metrics) -> dict[str, str]:
    return {
        "fps": f"{metrics.fps:.1f} fps",
        "detection_fps": f"{metrics.detection_fps:.1f} fps",
        "detection_ms": f"{metrics.detection_ms:.1f} ms",
        "render_ms": f"{metrics.render_ms:.1f} ms",
        "strings": f"{metrics.strings}/{metrics.max_strings}",
        "scale": metrics.scale,
        "voices": str(metrics.voices),
    }


class MetricsBar(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricsBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 10, 24, 12)
        layout.setSpacing(20)

        self._status = QLabel("")
        self._status.setObjectName("status")
        # The status gives way first when the window is narrow.
        self._status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self._status, 1)

        caption_font = QFont(self.font())
        caption_font.setPixelSize(10)
        caption_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 108)
        text_font = QFont(self.font())
        text_font.setPixelSize(13)
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPixelSize(12)
        self._values: dict[str, QLabel] = {}
        for key, caption, widest in METRICS:
            cell = QVBoxLayout()
            cell.setSpacing(2)
            caption_label = QLabel(caption.upper())
            caption_label.setObjectName("metricCaption")
            caption_label.setFont(caption_font)
            value = QLabel("-")
            value.setObjectName("metricValue")
            font = text_font if key == "scale" else mono
            value.setFont(font)
            value.setMinimumWidth(QFontMetrics(font).horizontalAdvance(widest))
            cell.addWidget(caption_label)
            cell.addWidget(value)
            layout.addLayout(cell)
            self._values[key] = value

    def show_metrics(self, metrics: Metrics) -> None:
        for key, text in format_metrics(metrics).items():
            self._values[key].setText(text)

    def value_text(self, key: str) -> str:
        return self._values[key].text()

    def show_status(self, text: str, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setProperty("error", "true" if error else "false")
        # A dynamic property only restyles after a re-polish.
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    @property
    def status_text(self) -> str:
        return self._status.text()

    @property
    def status_is_error(self) -> bool:
        return self._status.property("error") == "true"
