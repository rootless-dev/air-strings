"""Dark theme: palette, colors and the widget stylesheet."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

BACKGROUND = "#121212"
SURFACE = "#1a1a1a"
SURFACE_HOVER = "#232323"
SURFACE_PRESSED = "#2b2b2b"
BORDER = "#3c3c3c"
BORDER_HOVER = "#8a8a8a"
FRAME_BORDER = "#5a5a5a"  # around the camera view
TEXT = "#e6e6e6"
TEXT_MUTED = "#8c8c8c"
TEXT_FAINT = "#5e5e5e"
ERROR = "#ff8a7a"

# The strings' gradient (render.py / string_fx.py), as RGB: lime on the left, white-cyan on the right.
ACCENT_LEFT = QColor(165, 255, 80)
ACCENT_RIGHT = QColor(200, 255, 255)
GLOW_LEFT = QColor(120, 255, 30)
GLOW_RIGHT = QColor(50, 230, 255)

STYLESHEET = f"""
QWidget#mainWindow {{
    background: {BACKGROUND};
}}
QWidget {{
    color: {TEXT};
}}
QPushButton, QComboBox {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 9px 14px;
    font-size: 14px;
}}
QPushButton:hover, QComboBox:hover {{
    background: {SURFACE_HOVER};
    border-color: {BORDER_HOVER};
}}
QPushButton:pressed {{
    background: {SURFACE_PRESSED};
    border-color: {ACCENT_LEFT.name()};
}}
QComboBox:disabled {{
    color: {TEXT_MUTED};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: none;
    width: 0px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {SURFACE_PRESSED};
    outline: none;
    padding: 4px;
}}
QLabel#sectionTitle {{
    color: {TEXT_MUTED};
    font-size: 11px;
    letter-spacing: 1px;
}}
QLabel#hint {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QFrame#metricsBar {{
    background: #161616;
    border-top: 1px solid #2c2c2c;
}}
QLabel#metricCaption {{
    color: {TEXT_MUTED};
}}
QLabel#metricValue {{
    color: {TEXT};
}}
QLabel#status {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QLabel#status[error="true"] {{
    color: {ERROR};
}}
"""


def apply_theme(app: QApplication) -> None:
    """Fusion style with a dark palette, plus the stylesheet."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE_HOVER))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(SURFACE_PRESSED))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    app.setPalette(palette)
    font = app.font()
    font.setPixelSize(13)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET)
