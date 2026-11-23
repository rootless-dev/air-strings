"""The main window and the Qt event loop that runs it.

Detections reach the window through a Qt signal: the detection worker calls
the mailbox listener on its own thread, the listener emits a signal on a
QObject that lives in the GUI thread, and Qt delivers it there as a queued
call. The slot takes the newest detection without waiting, steps the engine
and repaints the video. There is no polling and no wait in the GUI thread, and
a frame is shown as soon as it is detected.

Switching cameras opens the new camera on a background thread (it can take a
second) and comes back to the GUI thread through another queued signal, where
the session swaps workers; the old camera keeps running until then, and stays
if the new one cannot be opened.
"""

from __future__ import annotations

import signal
import sys
import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from air_strings.camera import Camera, CameraInfo, CameraSession
from air_strings.engine import Engine
from air_strings.ui import theme
from air_strings.ui.cameras import list_cameras
from air_strings.ui.logo import LogoWidget, app_icon
from air_strings.ui.metrics_bar import MetricsBar
from air_strings.ui.video_view import VideoView

WINDOW_TITLE = "air-strings"
DEFAULT_SIZE = (1280, 740)
MINIMUM_SIZE = (820, 520)
SIDE_COLUMN_WIDTH = 210
METRICS_INTERVAL_MS = 250  # the bar is refreshed a few times per second, not every frame
STATUS_TIMEOUT_MS = 5000
GESTURE_HINTS = (
    ("Draw", "pinch with both hands, pull apart"),
    ("Resize", "pinch an end and drag"),
    ("Pluck", "push a finger through a string"),
)
KEYS_HINT = "c  clear     s  scale     q  quit"

CameraLister = Callable[[], list[CameraInfo]]
BackgroundRunner = Callable[[Callable[[], None]], None]


def start_daemon_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="air-strings-camera-open", daemon=True).start()


class _Bridge(QObject):
    """Signals emitted from other threads, delivered (queued) in the GUI thread."""

    detection_ready = Signal()
    camera_opened = Signal(int, object)  # index, the opened camera or None

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # Set when a wake-up is queued and not yet handled, so a worker faster than the
        # GUI does not pile up wake-ups (each would only find the mailbox empty).
        self._wake_pending = False

    def wake(self) -> None:
        """Mailbox listener, called on the worker thread."""
        if not self._wake_pending:
            self._wake_pending = True
            self.detection_ready.emit()

    def woken(self) -> None:
        """Called by the slot before it takes from the mailbox."""
        self._wake_pending = False


class ChevronComboBox(QComboBox):
    """A combo box that draws its own small chevron (the stylesheet hides the native arrow)."""

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(theme.TEXT_MUTED), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        x, y = self.width() - 18.0, self.height() / 2.0
        painter.drawPolyline([QPointF(x - 4.0, y - 2.0), QPointF(x, y + 2.0), QPointF(x + 4.0, y - 2.0)])
        painter.end()


class MainWindow(QWidget):
    """Logo, camera view, camera picker and buttons, and the metrics bar."""

    closed = Signal()

    def __init__(
        self,
        engine: Engine,
        session: CameraSession,
        camera_lister: CameraLister,
        max_frames: int = 0,
        run_in_background: BackgroundRunner = start_daemon_thread,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._session = session
        self._max_frames = max_frames
        self._run_in_background = run_in_background
        self._closing = False
        self._shut_down = threading.Event()  # seen by camera-opening threads
        self._switching_to: int | None = None
        self.frames_shown = 0
        self.error: BaseException | None = None  # raised by `run_window` once the loop ends

        self.setObjectName("mainWindow")
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(app_icon())
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.resize(*DEFAULT_SIZE)
        self.setMinimumSize(*MINIMUM_SIZE)
        self._build(camera_lister())

        self._bridge = _Bridge(self)
        self._bridge.detection_ready.connect(self._on_detection_ready)
        self._bridge.camera_opened.connect(self._on_camera_opened)
        session.set_listener(self._bridge.wake)
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(METRICS_INTERVAL_MS)
        self._metrics_timer.timeout.connect(self._refresh_metrics)
        self._metrics_timer.start()
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self.metrics_bar.show_status(""))
        # A detection may have arrived before the listener was set.
        QTimer.singleShot(0, self._on_detection_ready)
        self._refresh_metrics()

    # Layout

    def _build(self, cameras: list[CameraInfo]) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QVBoxLayout()
        content.setContentsMargins(24, 18, 24, 20)
        content.setSpacing(16)
        self.logo = LogoWidget()
        content.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignLeft)

        body = QHBoxLayout()
        body.setSpacing(22)
        self.video = VideoView()
        body.addWidget(self.video, 1)
        body.addWidget(self._build_side_column(cameras))
        content.addLayout(body, 1)
        root.addLayout(content, 1)

        self.metrics_bar = MetricsBar()
        root.addWidget(self.metrics_bar)

    def _build_side_column(self, cameras: list[CameraInfo]) -> QWidget:
        column = QWidget()
        column.setFixedWidth(SIDE_COLUMN_WIDTH)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(_section_title("CAMERA"))
        self.camera_box = ChevronComboBox()
        self.camera_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # typed keys are shortcuts, not item search
        if not any(camera.index == self._session.index for camera in cameras):
            cameras = [*cameras, CameraInfo(self._session.index, f"Camera {self._session.index}")]
        for camera in cameras:
            self.camera_box.addItem(camera.name, camera.index)
        self.camera_box.setCurrentIndex(self.camera_box.findData(self._session.index))
        self.camera_box.currentIndexChanged.connect(self._on_camera_picked)
        layout.addWidget(self.camera_box)

        layout.addSpacing(14)
        layout.addWidget(_section_title("STRINGS"))
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_strings)
        self.scale_button = QPushButton()
        self.scale_button.clicked.connect(self.cycle_scale)
        for button in (self.clear_button, self.scale_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(button)
        self._show_scale()

        layout.addSpacing(14)
        layout.addWidget(_section_title("GESTURES"))
        for name, text in GESTURE_HINTS:
            hint = QLabel(f"<span style='color:{theme.TEXT}'>{name}</span>&nbsp; {text}")
            hint.setObjectName("hint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
        layout.addStretch(1)
        keys = QLabel(KEYS_HINT)
        keys.setObjectName("hint")
        layout.addWidget(keys)
        return column

    # Actions

    def clear_strings(self) -> None:
        self._engine.clear()
        self._refresh_metrics()

    def cycle_scale(self) -> None:
        self._engine.cycle_scale()
        self._show_scale()
        self._refresh_metrics()

    def _show_scale(self) -> None:
        self.scale_button.setText(f"Scale: {self._engine.scale.label}")

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt naming
        key = event.key()
        if key == Qt.Key.Key_C:
            self.clear_strings()
        elif key == Qt.Key.Key_S:
            self.cycle_scale()
        elif key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    # Frames

    def _on_detection_ready(self) -> None:
        if self._closing:
            return
        self._bridge.woken()
        try:
            mailbox = self._session.mailbox
            detection = mailbox.take(0.0)
            if detection is not None:
                self._engine.step(detection)
                self.video.set_frame(detection.frame)
                self.frames_shown += 1
                if 0 < self._max_frames <= self.frames_shown:
                    self.close()
                elif mailbox.closed:
                    # Its close may have found a wake-up already pending: look again once
                    # this one is done, rather than waiting for a wake-up that will not come.
                    QTimer.singleShot(0, self._on_detection_ready)
            elif mailbox.closed:
                self.close()  # the worker ended on its own; the caller reports why
        except BaseException as error:  # noqa: BLE001 - re-raised by run_window, outside the Qt loop
            self.error = error
            self.close()

    def _refresh_metrics(self) -> None:
        self.metrics_bar.show_metrics(self._engine.metrics())

    def _show_status(self, text: str, error: bool = False, sticky: bool = False) -> None:
        self.metrics_bar.show_status(text, error)
        if sticky:
            self._status_timer.stop()
        else:
            self._status_timer.start(STATUS_TIMEOUT_MS)

    # Cameras

    def _camera_name(self, index: int) -> str:
        position = self.camera_box.findData(index)
        return self.camera_box.itemText(position) if position >= 0 else f"Camera {index}"

    def _on_camera_picked(self, position: int) -> None:
        index = self.camera_box.itemData(position)
        if index is None or index == self._session.index or self._switching_to is not None:
            return
        self._switching_to = index
        self.camera_box.setEnabled(False)
        self._show_status(f"Opening {self._camera_name(index)}...", sticky=True)
        session, bridge, shut_down = self._session, self._bridge, self._shut_down

        def open_camera() -> None:
            try:
                capture = session.open(index)
            except Exception:  # noqa: BLE001 - a camera that fails to open is reported, not fatal
                capture = None
            if shut_down.is_set():  # the window is gone: nobody will adopt this camera
                if capture is not None:
                    capture.release()
                return
            bridge.camera_opened.emit(index, capture)

        self._run_in_background(open_camera)

    def _on_camera_opened(self, index: int, capture: Camera | None) -> None:
        self._switching_to = None
        if self._closing:
            if capture is not None:
                capture.release()
            return
        self.camera_box.setEnabled(True)
        if capture is None:
            self._select_camera(self._session.index)
            self._show_status(
                f"Cannot open {self._camera_name(index)}; still on {self._camera_name(self._session.index)}",
                error=True,
            )
            return
        try:
            self._session.switch_to(index, capture)
        except BaseException as error:  # noqa: BLE001 - e.g. the new tracker failed to load
            self.error = error
            self.close()
            return
        self._show_status(f"Camera: {self._camera_name(index)}")

    def _select_camera(self, index: int) -> None:
        self.camera_box.blockSignals(True)
        self.camera_box.setCurrentIndex(self.camera_box.findData(index))
        self.camera_box.blockSignals(False)

    # Shutdown

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self._closing:
            self._closing = True
            self._shut_down.set()
            self._metrics_timer.stop()
            self._status_timer.stop()
            self._session.set_listener(None)
            self.closed.emit()
        event.accept()


def _section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


_application: QApplication | None = None  # kept referenced for the life of the process


def ensure_application() -> QApplication:
    """The process's QApplication, created and themed on first use."""
    global _application
    app = QApplication.instance()
    if app is None:
        app = QApplication([sys.argv[0]])
        app.setApplicationName(WINDOW_TITLE)
        theme.apply_theme(app)
        app.setWindowIcon(app_icon())
    _application = app
    return app


def run_window(
    engine: Engine,
    session: CameraSession,
    max_frames: int = 0,
    camera_lister: CameraLister | None = None,
) -> None:
    """Show the window and run the Qt event loop until it is closed; re-raises any error it caught."""
    app = ensure_application()
    lister = camera_lister or (lambda: list_cameras(in_use=session.index))
    window = MainWindow(engine, session, lister, max_frames)
    window.closed.connect(app.quit)
    window.show()
    window.raise_()
    window.activateWindow()
    # Ctrl+C in the terminal closes the window cleanly (the handler runs between Qt events).
    previous = signal.signal(signal.SIGINT, lambda *_: window.close())
    try:
        app.exec()
    finally:
        signal.signal(signal.SIGINT, previous)
        window.close()
    if window.error is not None:
        raise window.error
