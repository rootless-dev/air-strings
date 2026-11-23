"""The Qt window, offscreen: widgets, buttons, keys, metrics, video, cameras and shutdown."""

import functools
import math
import time

import numpy as np
import pytest
from fakes import CameraFactory, FakeCamera, FakeTracker, SilentAudio, TrackerFactory, detection_threads
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from air_strings.app import AirStringsApp
from air_strings.camera import CameraInfo, CameraSession
from air_strings.engine import Engine, Metrics
from air_strings.gestures import DrawGesture, GestureController, GrabGesture, PullGesture
from air_strings.particles import ParticleSystem
from air_strings.pipeline import Detection, DetectionWorker, LatestDetection
from air_strings.render import Renderer
from air_strings.scales import SCALES, PitchMapping
from air_strings.strings import StringSet
from air_strings.ui.cameras import number_like_opencv, probe_cameras
from air_strings.ui.main_window import MainWindow, run_window, start_daemon_thread
from air_strings.ui.video_view import VideoView, frame_to_image, letterbox

CAMERAS = [CameraInfo(0, "Built-in"), CameraInfo(1, "USB webcam"), CameraInfo(2, "Virtual")]


def run_now(work):
    work()


def wait_until(app, condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        app.processEvents()
        time.sleep(0.002)
    return True


def make_engine():
    strings = StringSet(PitchMapping(), SCALES[0])
    engine = Engine(
        strings=strings,
        gestures=GestureController(strings, DrawGesture(), GrabGesture(), PullGesture()),
        particles=ParticleSystem(rng=np.random.default_rng(0)),
        renderer=Renderer(),
        audio=SilentAudio(),
    )
    return engine, strings


def add_string(strings):
    strings.add((100.0, 100.0), (600.0, 120.0), math.hypot(1280, 720), now=0.0)


class Harness:
    """A window over a real (not started) camera session with fake cameras."""

    def __init__(self, qapp, cameras=CAMERAS, index=0, broken=(), max_frames=0, background=run_now):
        self.app = qapp
        self.factory = CameraFactory(broken)
        self.trackers = TrackerFactory()
        self.first = FakeCamera(index)
        self.session = CameraSession(index, self.first, self.factory, self.trackers)
        self.engine, self.strings = make_engine()
        self.window = MainWindow(self.engine, self.session, lambda: cameras, max_frames, background)
        self.closed = []
        self.window.closed.connect(lambda: self.closed.append(True))

    def publish(self, sequence=1, frame=None):
        frame = frame if frame is not None else np.zeros((72, 128, 3), dtype=np.uint8)
        self.session.mailbox.publish(Detection(frame, [], float(sequence), 0.0, sequence))
        self.app.processEvents()

    def close(self):
        self.window.close()
        self.session.stop()


@pytest.fixture
def harness(qapp):
    harnesses = []

    def make(**kwargs):
        h = Harness(qapp, **kwargs)
        harnesses.append(h)
        return h

    yield make
    for h in harnesses:
        h.close()


# Layout and controls


def test_the_window_has_logo_video_controls_and_metrics(harness):
    window = harness().window

    assert window.windowTitle() == "air-strings"
    assert not window.windowIcon().isNull()
    assert window.logo.width() > 0
    assert window.video.minimumWidth() > 0
    assert window.clear_button.text() == "Clear"
    assert window.scale_button.text() == "Scale: Pentatonic"
    assert [window.camera_box.itemText(i) for i in range(window.camera_box.count())] == [c.name for c in CAMERAS]
    assert window.metrics_bar.value_text("strings") == "0/6"


def test_clear_button_clears_the_strings(harness):
    h = harness()
    add_string(h.strings)

    QTest.mouseClick(h.window.clear_button, Qt.MouseButton.LeftButton)

    assert h.strings.strings == []
    assert h.window.metrics_bar.value_text("strings") == "0/6"


def test_scale_button_cycles_the_scale_and_shows_it(harness):
    h = harness()

    QTest.mouseClick(h.window.scale_button, Qt.MouseButton.LeftButton)
    assert h.strings.scale == SCALES[1]
    assert h.window.scale_button.text() == "Scale: Major"
    assert h.window.metrics_bar.value_text("scale") == "C major"

    QTest.mouseClick(h.window.scale_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(h.window.scale_button, Qt.MouseButton.LeftButton)
    assert h.strings.scale == SCALES[0]
    assert h.window.scale_button.text() == "Scale: Pentatonic"


def test_keys_clear_cycle_the_scale_and_quit(harness):
    h = harness()
    add_string(h.strings)

    QTest.keyClick(h.window, Qt.Key.Key_C)
    QTest.keyClick(h.window, Qt.Key.Key_S)
    assert h.strings.strings == []
    assert h.window.scale_button.text() == "Scale: Major"

    QTest.keyClick(h.window, Qt.Key.Key_Q)
    assert h.closed == [True]


def test_escape_quits(harness):
    h = harness()
    QTest.keyClick(h.window, Qt.Key.Key_Escape)
    assert h.closed == [True]


def test_metrics_bar_shows_the_metrics_it_is_given(harness):
    bar = harness().window.metrics_bar

    bar.show_metrics(Metrics(29.84, 30.1, 17.62, 4.25, 3, 6, "chromatic", 5))

    shown = {key: bar.value_text(key) for key in ("fps", "detection_fps", "detection_ms", "render_ms")}
    assert shown == {"fps": "29.8 fps", "detection_fps": "30.1 fps", "detection_ms": "17.6 ms", "render_ms": "4.2 ms"}
    assert (bar.value_text("strings"), bar.value_text("scale"), bar.value_text("voices")) == ("3/6", "chromatic", "5")


# Video


@pytest.mark.parametrize(
    ("size", "aspect", "expected"),
    [
        ((1600, 900), 16 / 9, (0, 0, 1600, 900)),  # exact fit
        ((2000, 900), 16 / 9, (200, 0, 1600, 900)),  # too wide: bars left and right
        ((1600, 1200), 16 / 9, (0, 150, 1600, 900)),  # too tall: bars above and below
        ((800, 800), 4 / 3, (0, 100, 800, 600)),
        ((0, 500), 16 / 9, (0, 0, 0, 0)),
    ],
)
def test_letterbox(size, aspect, expected):
    assert letterbox(*size, aspect) == expected


def test_a_bgr_frame_becomes_a_scaled_image_with_the_same_colors(qapp):
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    frame[:, :64] = (255, 0, 0)  # blue on the left (BGR)
    frame[:, 64:] = (0, 0, 255)  # red on the right

    image, buffer = frame_to_image(frame, 256, 144)

    assert (image.width(), image.height()) == (256, 144)
    assert image.pixelColor(10, 70).name() == "#0000ff"
    assert image.pixelColor(240, 70).name() == "#ff0000"
    assert buffer.shape == (144, 256, 4)


@pytest.mark.parametrize("size", [(900, 700), (1200, 400), (500, 500)])
def test_the_video_keeps_its_aspect_ratio_as_the_view_resizes(qapp, size):
    view = VideoView()
    view.resize(*size)
    view.set_frame(np.zeros((720, 1280, 3), dtype=np.uint8))

    rect = view.video_rect()
    assert rect.width() / rect.height() == pytest.approx(16 / 9, abs=0.01)
    assert rect.width() <= size[0] and rect.height() <= size[1]
    assert view.image.width() == round(rect.width() * view.devicePixelRatioF())


def test_a_4_by_3_camera_is_not_stretched(qapp):
    view = VideoView()
    view.resize(1000, 500)
    view.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    rect = view.video_rect()
    assert rect.width() / rect.height() == pytest.approx(4 / 3, abs=0.01)


# Frames from the session


def test_a_published_detection_is_stepped_and_shown(harness):
    h = harness()
    add_string(h.strings)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    h.publish(frame=frame)

    assert h.window.frames_shown == 1 and h.engine.frames == 1
    assert frame.any()  # the engine drew the string onto it
    assert h.window.video.image is not None


def test_the_window_closes_after_max_frames(harness):
    h = harness(max_frames=2)
    h.publish(1)
    assert h.closed == []
    h.publish(2)
    assert h.closed == [True]


def test_the_window_closes_when_the_worker_ends(harness):
    h = harness()
    h.session.mailbox.close()
    h.app.processEvents()
    assert h.closed == [True]


def test_an_error_while_stepping_is_kept_for_the_caller(harness):
    h = harness()
    h.session.mailbox.publish(Detection(None, [], 1.0, 0.0, 1))  # no frame: the step fails
    h.app.processEvents()
    assert h.closed == [True]
    assert h.window.error is not None


# Cameras


def test_the_camera_picker_lists_the_cameras_and_selects_the_current_one(harness):
    box = harness(index=1).window.camera_box
    assert [box.itemData(i) for i in range(box.count())] == [0, 1, 2]
    assert box.currentText() == "USB webcam"


def test_a_current_camera_missing_from_the_list_is_added(harness):
    box = harness(cameras=CAMERAS[:1], index=3).window.camera_box
    assert [box.itemText(i) for i in range(box.count())] == ["Built-in", "Camera 3"]
    assert box.currentData() == 3


def test_picking_a_camera_switches_the_session_to_it(harness):
    h = harness()
    h.session.start()

    h.window.camera_box.setCurrentIndex(2)

    assert h.session.index == 2
    assert [camera.index for camera in h.factory.opened] == [2]
    assert h.first.released
    assert h.session.worker.alive
    assert h.window.camera_box.isEnabled()
    assert "Virtual" in h.window.metrics_bar.status_text


def test_a_camera_that_fails_to_open_is_reported_and_the_picker_reverts(harness):
    h = harness(broken={1})
    h.session.start()
    worker = h.session.worker

    h.window.camera_box.setCurrentIndex(1)

    assert h.session.index == 0 and h.session.worker is worker and worker.alive
    assert h.window.camera_box.currentData() == 0
    assert h.window.camera_box.isEnabled()
    assert "Cannot open USB webcam" in h.window.metrics_bar.status_text
    assert h.window.metrics_bar.status_is_error


def test_cameras_open_on_a_background_thread(harness):
    h = harness(background=start_daemon_thread)
    h.session.start()

    h.window.camera_box.setCurrentIndex(1)

    assert not h.window.camera_box.isEnabled()  # until the camera is open and adopted
    assert wait_until(h.app, lambda: h.session.index == 1)
    assert h.window.camera_box.isEnabled()
    assert h.session.worker.alive


# The whole run, through AirStringsApp


def build_app(capture, tracker, frames, cameras=CAMERAS):
    factory = CameraFactory()
    session = CameraSession(0, capture, factory, lambda: tracker)
    engine, _ = make_engine()
    ui = functools.partial(run_window, camera_lister=lambda: cameras)
    return AirStringsApp(session, engine, max_frames=frames, ui=ui), session, factory


def test_a_bounded_windowed_run_shows_frames_and_stops_the_worker(qapp, capsys):
    app, session, _ = build_app(FakeCamera(), FakeTracker(), frames=5)

    assert app.run() == 0
    assert app.frames_processed == 5
    assert not session.worker.alive
    assert detection_threads() == []
    assert "air-strings: frames 5 in" in capsys.readouterr().err


def test_shutdown_after_a_camera_switch_stops_every_worker(qapp):
    captures = []

    def ui(engine, session, max_frames):
        def switch_then_quit(work):
            work()
            captures.append(session.worker)

        window = MainWindow(engine, session, lambda: CAMERAS, max_frames, switch_then_quit)
        window.camera_box.setCurrentIndex(1)
        window.camera_box.setCurrentIndex(2)
        window.close()

    first = FakeCamera()
    factory = CameraFactory()
    session = CameraSession(0, first, factory, lambda: FakeTracker())
    engine, _ = make_engine()
    app = AirStringsApp(session, engine, ui=ui)

    assert app.run() == 0
    assert session.index == 2
    assert len(captures) == 2 and not any(worker.alive for worker in captures)
    assert first.released and all(camera.released for camera in factory.opened)
    assert detection_threads() == []


def test_a_lost_camera_ends_the_windowed_run_with_exit_code_1(qapp, capsys):
    app, session, _ = build_app(FakeCamera(dead=True), FakeTracker(), frames=0)

    assert app.run() == 1
    assert "camera stopped delivering frames" in capsys.readouterr().err
    assert detection_threads() == []


def test_a_tracker_error_surfaces_from_the_windowed_run(qapp):
    app, session, _ = build_app(FakeCamera(), FakeTracker(error=ValueError("model crashed")), frames=0)

    with pytest.raises(ValueError, match="model crashed"):
        app.run()
    assert not session.worker.alive


# Camera names


def test_cameras_are_numbered_by_unique_id_like_opencv():
    devices = [("7626-obs", "OBS Virtual Camera"), ("1FD4-facetime", "FaceTime HD"), ("0x1300-usb", "USB webcam")]
    assert number_like_opencv(devices) == [
        CameraInfo(0, "USB webcam"),
        CameraInfo(1, "FaceTime HD"),
        CameraInfo(2, "OBS Virtual Camera"),
    ]


def test_probing_releases_every_capture_and_skips_the_camera_in_use():
    opened = []

    class Probe:
        def __init__(self, index):
            self.index = index
            self.released = False
            opened.append(self)

        def isOpened(self):  # noqa: N802 - OpenCV naming
            return self.index in (0, 1, 3)

        def release(self):
            self.released = True

    found = probe_cameras(in_use=1, limit=4, opener=Probe)

    assert found == [CameraInfo(0, "Camera 0"), CameraInfo(1, "Camera 1"), CameraInfo(3, "Camera 3")]
    assert [p.index for p in opened] == [0, 2, 3] and all(p.released for p in opened)


def test_the_worker_factory_default_is_a_detection_worker():
    session = CameraSession(0, FakeCamera(), CameraFactory(), lambda: FakeTracker())
    assert isinstance(session.worker, DetectionWorker)
    assert isinstance(session.mailbox, LatestDetection)
