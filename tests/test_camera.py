"""Switching cameras: the new one is opened first, then the old worker, capture and tracker go."""

from fakes import CameraFactory, FakeCamera, TrackerFactory, detection_threads

from air_strings.camera import CameraSession


def make_session(broken=()):
    factory = CameraFactory(broken)
    trackers = TrackerFactory()
    first = FakeCamera(0)
    session = CameraSession(0, first, factory, trackers)
    return session, first, factory, trackers


def test_switch_replaces_capture_tracker_and_worker():
    session, first, factory, trackers = make_session()
    session.start()
    old_worker = session.worker

    assert session.switch(2)

    assert session.index == 2
    assert [camera.index for camera in factory.opened] == [2]
    assert first.released and trackers.made[0].closed
    assert not old_worker.alive
    assert session.worker is not old_worker and session.worker.alive
    assert session.mailbox.take(timeout=2.0) is not None  # the new camera delivers
    session.stop()
    assert factory.opened[0].released and trackers.made[1].closed
    assert detection_threads() == []


def test_a_camera_that_does_not_open_leaves_the_running_one_alone():
    session, first, factory, trackers = make_session(broken={1})
    session.start()
    worker = session.worker

    assert not session.switch(1)

    assert session.index == 0 and session.worker is worker and worker.alive
    assert not first.released and not trackers.made[0].closed
    session.stop()
    assert first.released and not worker.alive


def test_switching_to_the_current_camera_does_nothing():
    session, _, factory, _ = make_session()
    assert session.switch(0)
    assert factory.opened == []


def test_the_listener_follows_the_session_across_switches():
    session, *_ = make_session()
    calls = []
    session.set_listener(lambda: calls.append(1))
    session.start()
    session.switch(1)
    assert session.mailbox.take(timeout=2.0) is not None
    assert calls  # the new worker's publishes reach the same listener
    session.stop()


def test_stop_is_idempotent_and_a_late_camera_is_released():
    session, first, factory, _ = make_session()
    session.start()
    session.stop()
    session.stop()
    late = FakeCamera(3)

    session.switch_to(3, late)  # e.g. a camera that finished opening after shutdown

    assert late.released and first.released
    assert detection_threads() == []
