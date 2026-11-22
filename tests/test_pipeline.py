import threading
import time

import numpy as np

from air_strings.pipeline import CameraStoppedError, Detection, DetectionWorker, LatestDetection


def detection(sequence):
    return Detection(np.zeros((2, 2, 3), dtype=np.uint8), [], float(sequence), 0.0, sequence)


def test_latest_detection_wins_over_older_ones():
    mailbox = LatestDetection()
    for sequence in (1, 2, 3):
        mailbox.publish(detection(sequence))

    assert mailbox.take(timeout=0).sequence == 3
    assert mailbox.take(timeout=0) is None  # older ones were dropped, not queued


def test_take_waits_for_a_publish_from_another_thread():
    mailbox = LatestDetection()
    publisher = threading.Timer(0.02, mailbox.publish, args=(detection(7),))
    publisher.start()

    assert mailbox.take(timeout=2.0).sequence == 7
    publisher.join()


def test_close_wakes_a_waiting_take_but_keeps_a_pending_detection():
    mailbox = LatestDetection()
    mailbox.publish(detection(1))
    mailbox.close()

    assert mailbox.closed
    assert mailbox.take(timeout=2.0).sequence == 1
    started = time.monotonic()
    assert mailbox.take(timeout=2.0) is None
    assert time.monotonic() - started < 1.0


class SequenceCapture:
    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, 0] = 255  # left column, to see the mirror flip
        return True, frame


class RecordingTracker:
    def __init__(self):
        self.timestamps = []

    def process(self, frame, now):
        self.timestamps.append(now)
        return []


def test_worker_mirrors_frames_with_monotonic_timestamps_and_stops_on_request():
    tracker = RecordingTracker()
    worker = DetectionWorker(SequenceCapture(), tracker, LatestDetection())
    worker.start()
    first = worker.mailbox.take(timeout=2.0)
    time.sleep(0.02)
    latest = worker.mailbox.take(timeout=2.0)

    assert worker.stop()
    assert not worker.alive
    assert worker.error is None
    assert worker.mailbox.closed
    assert latest.sequence > first.sequence and latest.timestamp > first.timestamp
    assert latest.dt > 0.0
    assert (first.frame[:, -1] == 255).all() and (first.frame[:, 0] == 0).all()
    assert all(b >= a for a, b in zip(tracker.timestamps, tracker.timestamps[1:]))


def test_worker_gives_up_after_too_many_failed_reads():
    class FlakyCapture:
        def __init__(self):
            self.reads = 0

        def read(self):
            self.reads += 1
            return False, None

    capture = FlakyCapture()
    worker = DetectionWorker(capture, RecordingTracker(), LatestDetection(), max_failed_reads=5)
    worker.start()

    assert worker.mailbox.take(timeout=2.0) is None
    assert worker.stop()
    assert isinstance(worker.error, CameraStoppedError)
    assert capture.reads == 5


def test_stop_before_start_is_a_no_op():
    worker = DetectionWorker(SequenceCapture(), RecordingTracker(), LatestDetection())
    assert worker.stop()
