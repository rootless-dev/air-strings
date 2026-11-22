import numpy as np

from air_strings.hands import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST, OneEuroFilter, PinchDetector


def hand_with_gap(gap: float) -> np.ndarray:
    points = np.zeros((21, 2))
    points[WRIST] = (0.0, 0.0)
    points[MIDDLE_MCP] = (0.0, -100.0)  # hand size 100 px
    points[THUMB_TIP] = (50.0, -120.0)
    points[INDEX_TIP] = (50.0 + gap, -120.0)
    return points


def test_pinch_hysteresis_does_not_flicker():
    detector = PinchDetector(on_ratio=0.28, off_ratio=0.42)
    assert detector.update(hand_with_gap(60.0)) is False
    assert detector.update(hand_with_gap(35.0)) is False  # between thresholds, not yet on
    assert detector.update(hand_with_gap(20.0)) is True
    assert detector.update(hand_with_gap(35.0)) is True  # between thresholds, stays on
    assert detector.update(hand_with_gap(45.0)) is False


def test_one_euro_filter_reduces_jitter_but_follows_motion():
    rng = np.random.default_rng(0)
    smoothing = OneEuroFilter()
    still = np.array([[300.0, 200.0]])
    outputs = [smoothing(still + rng.normal(0, 3.0, still.shape), i / 30) for i in range(60)]
    jitter = np.std(np.array(outputs[30:])[:, 0, 0])
    assert jitter < 1.5

    fast = OneEuroFilter()
    for i in range(30):
        value = fast(np.array([[300.0 + 60.0 * i, 200.0]]), i / 30)
    assert abs(value[0, 0] - (300.0 + 60.0 * 29)) < 60.0  # lags well under 2 frames
