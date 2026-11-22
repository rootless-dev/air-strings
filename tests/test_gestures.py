import math

import numpy as np
import pytest

from air_strings.gestures import (
    GRAB_RADIUS_FRACTION,
    PULL_THRESHOLD_VELOCITY,
    DrawGesture,
    GestureController,
    GrabGesture,
    PullGesture,
)
from air_strings.hands import INDEX_TIP, THUMB_TIP, Hand
from air_strings.scales import SCALES, PitchMapping
from air_strings.strings import COMMIT_GRACE_S, MIN_STRING_LENGTH_PX, PLUCK_COOLDOWN_S, StringSet

DIAGONAL = math.hypot(1280, 720)
DT = 1 / 30
GRAB_RADIUS = GRAB_RADIUS_FRACTION * DIAGONAL


def hand(hand_id, tip, pinching=False, previous_tip=None):
    points = np.zeros((21, 2))
    points[:] = tip
    points[INDEX_TIP] = tip
    points[THUMB_TIP] = tip
    return Hand(hand_id, points, pinching, previous_tip)


class Rig:
    """A StringSet and a GestureController driven frame by frame, 30 fps, from t = 1 s."""

    def __init__(self, *segments):
        self.strings = StringSet(PitchMapping(), SCALES[0])
        for start, end in segments:
            self.strings.add(start, end, DIAGONAL, now=0.0)
        self.gestures = GestureController(self.strings, DrawGesture(), GrabGesture(), PullGesture())
        self.now = 1.0
        self.tips = {}

    def step(self, *hands):
        plucks = self.gestures.update(list(hands), self.now, DT, DIAGONAL)
        self.now += DT
        return plucks

    def finger(self, hand_id, tip):
        """An open hand whose previous tip is the one it had in the last call."""
        h = hand(hand_id, tip, False, self.tips.get(hand_id))
        self.tips[hand_id] = tip
        return h

    def wait(self, seconds):
        self.now += seconds

    @property
    def view(self):
        return self.gestures.view()


# --- draw ---------------------------------------------------------------------


def test_draw_gesture_commits_after_release():
    gesture = DrawGesture(min_length=80.0, release_frames=2)
    assert gesture.update([(500.0, 300.0), (100.0, 320.0)]) is None
    assert gesture.pending == ((100.0, 320.0), (500.0, 300.0))
    assert gesture.update([(100.0, 320.0)]) is None  # one-frame dropout tolerated
    assert gesture.pending is not None
    assert gesture.update([]) == ((100.0, 320.0), (500.0, 300.0))
    assert gesture.pending is None
    assert gesture.update([]) is None


def test_draw_gesture_ignores_short_strings():
    gesture = DrawGesture(min_length=80.0, release_frames=1)
    gesture.update([(100.0, 100.0), (130.0, 110.0)])
    assert gesture.update([]) is None


def test_two_hand_pinch_away_from_endpoints_draws_with_a_ruler_from_the_left_end():
    rig = Rig(((100.0, 600.0), (700.0, 600.0)))
    rig.step(hand(0, (900.0, 300.0), True), hand(1, (300.0, 300.0), True))
    view = rig.view
    assert view.pending == ((300.0, 300.0), (900.0, 300.0))
    [stretch] = view.stretches
    assert stretch.anchor == (300.0, 300.0) and stretch.moving == (900.0, 300.0)
    assert stretch.string_id is None
    assert stretch.midi == rig.strings.note_for_length(600.0, DIAGONAL)
    rig.step()
    rig.step()
    assert len(rig.strings.strings) == 2
    assert rig.view.stretches == ()


# --- grab ---------------------------------------------------------------------


def test_one_hand_pinch_near_an_endpoint_grabs_the_nearest_one():
    rig = Rig(((100.0, 300.0), (500.0, 300.0)), ((520.0, 300.0), (900.0, 300.0)))
    first, second = rig.strings.strings
    rig.step(hand(0, (515.0, 302.0), True))  # 15 px from first.end, 5 px from second.start
    [stretch] = rig.view.stretches
    assert stretch.string_id == second.string_id
    assert stretch.anchor == (900.0, 300.0)
    rig.step(hand(0, (450.0, 302.0), True))
    assert second.start == pytest.approx((455.0, 300.0))  # follows, keeping the grab offset
    assert second.end == (900.0, 300.0)
    assert first.end == (500.0, 300.0)


def test_pinch_out_of_range_does_not_grab():
    rig = Rig(((100.0, 300.0), (500.0, 300.0)))
    rig.step(hand(0, (500.0 + GRAB_RADIUS + 5.0, 300.0), True))
    assert rig.view.stretches == ()


def test_grab_takes_priority_over_drawing():
    rig = Rig(((100.0, 300.0), (500.0, 300.0)))
    [string] = rig.strings.strings
    # Both hands start pinching in the same frame; one is next to an endpoint.
    rig.step(hand(0, (505.0, 300.0), True), hand(1, (900.0, 500.0), True))
    view = rig.view
    assert view.pending is None
    assert [s.string_id for s in view.stretches] == [string.string_id]
    # The grabbing hand never counts as one of the two drawing hands.
    for x in (600.0, 650.0, 700.0):
        rig.step(hand(0, (x, 300.0), True), hand(1, (900.0, 500.0), True))
        assert rig.view.pending is None
    rig.step()
    rig.step()
    assert len(rig.strings.strings) == 1
    assert string.end == pytest.approx((695.0, 300.0))


def test_pinch_that_started_elsewhere_does_not_grab_when_moved_onto_an_endpoint():
    rig = Rig(((100.0, 300.0), (500.0, 300.0)))
    rig.step(hand(0, (700.0, 300.0), True))
    rig.step(hand(0, (502.0, 300.0), True))
    assert rig.view.stretches == ()


def test_no_new_grab_while_drawing():
    rig = Rig(((100.0, 300.0), (500.0, 300.0)))
    rig.step(hand(0, (800.0, 100.0), True), hand(1, (1100.0, 100.0), True))
    # Hand 1 drops out for a frame and comes back with a new id next to an endpoint.
    rig.step(hand(0, (800.0, 100.0), True))
    rig.step(hand(0, (800.0, 100.0), True), hand(7, (502.0, 300.0), True))
    view = rig.view
    assert view.pending is not None
    assert all(s.string_id is None for s in view.stretches)


def test_resize_updates_the_note_live_and_keeps_it_after_release():
    rig = Rig(((100.0, 300.0), (900.0, 300.0)))
    [string] = rig.strings.strings
    low = string.midi
    rig.step(hand(0, (900.0, 300.0), True))
    notes = []
    for x in np.linspace(850.0, 350.0, 11):
        rig.step(hand(0, (float(x), 300.0), True))
        notes.append(rig.view.stretches[0].midi)
        assert string.midi == rig.strings.note_for_length(string.length, DIAGONAL)
    assert notes == sorted(notes) and notes[-1] > low  # shorter and shorter: higher and higher
    rig.step()
    rig.step()
    assert rig.view.stretches == ()
    assert string.end == pytest.approx((350.0, 300.0))
    assert string.midi == notes[-1]


def test_resize_is_clamped_to_the_minimum_length():
    rig = Rig(((100.0, 300.0), (500.0, 300.0)))
    [string] = rig.strings.strings
    rig.step(hand(0, (500.0, 300.0), True))
    rig.step(hand(0, (120.0, 300.0), True))
    assert string.length == pytest.approx(MIN_STRING_LENGTH_PX)
    assert string.end == pytest.approx((100.0 + MIN_STRING_LENGTH_PX, 300.0))
    rig.step(hand(0, (100.0, 300.0), True))  # right on the fixed end: keep the direction
    assert string.length == pytest.approx(MIN_STRING_LENGTH_PX)


def test_grab_rides_over_a_one_frame_dropout_and_a_new_hand_id():
    rig = Rig(((100.0, 300.0), (500.0, 300.0)))
    [string] = rig.strings.strings
    rig.step(hand(0, (500.0, 300.0), True))
    rig.step(hand(0, (550.0, 300.0), True))
    rig.step()  # tracking dropout
    assert len(rig.view.stretches) == 1
    rig.step(hand(3, (560.0, 300.0), True))  # back, under a new id
    assert string.end == pytest.approx((560.0, 300.0))
    rig.step(hand(3, (600.0, 300.0), False))  # pinch released...
    assert len(rig.view.stretches) == 1
    rig.step(hand(3, (600.0, 300.0), False))  # ...for two frames: the grab ends
    assert rig.view.stretches == ()
    assert string.end == pytest.approx((560.0, 300.0))


def test_grabbing_hand_does_not_pluck():
    rig = Rig(((100.0, 300.0), (500.0, 300.0)), ((100.0, 400.0), (700.0, 400.0)))
    rig.step(hand(0, (500.0, 300.0), True))
    # Pinch flickers open for a frame while the fingertip sweeps across the other string.
    plucks = rig.step(hand(0, (300.0, 500.0), False, previous_tip=(300.0, 350.0)))
    assert plucks == []
    assert rig.view.bends == ()


# --- pull ---------------------------------------------------------------------


def test_catch_bend_and_release_past_the_threshold_plucks_once():
    rig = Rig(((100.0, 300.0), (700.0, 300.0)))  # 600 px: threshold 60 px
    [string] = rig.strings.strings
    assert rig.step(rig.finger(0, (400.0, 280.0))) == []
    assert rig.step(rig.finger(0, (400.0, 305.0))) == []  # crossed: caught
    [bend] = rig.view.bends
    assert bend.string is string and bend.contact == (400.0, 305.0)
    assert bend.position == pytest.approx(0.5)
    assert rig.step(rig.finger(0, (410.0, 330.0))) == []
    assert rig.step(rig.finger(0, (420.0, 355.0))) == []
    assert rig.view.bends[0].position == pytest.approx(320.0 / 600.0)
    [pluck] = rig.step(rig.finger(0, (420.0, 370.0)))  # 70 px: slips off
    assert pluck.string is string
    assert pluck.velocity == pytest.approx(PULL_THRESHOLD_VELOCITY * 70.0 / 60.0)
    assert pluck.position == pytest.approx(320.0 / 600.0)
    assert pluck.displacement == pytest.approx(70.0)  # down, along +normal
    assert pluck.origin == pytest.approx((420.0, 370.0))
    assert rig.view.bends == ()
    assert string.is_vibrating(rig.now)
    for _ in range(5):
        assert rig.step(rig.finger(0, (420.0, 380.0))) == []


def test_fast_swipe_plucks_in_the_same_frame():
    rig = Rig(((100.0, 300.0), (700.0, 300.0)))
    rig.step(rig.finger(0, (400.0, 200.0)))
    [pluck] = rig.step(rig.finger(0, (400.0, 420.0)))
    assert pluck.position == pytest.approx(0.5)
    assert pluck.velocity == 1.0  # 120 px past a 60 px threshold, and fast
    assert rig.view.bends == ()


def test_strum_across_several_strings_in_one_move_plucks_them_all():
    rig = Rig(*[((100.0, 200.0 + 50 * i), (900.0, 200.0 + 50 * i)) for i in range(4)])
    rig.step(rig.finger(0, (400.0, 150.0)))
    plucks = rig.step(rig.finger(0, (420.0, 500.0)))  # crosses all four, ends 150+ px past the last
    assert sorted(p.string.string_id for p in plucks) == [0, 1, 2, 3]
    assert rig.view.bends == ()


def test_slower_strum_catches_and_releases_each_string_in_turn():
    rig = Rig(*[((100.0, 200.0 + 60 * i), (700.0, 200.0 + 60 * i)) for i in range(3)])
    plucked = []
    for y in np.arange(170.0, 420.0, 20.0):
        plucked += [p.string.string_id for p in rig.step(rig.finger(0, (400.0, float(y))))]
    assert plucked == [0, 1, 2]


def test_backing_out_releases_silently_and_does_not_double_trigger():
    rig = Rig(((100.0, 300.0), (700.0, 300.0)))
    rig.step(rig.finger(0, (400.0, 280.0)))
    rig.step(rig.finger(0, (400.0, 320.0)))  # caught, 20 px
    assert len(rig.view.bends) == 1
    assert rig.step(rig.finger(0, (400.0, 270.0))) == []  # back across the rest line
    assert rig.view.bends == ()  # and that move did not catch it again from the other side
    assert rig.step(rig.finger(0, (400.0, 200.0))) == []
    # A real pluck afterwards fires exactly once.
    plucks = [p for y in (290.0, 310.0, 340.0, 370.0, 380.0) for p in rig.step(rig.finger(0, (400.0, y)))]
    assert len(plucks) == 1


def test_wiggle_after_a_pluck_does_not_retrigger():
    rig = Rig(((100.0, 300.0), (700.0, 300.0)))
    rig.step(rig.finger(0, (400.0, 200.0)))
    assert len(rig.step(rig.finger(0, (400.0, 420.0)))) == 1
    rig.wait(PLUCK_COOLDOWN_S)
    assert rig.step(rig.finger(0, (400.0, 290.0))) == []  # back up across: caught, barely bent
    assert rig.step(rig.finger(0, (400.0, 310.0))) == []  # backs out again: silent
    assert rig.view.bends == ()


def test_sliding_off_the_end_soft_plucks_and_barely_bent_is_silent():
    rig = Rig(((100.0, 300.0), (700.0, 300.0)))
    rig.step(rig.finger(0, (650.0, 280.0)))
    rig.step(rig.finger(0, (660.0, 330.0)))  # caught, 30 px
    [pluck] = rig.step(rig.finger(0, (720.0, 335.0)))  # past the end
    assert pluck.velocity == pytest.approx(PULL_THRESHOLD_VELOCITY * 30.0 / 60.0)
    assert pluck.position == pytest.approx(560.0 / 600.0)

    rig = Rig(((100.0, 300.0), (700.0, 300.0)))
    rig.step(rig.finger(0, (650.0, 290.0)))
    rig.step(rig.finger(0, (660.0, 305.0)))  # caught, 5 px
    assert rig.step(rig.finger(0, (720.0, 305.0))) == []
    assert rig.view.bends == ()


def test_losing_the_hand_or_pinching_lets_the_string_go():
    rig = Rig(((100.0, 300.0), (700.0, 300.0)))
    rig.step(rig.finger(0, (400.0, 280.0)))
    rig.step(rig.finger(0, (400.0, 340.0)))  # caught, 40 px
    [pluck] = rig.step()  # hand lost
    assert pluck.velocity == pytest.approx(PULL_THRESHOLD_VELOCITY * 40.0 / 60.0)

    rig = Rig(((100.0, 300.0), (700.0, 300.0)))
    rig.step(rig.finger(0, (400.0, 280.0)))
    rig.step(rig.finger(0, (400.0, 340.0)))
    [pluck] = rig.step(hand(0, (400.0, 345.0), True, (400.0, 340.0)))  # starts pinching
    assert rig.view.bends == ()


def test_a_caught_string_is_held_by_one_fingertip_only():
    rig = Rig(((100.0, 300.0), (700.0, 300.0)))
    rig.step(rig.finger(0, (300.0, 280.0)), rig.finger(1, (500.0, 280.0)))
    rig.step(rig.finger(0, (300.0, 320.0)), rig.finger(1, (500.0, 280.0)))
    [bend] = rig.view.bends
    rig.step(rig.finger(0, (300.0, 330.0)), rig.finger(1, (500.0, 320.0)))  # finger 1 crosses too
    [bend] = rig.view.bends
    assert bend.contact == (300.0, 330.0)


def test_pinching_hands_grace_and_cooldown_block_catching():
    rig = Rig(((100.0, 300.0), (700.0, 300.0)))
    assert rig.step(hand(0, (400.0, 420.0), True, (400.0, 200.0))) == []  # pinching
    assert rig.view.bends == ()

    strings = StringSet(PitchMapping(), SCALES[0])
    strings.add((100.0, 300.0), (700.0, 300.0), DIAGONAL, now=0.0)
    pull = PullGesture()
    swipe = hand(0, (400.0, 420.0), False, (400.0, 200.0))
    assert pull.update([swipe], strings, COMMIT_GRACE_S / 2, DT, DIAGONAL, set()) == []
    assert len(pull.update([swipe], strings, 1.0, DT, DIAGONAL, set())) == 1
    back = hand(0, (400.0, 200.0), False, (400.0, 420.0))
    assert pull.update([back], strings, 1.0 + PLUCK_COOLDOWN_S / 2, DT, DIAGONAL, set()) == []
    assert len(pull.update([back], strings, 1.0 + PLUCK_COOLDOWN_S * 1.5, DT, DIAGONAL, set())) == 1
    # A string being resized cannot be caught.
    [string] = strings.strings
    assert pull.update([swipe], strings, 5.0, DT, DIAGONAL, {string.string_id}) == []


def test_threshold_has_an_absolute_minimum():
    strings = StringSet(PitchMapping(), SCALES[0])
    short = strings.add((100.0, 300.0), (200.0, 300.0), DIAGONAL, now=0.0)
    long = strings.add((100.0, 400.0), (900.0, 400.0), DIAGONAL, now=0.0)
    pull = PullGesture()
    assert pull.threshold(long, DIAGONAL) == pytest.approx(80.0)
    assert pull.threshold(short, DIAGONAL) > 0.1 * short.length
    assert pull.threshold(short, DIAGONAL) == pytest.approx(0.015 * DIAGONAL)
