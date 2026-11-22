"""Hand gestures on the strings, as per-frame state machines.

- Draw: two pinching hands stretch a pending string; releasing a pinch commits it.
- Grab: one hand pinching next to an endpoint of a string drags that endpoint
  (the string is re-tuned live); releasing the pinch leaves it there.
- Pull: an open hand's index fingertip pushing through a string catches it; the
  string bends through the fingertip and is plucked when it slips off.

`GestureController` runs the three in priority order (grab, draw, pull) and
applies their results to the `StringSet`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from air_strings.geometry import Point, distance, side_of_line, unit_normal
from air_strings.hands import Hand
from air_strings.strings import MIN_STRING_LENGTH_PX, AirString, StringSet

# A one-hand pinch this close to an endpoint grabs it, as a fraction of the frame
# diagonal (about 40 px at 1280x720).
GRAB_RADIUS_FRACTION = 0.027
# Consecutive frames without the pinch before a draw or a grab counts as released;
# rides over one-frame tracking dropouts and pinch flicker.
RELEASE_FRAMES = 2

# A caught string slips off the fingertip once deflected this far: a fraction of
# its length, but never less than PULL_MIN_RELEASE_FRACTION of the frame diagonal
# (about 22 px at 1280x720), so short strings do not fire on tracking jitter.
PULL_RELEASE_FRACTION = 0.10
PULL_MIN_RELEASE_FRACTION = 0.015
# Velocity of a pluck that slips off exactly at the threshold; a fast move that
# overshoots it in one frame is louder, up to 1.
PULL_THRESHOLD_VELOCITY = 0.6
# A string let go before the threshold (the fingertip slid off one of its ends,
# the hand was lost or started pinching) still sounds, softly, if it was
# deflected at least this fraction of the threshold.
PULL_SOFT_MIN_RATIO = 0.25
# The bent shape the vibration starts from is capped at this multiple of the threshold.
PULL_VISUAL_CAP_RATIO = 1.5
# Fingertip speed, in frame diagonals per second, that counts as a full-force pluck.
FULL_VELOCITY_DIAGONALS_PER_S = 2.0


@dataclass(frozen=True)
class Pluck:
    string: AirString
    position: float  # 0..1 along the string, from its start
    velocity: float  # 0..1 loudness
    displacement: float  # signed bend at release, in pixels along the string normal

    @property
    def origin(self) -> Point:
        """The bent point of the string at release."""
        s = self.string
        normal = unit_normal(s.start, s.end)
        return (
            s.start[0] + (s.end[0] - s.start[0]) * self.position + normal[0] * self.displacement,
            s.start[1] + (s.end[1] - s.start[1]) * self.position + normal[1] * self.displacement,
        )


@dataclass(frozen=True)
class Stretch:
    """A string being stretched; its ruler is measured from `anchor` towards `moving`."""

    anchor: Point
    moving: Point
    midi: int  # the note for the current length
    string_id: int | None = None  # the string being resized; None while drawing a new one


@dataclass(frozen=True)
class Bend:
    """A string caught on a fingertip: drawn bent through `contact`."""

    string: AirString
    position: float  # 0..1 along the string
    contact: Point


@dataclass(frozen=True)
class GestureView:
    pending: tuple[Point, Point] | None
    stretches: tuple[Stretch, ...]
    bends: tuple[Bend, ...]


def _string_coordinates(string: AirString, p: Point) -> tuple[float, float]:
    """(parameter along the string, unclamped; signed distance in pixels along its normal)."""
    dx, dy = string.end[0] - string.start[0], string.end[1] - string.start[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return 0.0, 0.0
    px, py = p[0] - string.start[0], p[1] - string.start[1]
    return (px * dx + py * dy) / length_sq, (px * -dy + py * dx) / math.sqrt(length_sq)


class DrawGesture:
    """Two pinching hands stretch a pending string; releasing a pinch commits it."""

    def __init__(self, min_length: float = MIN_STRING_LENGTH_PX, release_frames: int = RELEASE_FRAMES) -> None:
        self._min_length = min_length
        self._release_frames = release_frames
        self._missing = 0
        self.pending: tuple[Point, Point] | None = None

    def update(self, pinch_points: list[Point]) -> tuple[Point, Point] | None:
        """Feed the pinch midpoints of the pinching hands that are free to draw.

        Returns the segment to commit once a two-hand pinch has ended (for
        `release_frames` consecutive frames, which rides over one-frame
        tracking dropouts) and the string is long enough, otherwise None.
        The segment is sorted, so its first point is the leftmost on screen.
        """
        if len(pinch_points) >= 2:
            a, b = sorted(pinch_points[:2])
            self.pending = (a, b)
            self._missing = 0
            return None
        if self.pending is None:
            return None
        self._missing += 1
        if self._missing < self._release_frames:
            return None
        finished, self.pending = self.pending, None
        self._missing = 0
        if distance(*finished) >= self._min_length:
            return finished
        return None


@dataclass(eq=False)
class _Grab:
    string: AirString
    endpoint: int  # 0: the string's start follows the pinch, 1: its end does
    hand_id: int
    offset: tuple[float, float]  # endpoint minus pinch point at grab time, so it does not jump
    pinch: Point  # last seen pinch point
    missing: int = 0

    @property
    def fixed(self) -> Point:
        return self.string.end if self.endpoint == 0 else self.string.start

    @property
    def moving(self) -> Point:
        return self.string.start if self.endpoint == 0 else self.string.end


class GrabGesture:
    """One pinching hand next to an endpoint drags it; the other endpoint stays put.

    A grab only starts on the frame a pinch begins (so a pinch moved onto an
    endpoint, or a drawing hand passing by one, does not grab), picks the
    nearest free endpoint within the radius, and ends after `release_frames`
    frames without the pinch. If the tracker loses the hand for a frame and it
    comes back under a new id, still pinching near the endpoint, the grab goes on.
    """

    def __init__(
        self,
        radius_fraction: float = GRAB_RADIUS_FRACTION,
        min_length: float = MIN_STRING_LENGTH_PX,
        release_frames: int = RELEASE_FRAMES,
    ) -> None:
        self._radius_fraction = radius_fraction
        self._min_length = min_length
        self._release_frames = release_frames
        self._grabs: list[_Grab] = []
        self._was_pinching: set[int] = set()

    def grabbed_ids(self) -> set[int]:
        return {grab.string.string_id for grab in self._grabs}

    def stretches(self) -> list[Stretch]:
        return [Stretch(g.fixed, g.moving, g.string.midi, g.string.string_id) for g in self._grabs]

    def update(
        self, hands: list[Hand], strings: StringSet, now: float, frame_diagonal: float, allow_new: bool
    ) -> set[int]:
        """Advance the grabs; returns the ids of the hands that are grabbing (or may come back to)."""
        pinching = {h.hand_id: h for h in hands if h.pinching}
        radius = self._radius_fraction * frame_diagonal
        claimed = {grab.hand_id for grab in self._grabs}
        kept: list[_Grab] = []
        for grab in self._grabs:
            if grab.string not in strings:
                continue  # cleared or replaced by a newer string
            hand = pinching.get(grab.hand_id)
            if hand is None:
                hand = self._nearest_hand(pinching, claimed, grab.pinch, radius)
                if hand is not None:
                    claimed.add(hand.hand_id)
                    grab.hand_id = hand.hand_id
            if hand is None:
                grab.missing += 1
                if grab.missing >= self._release_frames:
                    strings.end_resize(grab.string, now)
                    continue
                kept.append(grab)
                continue
            grab.missing = 0
            grab.pinch = hand.pinch_point
            self._move(grab, strings, frame_diagonal)
            kept.append(grab)
        self._grabs = kept

        if allow_new:
            for hand in pinching.values():
                if hand.hand_id in claimed or hand.hand_id in self._was_pinching:
                    continue
                grab = self._start(hand, strings, radius)
                if grab is not None:
                    claimed.add(hand.hand_id)
                    self._grabs.append(grab)
        self._was_pinching = set(pinching)
        return {grab.hand_id for grab in self._grabs}

    def _start(self, hand: Hand, strings: StringSet, radius: float) -> _Grab | None:
        grabbed = self.grabbed_ids()
        pinch = hand.pinch_point
        best: _Grab | None = None
        best_distance = radius
        for string in strings.strings:
            if string.string_id in grabbed:
                continue
            for endpoint, point in ((0, string.start), (1, string.end)):
                d = distance(point, pinch)
                if d <= best_distance:
                    offset = (point[0] - pinch[0], point[1] - pinch[1])
                    best, best_distance = _Grab(string, endpoint, hand.hand_id, offset, pinch), d
        return best

    @staticmethod
    def _nearest_hand(pinching: dict[int, Hand], claimed: set[int], point: Point, radius: float) -> Hand | None:
        free = [h for h in pinching.values() if h.hand_id not in claimed and distance(h.pinch_point, point) <= radius]
        return min(free, key=lambda h: distance(h.pinch_point, point), default=None)

    def _move(self, grab: _Grab, strings: StringSet, frame_diagonal: float) -> None:
        target = (grab.pinch[0] + grab.offset[0], grab.pinch[1] + grab.offset[1])
        fixed = grab.fixed
        length = distance(fixed, target)
        if length < self._min_length:
            # Clamp to the minimum length, keeping the direction (or the current
            # one, if the pinch sits right on the fixed end).
            towards = target if length > 1e-6 else grab.moving
            d = distance(fixed, towards)
            if d < 1e-6:
                return
            scale = self._min_length / d
            target = (fixed[0] + (towards[0] - fixed[0]) * scale, fixed[1] + (towards[1] - fixed[1]) * scale)
        strings.move_endpoint(grab.string, grab.endpoint, target, frame_diagonal)


@dataclass(eq=False)
class _Hold:
    string: AirString
    hand_id: int
    side: float  # +1 or -1: the side of the rest line the fingertip came from
    position: float  # contact point along the string, 0..1; follows the fingertip
    contact: Point
    deflection: float  # pixels past the rest line, in the pull direction


class PullGesture:
    """Catch-and-release plucking with an open hand's index fingertip.

    A fingertip crossing a string catches it; the string then bends through the
    fingertip (the contact point follows the fingertip's projection on the
    string). It is released:

    - past the threshold deflection: a pluck, louder the further past it is;
      when a single frame's move crosses and already ends past the threshold
      (a fast swipe or strum), the pluck fires in that same frame, at the
      crossing point, and the fingertip's speed counts too;
    - when the fingertip slides off either end of the string, the hand is lost
      or it starts pinching: a soft pluck from the last deflection, or silence
      if it was barely bent;
    - when the fingertip backs out across the rest line: silently (the string
      just returns to rest), and that move does not catch it again.

    A string is held by one fingertip at a time. Strings in cooldown, in their
    post-commit grace period or being resized cannot be caught.
    """

    def __init__(
        self,
        release_fraction: float = PULL_RELEASE_FRACTION,
        min_release_fraction: float = PULL_MIN_RELEASE_FRACTION,
        full_velocity_diagonals_per_s: float = FULL_VELOCITY_DIAGONALS_PER_S,
    ) -> None:
        self._release_fraction = release_fraction
        self._min_release_fraction = min_release_fraction
        self._full_velocity = full_velocity_diagonals_per_s
        self._holds: list[_Hold] = []

    def threshold(self, string: AirString, frame_diagonal: float) -> float:
        return max(self._release_fraction * string.length, self._min_release_fraction * frame_diagonal)

    def bends(self) -> list[Bend]:
        return [Bend(h.string, h.position, h.contact) for h in self._holds]

    def update(
        self,
        hands: list[Hand],
        strings: StringSet,
        now: float,
        dt: float,
        frame_diagonal: float,
        blocked: set[int],
    ) -> list[Pluck]:
        """Advance the holds and catch new strings; returns the plucks released this frame.

        `hands` are the hands free to pluck; `blocked` holds the ids of strings
        that must not be caught (being resized).
        """
        by_id = {h.hand_id: h for h in hands}
        plucks: list[Pluck] = []
        done: set[tuple[int, int]] = set()  # (hand, string) pairs already handled this frame
        kept: list[_Hold] = []
        for hold in self._holds:
            string = hold.string
            if string not in strings or string.string_id in blocked:
                continue  # removed or grabbed: let go without a sound
            hand = by_id.get(hold.hand_id)
            if hand is None or hand.pinching:
                self._soft_release(hold, now, frame_diagonal, plucks)
                continue
            done.add((hand.hand_id, string.string_id))
            along, perp = _string_coordinates(string, hand.tip)
            deflection = -hold.side * perp
            threshold = self.threshold(string, frame_diagonal)
            if deflection < 0:
                continue  # backed out across the rest line: silent
            if not 0.0 <= along <= 1.0:
                self._soft_release(hold, now, frame_diagonal, plucks)
                continue
            if deflection >= threshold:
                velocity = self._slip_velocity(hand, deflection, threshold, dt, frame_diagonal)
                plucks.append(self._release(string, along, deflection, hold.side, threshold, velocity, now))
                continue
            hold.position, hold.contact, hold.deflection = along, hand.tip, deflection
            kept.append(hold)
        self._holds = kept

        held = {hold.string.string_id for hold in kept}
        for hand in hands:
            if hand.pinching or hand.previous_tip is None:
                continue
            for crossing in strings.find_crossings(hand.previous_tip, hand.tip, now):
                string = crossing.string
                if string.string_id in held or string.string_id in blocked:
                    continue
                if (hand.hand_id, string.string_id) in done:
                    continue
                side = math.copysign(1.0, side_of_line(string.start, string.end, hand.previous_tip))
                along, perp = _string_coordinates(string, hand.tip)
                deflection = -side * perp
                threshold = self.threshold(string, frame_diagonal)
                if deflection >= threshold:
                    velocity = self._slip_velocity(hand, deflection, threshold, dt, frame_diagonal)
                    plucks.append(self._release(string, crossing.position, deflection, side, threshold, velocity, now))
                elif 0.0 <= along <= 1.0:
                    self._holds.append(_Hold(string, hand.hand_id, side, along, hand.tip, max(0.0, deflection)))
                    held.add(string.string_id)
        return plucks

    @staticmethod
    def _deflection_velocity(deflection: float, threshold: float) -> float:
        return min(1.0, PULL_THRESHOLD_VELOCITY * deflection / threshold)

    def _slip_velocity(self, hand: Hand, deflection: float, threshold: float, dt: float, frame_diagonal: float) -> float:
        """Loudness of a string slipping off past the threshold: deflection, or fingertip speed if higher."""
        speed_velocity = 0.0
        if dt > 0 and hand.previous_tip is not None:
            speed = distance(hand.previous_tip, hand.tip) / dt / frame_diagonal
            speed_velocity = min(1.0, speed / self._full_velocity)
        return max(self._deflection_velocity(deflection, threshold), speed_velocity)

    def _soft_release(self, hold: _Hold, now: float, frame_diagonal: float, plucks: list[Pluck]) -> None:
        threshold = self.threshold(hold.string, frame_diagonal)
        if hold.deflection < PULL_SOFT_MIN_RATIO * threshold:
            return
        velocity = self._deflection_velocity(hold.deflection, threshold)
        position = min(1.0, max(0.0, hold.position))
        plucks.append(self._release(hold.string, position, hold.deflection, hold.side, threshold, velocity, now))

    @staticmethod
    def _release(
        string: AirString,
        position: float,
        deflection: float,
        side: float,
        threshold: float,
        velocity: float,
        now: float,
    ) -> Pluck:
        # The fingertip came from `side` and pushed the string to the other one.
        displacement = -side * min(deflection, PULL_VISUAL_CAP_RATIO * threshold)
        string.pluck(now, position, velocity, displacement)
        return Pluck(string, position, velocity, displacement)


class GestureController:
    """Runs grab, draw and pull each frame, in that priority order, on one `StringSet`.

    A hand that is grabbing an endpoint neither counts as a drawing hand nor plucks.
    New grabs are not started while a two-hand string is being drawn.
    """

    def __init__(self, strings: StringSet, draw: DrawGesture, grab: GrabGesture, pull: PullGesture) -> None:
        self._strings = strings
        self._draw = draw
        self._grab = grab
        self._pull = pull
        self._frame_diagonal = 1.0

    def update(self, hands: list[Hand], now: float, dt: float, frame_diagonal: float) -> list[Pluck]:
        self._frame_diagonal = frame_diagonal
        grabbing = self._grab.update(hands, self._strings, now, frame_diagonal, allow_new=self._draw.pending is None)
        free_pinches = [h.pinch_point for h in hands if h.pinching and h.hand_id not in grabbing]
        committed = self._draw.update(free_pinches)
        if committed is not None:
            self._strings.add(committed[0], committed[1], frame_diagonal, now)
        free_hands = [h for h in hands if h.hand_id not in grabbing]
        return self._pull.update(free_hands, self._strings, now, dt, frame_diagonal, self._grab.grabbed_ids())

    def view(self) -> GestureView:
        stretches = list(self._grab.stretches())
        pending = self._draw.pending
        if pending is not None:
            a, b = pending  # sorted: the ruler is measured from the left end
            stretches.append(Stretch(a, b, self._strings.note_for_length(distance(a, b), self._frame_diagonal)))
        return GestureView(pending, tuple(stretches), tuple(self._pull.bends()))
