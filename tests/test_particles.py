import numpy as np

from air_strings.particles import (
    LIFETIME_S,
    MAX_SPARKS,
    MIN_SPARKS,
    STREAK_TILT_RAD,
    ParticleSystem,
)

START = (100.0, 300.0)
END = (900.0, 300.0)  # a horizontal string, 800 px long
TINT = (200.0, 255.0, 220.0)


def make_system(capacity=500, seed=1):
    return ParticleSystem(capacity=capacity, rng=np.random.default_rng(seed))


def emit(system, position=0.5, strength=1.0, displacement=0.0, start=START, end=END):
    return system.emit(start, end, position, strength, TINT, displacement)


def test_spark_count_scales_with_pluck_strength():
    system = make_system()
    assert emit(system, strength=0.0) == MIN_SPARKS
    assert emit(system, strength=0.5) == round((MIN_SPARKS + MAX_SPARKS) / 2)
    assert emit(system, strength=1.0) == MAX_SPARKS
    assert emit(system, strength=7.0) == MAX_SPARKS  # clamped
    assert system.count == MIN_SPARKS + round((MIN_SPARKS + MAX_SPARKS) / 2) + 2 * MAX_SPARKS


def test_sparks_spawn_along_the_whole_string_denser_near_the_pluck():
    system = make_system()
    for _ in range(4):
        emit(system, position=0.3)
    along = (system.positions[:, 0] - START[0]) / (END[0] - START[0])
    assert along.min() < 0.1 and along.max() > 0.9  # from one end to the other
    near = np.count_nonzero(np.abs(along - 0.3) < 0.1)
    far = np.count_nonzero(np.abs(along - 0.75) < 0.1)  # same width, away from the pluck
    assert near > 2 * far > 0


def test_sparks_start_on_the_vibrating_string():
    system = make_system()
    emit(system, position=0.5, displacement=40.0)
    lateral = np.abs(system.positions[:, 1] - START[1])
    along = (system.positions[:, 0] - START[0]) / (END[0] - START[0])
    reach = 40.0 * np.minimum(along / 0.5, (1 - along) / 0.5)
    assert np.all(lateral <= reach + 6.0)  # within the bend's reach (plus a little jitter)
    assert lateral.max() > 15.0


def test_sparks_burst_to_both_sides_of_the_string():
    system = make_system()
    emit(system)
    system.update(0.1)
    y = system.positions[:, 1]
    assert np.any(y < START[1] - 5) and np.any(y > START[1] + 5)


def test_sparks_slow_down_then_float_up():
    system = make_system()
    emit(system)
    before = system.positions.copy()
    system.update(0.05)
    first_step = np.abs(system.positions - before).mean()
    for _ in range(14):  # 0.75 s: still alive, burst mostly spent
        system.update(0.05)
    before = system.positions.copy()
    system.update(0.05)
    late_step = system.positions - before
    assert np.abs(late_step[:, 0]).mean() < first_step / 3  # drag
    assert np.all(late_step[:, 1] < 0)  # everyone is drifting up by now


def test_streaks_are_short_and_mostly_vertical():
    system = make_system()
    emit(system)
    heads, tails, _ = system.streaks()
    segment = heads - tails
    lengths = np.hypot(segment[:, 0], segment[:, 1])
    assert np.all((lengths > 1.0) & (lengths < 15.0))
    tilt = np.arctan2(np.abs(segment[:, 0]), np.abs(segment[:, 1]))
    assert np.all(tilt <= STREAK_TILT_RAD + 1e-9)


def test_particles_linger_then_age_out():
    system = make_system()
    emit(system)
    for _ in range(int(LIFETIME_S[0] / 0.05) - 1):
        system.update(0.05)
    assert system.count == MAX_SPARKS  # nobody dies before the shortest lifetime
    for _ in range(int(LIFETIME_S[1] / 0.05) + 2):
        system.update(0.05)
    assert system.count == 0


def test_fading_dims_the_streaks():
    system = make_system()
    emit(system)
    _, _, fresh = system.streaks()
    fresh = fresh.copy()
    system.update(0.4)
    _, _, older = system.streaks()
    assert np.all(older < fresh)
    assert np.all(older >= 0)


def test_capacity_is_respected_and_the_newest_survive():
    system = make_system(capacity=100)
    for i in range(10):
        offset = 1000.0 * i
        emit(system, start=(offset, 0.0), end=(offset + 100.0, 0.0))
        assert system.count <= 100
    assert system.count == 100
    # Only sparks of the last two emissions remain: all from x >= 8000.
    assert np.all(system.positions[:, 0] > 7990.0)


def test_clear_removes_everything():
    system = make_system()
    emit(system)
    system.clear()
    assert system.count == 0
    heads, tails, colors = system.streaks()
    assert len(heads) == len(tails) == len(colors) == 0
