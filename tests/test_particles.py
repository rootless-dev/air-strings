import numpy as np

from air_strings.particles import LIFETIME_S, MAX_SPARKS, MIN_SPARKS, ParticleSystem

ORIGIN = (400.0, 300.0)
TANGENT = (1.0, 0.0)
NORMAL = (0.0, 1.0)
TINT = (200.0, 255.0, 220.0)


def make_system(capacity=400):
    return ParticleSystem(capacity=capacity, rng=np.random.default_rng(1))


def test_spark_count_scales_with_pluck_strength():
    system = make_system()
    assert system.emit(ORIGIN, TANGENT, NORMAL, 0.0, TINT) == MIN_SPARKS
    assert system.emit(ORIGIN, TANGENT, NORMAL, 0.5, TINT) == (MIN_SPARKS + MAX_SPARKS) // 2
    assert system.emit(ORIGIN, TANGENT, NORMAL, 1.0, TINT) == MAX_SPARKS
    assert system.emit(ORIGIN, TANGENT, NORMAL, 7.0, TINT) == MAX_SPARKS  # clamped
    assert system.count == MIN_SPARKS + (MIN_SPARKS + MAX_SPARKS) // 2 + 2 * MAX_SPARKS


def test_sparks_fly_to_both_sides_of_the_string():
    system = make_system()
    system.emit(ORIGIN, TANGENT, NORMAL, 1.0, TINT)
    system.update(0.02)
    heads, _, _ = system.streaks()
    assert np.any(heads[:, 1] < ORIGIN[1] - 2) and np.any(heads[:, 1] > ORIGIN[1] + 2)


def test_particles_age_out():
    system = make_system()
    system.emit(ORIGIN, TANGENT, NORMAL, 1.0, TINT)
    for _ in range(int(LIFETIME_S[0] / 0.05) - 1):
        system.update(0.05)
    assert system.count == MAX_SPARKS  # nobody dies before the shortest lifetime
    for _ in range(int(LIFETIME_S[1] / 0.05) + 2):
        system.update(0.05)
    assert system.count == 0


def test_fading_dims_the_streaks():
    system = make_system()
    system.emit(ORIGIN, TANGENT, NORMAL, 1.0, TINT)
    _, _, fresh = system.streaks()
    fresh = fresh.copy()
    system.update(0.4)
    _, _, older = system.streaks()
    assert np.all(older < fresh)
    assert np.all(older >= 0)


def test_capacity_is_respected_and_the_newest_survive():
    system = make_system(capacity=100)
    for i in range(10):
        system.emit((10.0 * i, 0.0), TANGENT, NORMAL, 1.0, TINT)
        assert system.count <= 100
    assert system.count == 100
    heads, _, _ = system.streaks()
    # The last 2.5 emissions (40 sparks each) remain: all from x >= 70.
    assert np.all(heads[:, 0] > 60.0)


def test_gravity_and_drag():
    system = make_system()
    system.emit(ORIGIN, TANGENT, NORMAL, 1.0, TINT)
    heads, tails, _ = system.streaks()
    velocity_before = (heads - tails).copy()
    system.update(0.1)
    heads, tails, _ = system.streaks()
    velocity_after = heads - tails
    assert np.mean(velocity_after[:, 1] - velocity_before[:, 1]) > 0  # pulled down (+y)
    assert np.mean(np.abs(velocity_after[:, 0])) < np.mean(np.abs(velocity_before[:, 0]))  # slowed sideways


def test_clear_removes_everything():
    system = make_system()
    system.emit(ORIGIN, TANGENT, NORMAL, 1.0, TINT)
    system.clear()
    assert system.count == 0
    heads, tails, colors = system.streaks()
    assert len(heads) == len(tails) == len(colors) == 0
