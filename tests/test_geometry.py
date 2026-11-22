import pytest

from air_strings.geometry import crossing_position, project_param, side_of_line

STRING_A = (100.0, 200.0)
STRING_B = (500.0, 200.0)


def test_side_of_line_sign_flips_across_the_line():
    assert side_of_line(STRING_A, STRING_B, (300.0, 150.0)) * side_of_line(STRING_A, STRING_B, (300.0, 250.0)) < 0
    assert side_of_line(STRING_A, STRING_B, (300.0, 200.0)) == 0


def test_crossing_inside_the_segment_triggers_with_position():
    position = crossing_position((200.0, 150.0), (200.0, 260.0), STRING_A, STRING_B)
    assert position == pytest.approx(0.25)


def test_crossing_in_either_direction_triggers():
    assert crossing_position((400.0, 260.0), (400.0, 150.0), STRING_A, STRING_B) == pytest.approx(0.75)


def test_diagonal_crossing_reports_intersection_point():
    # Movement from (250,100) to (350,300) crosses y=200 at x=300 -> u = 0.5.
    assert crossing_position((250.0, 100.0), (350.0, 300.0), STRING_A, STRING_B) == pytest.approx(0.5)


def test_crossing_the_line_outside_the_segment_does_not_trigger():
    assert crossing_position((50.0, 150.0), (50.0, 260.0), STRING_A, STRING_B) is None
    assert crossing_position((620.0, 150.0), (620.0, 260.0), STRING_A, STRING_B) is None


def test_movement_staying_on_one_side_does_not_trigger():
    assert crossing_position((200.0, 150.0), (300.0, 190.0), STRING_A, STRING_B) is None


def test_parallel_movement_does_not_trigger():
    assert crossing_position((150.0, 180.0), (450.0, 180.0), STRING_A, STRING_B) is None
    # Collinear sliding along the string itself is not a pluck either.
    assert crossing_position((150.0, 200.0), (450.0, 200.0), STRING_A, STRING_B) is None


def test_starting_on_the_line_does_not_trigger_but_landing_on_it_does():
    assert crossing_position((300.0, 200.0), (300.0, 260.0), STRING_A, STRING_B) is None
    assert crossing_position((300.0, 150.0), (300.0, 200.0), STRING_A, STRING_B) == pytest.approx(0.5)


def test_short_movement_that_stops_before_the_string_does_not_trigger():
    assert crossing_position((300.0, 100.0), (300.0, 190.0), STRING_A, STRING_B) is None


def test_project_param_is_clamped():
    assert project_param(STRING_A, STRING_B, (300.0, 0.0)) == pytest.approx(0.5)
    assert project_param(STRING_A, STRING_B, (0.0, 0.0)) == 0.0
    assert project_param(STRING_A, STRING_B, (900.0, 0.0)) == 1.0
