"""Goalkeeper analysis: goal geometry, identification, stance, and dives."""

from __future__ import annotations

import numpy as np
import pytest

from soccer_mocap.calibration.pitch import PitchCalibration
from soccer_mocap.exceptions import CalibrationError
from soccer_mocap.goalkeeper import (
    GOAL_WIDTH_M,
    DiveDirection,
    GoalGeometry,
    analyse_stance,
    build_report,
    detect_dives,
    identify_goalkeeper,
)
from soccer_mocap.goalkeeper.stance import find_set_position
from soccer_mocap.types import Track

from .synthetic import make_dive_track, make_pose, make_standing_track

FPS = 60.0
#: 732 px across the mouth makes the scale exactly 100 px/m, which keeps the
#: expected values in these tests readable.
GOAL = GoalGeometry(left_post=(120.0, 300.0), right_post=(852.0, 300.0))


# -- goal geometry ---------------------------------------------------------


def test_scale_comes_from_the_known_goal_width():
    assert GOAL.width_px == pytest.approx(732.0)
    assert GOAL.px_per_m == pytest.approx(100.0)


def test_posts_are_ordered_left_to_right():
    """Sign conventions downstream depend on this, so it is enforced."""
    flipped = GoalGeometry(left_post=(852.0, 300.0), right_post=(120.0, 300.0))
    assert flipped.left_post[0] < flipped.right_post[0]


def test_lateral_offset_is_signed_from_the_centre():
    centre = GOAL.center
    assert GOAL.lateral_offset_m(centre) == pytest.approx(0.0)
    assert GOAL.lateral_offset_m(centre + np.array([100.0, 0.0])) == pytest.approx(1.0)
    assert GOAL.lateral_offset_m(centre - np.array([100.0, 0.0])) == pytest.approx(-1.0)


def test_coverage_fraction_spans_the_posts():
    assert GOAL.coverage_fraction(GOAL.left_post) == pytest.approx(0.0, abs=1e-6)
    assert GOAL.coverage_fraction(GOAL.center) == pytest.approx(0.5)
    assert GOAL.coverage_fraction(GOAL.right_post) == pytest.approx(1.0, abs=1e-6)


def test_distance_off_line_is_positive_into_the_field():
    """Image y grows downward, so 'into the field' is +y here."""
    point = GOAL.center + np.array([0.0, 200.0])
    assert GOAL.distance_off_line_m(point) == pytest.approx(2.0)


def test_crossbar_gives_a_consistency_check():
    good = GoalGeometry((120.0, 300.0), (852.0, 300.0), crossbar_y=300.0 - 244.0)
    assert good.scale_consistency() == pytest.approx(1.0, abs=0.02)

    bad = GoalGeometry((120.0, 300.0), (852.0, 300.0), crossbar_y=300.0 - 100.0)
    assert bad.scale_consistency() < 0.6


def test_coincident_posts_are_rejected():
    with pytest.raises(CalibrationError):
        _ = GoalGeometry((100.0, 100.0), (100.0, 100.0)).px_per_m


def test_from_config_requires_two_points():
    with pytest.raises(CalibrationError, match="exactly two"):
        GoalGeometry.from_config([[1.0, 2.0]])


# -- dive detection --------------------------------------------------------


def test_standing_keeper_produces_no_dives():
    """The false-positive case that matters most in practice."""
    assert detect_dives(make_standing_track(num_frames=120), FPS) == []


def test_short_track_produces_no_dives():
    assert detect_dives(make_standing_track(num_frames=3), FPS) == []


@pytest.mark.parametrize(
    ("direction", "expected"),
    [(1, DiveDirection.RIGHT), (-1, DiveDirection.LEFT)],
)
def test_dive_direction_is_detected(direction, expected):
    track = make_dive_track(direction=direction, fps=FPS)
    dives = detect_dives(track, FPS, goal=GOAL)

    assert len(dives) == 1
    assert dives[0].direction is expected


def test_dive_displacement_is_symmetric_between_directions():
    """Physics is symmetric; a scale bug would show up as asymmetry."""
    right = detect_dives(make_dive_track(direction=1, fps=FPS), FPS, goal=GOAL)[0]
    left = detect_dives(make_dive_track(direction=-1, fps=FPS), FPS, goal=GOAL)[0]

    assert right.lateral_displacement_m == pytest.approx(
        -left.lateral_displacement_m, rel=0.05
    )
    assert right.peak_speed_bh_s == pytest.approx(left.peak_speed_bh_s, rel=0.05)


def test_dive_speed_is_physically_plausible():
    """A human dive is single-digit body-heights per second, not hundreds."""
    dive = detect_dives(make_dive_track(fps=FPS), FPS, goal=GOAL)[0]
    assert 1.0 < dive.peak_speed_bh_s < 15.0


def test_dive_travels_roughly_the_commanded_distance():
    track = make_dive_track(fps=FPS, lateral_travel_px=260.0)
    dive = detect_dives(track, FPS, goal=GOAL)[0]
    # 260 px at 100 px/m = 2.6 m; the COM lags the commanded centre slightly.
    assert dive.lateral_displacement_m == pytest.approx(2.6, rel=0.25)


def test_dive_phases_are_in_chronological_order():
    dive = detect_dives(make_dive_track(fps=FPS), FPS, goal=GOAL)[0]
    phases = dive.phases

    ordered = [
        phases.set_frame,
        phases.preparatory_start,
        phases.takeoff,
        phases.flight_start,
        phases.peak,
        phases.landing,
    ]
    present = [p for p in ordered if p is not None]
    assert present == sorted(present), f"phases out of order: {phases.to_dict()}"


def test_a_high_speed_but_upright_move_is_not_a_dive():
    """A sidestep is fast without leaning; the lean test must reject it."""
    track = Track(track_id=0)
    for frame in range(90):
        # Fast lateral translation, trunk stays vertical throughout.
        track.add(frame, make_pose(com_xy=(200.0 + 6.0 * frame, 240.0), lean_deg=0.0))

    assert detect_dives(track, FPS, lean_threshold_deg=25.0) == []


def test_raising_the_threshold_suppresses_the_dive():
    track = make_dive_track(fps=FPS)
    assert detect_dives(track, FPS, speed_threshold=1.2)
    assert detect_dives(track, FPS, speed_threshold=500.0) == []


# -- stance ----------------------------------------------------------------


def test_stance_measures_a_known_posture():
    """The synthetic figure guarantees knee angle == 180 - flexion."""
    pose = make_pose(height_px=300.0, knee_flexion_deg=30.0, stance_width=1.4)
    stance = analyse_stance(pose, frame_index=0)

    assert stance.is_valid
    assert stance.stance_width_ratio == pytest.approx(1.4, rel=0.10)
    assert stance.mean_knee_angle == pytest.approx(150.0, abs=2.0)
    assert stance.trunk_flexion_deg == pytest.approx(0.0, abs=2.0)


@pytest.mark.parametrize("flexion", [0.0, 20.0, 45.0, 70.0])
def test_knee_angle_tracks_flexion(flexion):
    stance = analyse_stance(make_pose(knee_flexion_deg=flexion))
    assert stance.mean_knee_angle == pytest.approx(180.0 - flexion, abs=2.0)


def test_straight_legs_are_flagged():
    stance = analyse_stance(make_pose(knee_flexion_deg=0.0))
    assert any("straight" in flag for flag in stance.flags)


def test_narrow_stance_is_flagged():
    stance = analyse_stance(make_pose(stance_width=0.4))
    assert any("narrow" in flag for flag in stance.flags)


def test_raised_hands_read_as_above_the_hips():
    low = analyse_stance(make_pose(arm_raise=0.0))
    high = analyse_stance(make_pose(arm_raise=1.0))
    assert high.hand_height_ratio > low.hand_height_ratio


def test_stance_is_invalid_when_the_body_is_not_visible():
    pose = make_pose()
    pose.confidence[:] = 0.0
    assert not analyse_stance(pose).is_valid


def test_set_position_is_found_before_the_dive():
    track = make_dive_track(fps=FPS, dive_start_frame=40)
    frame = find_set_position(track, FPS)

    assert frame is not None
    assert frame < 40


# -- identification --------------------------------------------------------


def test_manual_selection_wins():
    tracks = {0: make_standing_track(track_id=0), 1: make_standing_track(track_id=1)}
    track_id, candidates = identify_goalkeeper(
        tracks, goal=GOAL, method="manual", track_id=1
    )
    assert track_id == 1
    assert candidates[0].reasons == ["selected manually"]


def test_geometry_prefers_the_track_nearer_the_goal():
    near = make_standing_track(track_id=0, com_xy=(486.0, 340.0), num_frames=60)
    far = make_standing_track(track_id=1, com_xy=(486.0, 1200.0), num_frames=60)

    track_id, candidates = identify_goalkeeper({0: near, 1: far}, goal=GOAL)

    assert track_id == 0
    assert candidates[0].score > candidates[1].score


def test_identification_falls_back_to_the_longest_track():
    short = make_standing_track(track_id=0, num_frames=20)
    long = make_standing_track(track_id=1, num_frames=200)

    track_id, candidates = identify_goalkeeper({0: short, 1: long}, goal=None)

    assert track_id == 1
    assert candidates[0].score < 0.5, "a weak inference must report low confidence"


def test_no_tracks_gives_no_goalkeeper():
    assert identify_goalkeeper({}, goal=GOAL) == (None, [])


# -- full report -----------------------------------------------------------


def test_report_contains_positioning_stance_and_dives():
    track = make_dive_track(fps=FPS)
    report = build_report(track, FPS, goal=GOAL)

    assert report.calibrated is True
    assert report.dive_count == 1
    assert report.set_position is not None
    assert report.positioning.frames_analysed > 0
    assert report.positioning.distance_covered_m is not None


def test_report_without_a_goal_warns_and_omits_metres():
    report = build_report(make_dive_track(fps=FPS), FPS, goal=None)

    assert report.calibrated is False
    assert report.positioning.distance_covered_m is None
    assert np.isfinite(report.positioning.distance_covered_bh)
    assert any("No goal was marked" in w for w in report.warnings)


def test_low_frame_rate_is_warned_about():
    track = make_dive_track(fps=30.0, num_frames=60, dive_start_frame=20, dive_frames=10)
    report = build_report(track, 30.0, goal=GOAL)
    if report.dives:
        assert any("fps" in w for w in report.warnings)


def test_report_serialises_to_plain_types():
    import json

    report = build_report(make_dive_track(fps=FPS), FPS, goal=GOAL)
    # Must not raise: the JSON writer depends on this being clean.
    json.dumps(report.to_dict(), default=float)


def test_report_renders_a_text_summary():
    report = build_report(make_dive_track(fps=FPS), FPS, goal=GOAL)
    text = str(report)
    assert "Positioning" in text
    assert "Dives detected" in text


# -- pitch calibration -----------------------------------------------------


def test_homography_round_trips():
    image = [[100.0, 500.0], [900.0, 500.0], [950.0, 300.0], [50.0, 300.0]]
    pitch = [[-20.0, -30.0], [20.0, -30.0], [20.0, 10.0], [-20.0, 10.0]]

    calibration = PitchCalibration.from_points(image, pitch)

    assert calibration.is_reliable
    for image_point, pitch_point in zip(image, pitch, strict=True):
        np.testing.assert_allclose(
            calibration.to_pitch(np.array(image_point)), pitch_point, atol=0.1
        )


def test_homography_needs_four_points():
    with pytest.raises(CalibrationError, match="at least 4"):
        PitchCalibration.from_points([[0.0, 0.0]], [[0.0, 0.0]])


def test_homography_rejects_mismatched_arrays():
    with pytest.raises(CalibrationError):
        PitchCalibration.from_points(
            [[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        )


def test_distance_between_two_points_is_metric():
    image = [[100.0, 500.0], [900.0, 500.0], [950.0, 300.0], [50.0, 300.0]]
    pitch = [[-20.0, -30.0], [20.0, -30.0], [20.0, 10.0], [-20.0, 10.0]]
    calibration = PitchCalibration.from_points(image, pitch)

    distance = calibration.distance_m(np.array([100.0, 500.0]), np.array([900.0, 500.0]))
    assert distance == pytest.approx(40.0, rel=0.02)


def test_goal_width_constant_matches_the_laws():
    assert GOAL_WIDTH_M == pytest.approx(7.32)
