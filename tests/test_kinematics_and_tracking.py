"""Kinematics correctness and tracker robustness.

The kinematics tests check numbers against known ground truth rather than
against whatever the code currently returns, because a scale error here
silently corrupts every metric in the project.
"""

from __future__ import annotations

import numpy as np
import pytest

from soccer_mocap.analysis.kinematics import (
    body_height_px,
    center_of_mass,
    filter_trajectory,
    joint_angle,
    reference_height_px,
    segment_angle_from_vertical,
    velocity,
)
from soccer_mocap.skeleton import J
from soccer_mocap.tracking.iou import IoUTracker, suppress_duplicates
from soccer_mocap.types import BBox, FramePoses, PersonPose

from .synthetic import make_pose

TRUE_HEIGHT_PX = 300.0
#: The synthetic figure's proportions differ slightly from Winter's, so the
#: estimator is expected to land close but not exactly on the true value.
HEIGHT_TOLERANCE = 0.06


# -- body height -----------------------------------------------------------


def test_height_estimate_is_close_to_truth():
    pose = make_pose(height_px=TRUE_HEIGHT_PX)
    estimate = body_height_px(pose.xy, pose.confidence)
    assert estimate == pytest.approx(TRUE_HEIGHT_PX, rel=HEIGHT_TOLERANCE)


@pytest.mark.parametrize("lean_deg", [0, 15, 30, 45, 60, 75, 90])
def test_height_estimate_is_orientation_invariant(lean_deg):
    """A diving keeper is horizontal; vertical extent would collapse to zero.

    This is the bug that produced velocities of thousands of body-heights
    per second, so it gets a test at every angle.
    """
    pose = make_pose(height_px=TRUE_HEIGHT_PX, lean_deg=lean_deg)
    estimate = body_height_px(pose.xy, pose.confidence)
    assert estimate == pytest.approx(TRUE_HEIGHT_PX, rel=HEIGHT_TOLERANCE)


def test_height_estimate_survives_missing_legs():
    """Phone footage cuts feet off constantly; a partial skeleton must still work."""
    pose = make_pose(height_px=TRUE_HEIGHT_PX)
    for joint in (J.LEFT_ANKLE, J.RIGHT_ANKLE, J.LEFT_KNEE, J.RIGHT_KNEE):
        pose.confidence[joint] = 0.0

    estimate = body_height_px(pose.xy, pose.confidence)
    assert np.isfinite(estimate)
    assert estimate == pytest.approx(TRUE_HEIGHT_PX, rel=0.20)


def test_implausibly_small_detection_is_rejected():
    pose = make_pose(height_px=10.0)
    assert np.isnan(body_height_px(pose.xy, pose.confidence))


def test_height_is_nan_when_nothing_is_visible():
    pose = make_pose()
    pose.confidence[:] = 0.0
    assert np.isnan(body_height_px(pose.xy, pose.confidence))


def test_reference_height_is_robust_to_bad_frames():
    """A handful of collapsed estimates must not drag the reference down."""
    heights = [300.0] * 20 + [12.0, 8.0, np.nan]
    assert reference_height_px(heights) == pytest.approx(300.0)


# -- angles ----------------------------------------------------------------


def test_right_angle():
    assert joint_angle(
        np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    ) == pytest.approx(90.0)


def test_straight_limb_is_180_degrees():
    assert joint_angle(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])
    ) == pytest.approx(180.0)


def test_angle_with_missing_joint_is_nan():
    assert np.isnan(
        joint_angle(np.array([np.nan, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    )


def test_segment_angle_sign_follows_screen_x():
    """Positive lean must mean 'toward screen right' in image coordinates."""
    hip = np.array([0.0, 100.0])
    shoulder_right = np.array([10.0, 50.0])   # up and to the right
    shoulder_left = np.array([-10.0, 50.0])

    assert segment_angle_from_vertical(hip, shoulder_right) > 0
    assert segment_angle_from_vertical(hip, shoulder_left) < 0


def test_upright_segment_is_zero_degrees():
    assert segment_angle_from_vertical(
        np.array([0.0, 100.0]), np.array([0.0, 0.0])
    ) == pytest.approx(0.0, abs=1e-6)


# -- centre of mass and velocity -------------------------------------------


def test_com_lies_between_shoulders_and_hips():
    pose = make_pose(com_xy=(320.0, 240.0), height_px=TRUE_HEIGHT_PX)
    com = center_of_mass(pose.xy, pose.confidence)

    shoulder_y = pose.xy[[J.LEFT_SHOULDER, J.RIGHT_SHOULDER], 1].mean()
    hip_y = pose.xy[[J.LEFT_HIP, J.RIGHT_HIP], 1].mean()

    assert shoulder_y < com[1] < hip_y
    assert com[0] == pytest.approx(320.0, abs=2.0)


def test_velocity_matches_known_constant_speed():
    """Ground-truth check: a scale error here corrupts every derived metric."""
    fps = 60.0
    speed_px_s = 600.0
    times = np.arange(120) / fps
    positions = np.stack([speed_px_s * times, np.zeros_like(times)], axis=1)

    result = velocity(positions, fps, filter_cutoff_hz=8.0)

    # Trim the filter's edge transients before comparing.
    assert np.median(result[20:-20, 0]) == pytest.approx(speed_px_s, rel=0.02)


def test_filter_preserves_nan_gaps():
    signal = np.arange(100, dtype=float).reshape(100, 1)
    signal[50] = np.nan

    smoothed = filter_trajectory(signal, 60.0, 6.0)

    assert np.isnan(smoothed[50, 0])
    assert np.isfinite(smoothed[10, 0])


def test_filter_is_a_noop_above_nyquist():
    signal = np.random.default_rng(0).normal(size=(100, 2))
    np.testing.assert_array_equal(filter_trajectory(signal, 30.0, 20.0), signal)


# -- duplicate suppression -------------------------------------------------


def _pose_with_box(box: BBox, confidence: float) -> PersonPose:
    pose = make_pose()
    pose.bbox = box
    pose.confidence[:] = confidence
    return pose


def test_nested_duplicate_is_suppressed():
    """The frame-82 failure: a partial skeleton inside the real one."""
    real = _pose_with_box(BBox(747, 470, 905, 691), 0.9)
    spurious = _pose_with_box(BBox(760, 516, 862, 616), 0.6)

    kept = suppress_duplicates([real, spurious])

    assert len(kept) == 1
    assert kept[0] is real


def test_two_genuinely_separate_people_are_both_kept():
    left = _pose_with_box(BBox(0, 0, 100, 300), 0.9)
    right = _pose_with_box(BBox(400, 0, 500, 300), 0.9)

    assert len(suppress_duplicates([left, right])) == 2


def test_suppression_keeps_the_more_confident_detection():
    weak = _pose_with_box(BBox(0, 0, 100, 100), 0.4)
    strong = _pose_with_box(BBox(2, 2, 102, 102), 0.95)

    kept = suppress_duplicates([weak, strong])

    assert len(kept) == 1
    assert kept[0] is strong


# -- tracking --------------------------------------------------------------


def _frame(index: int, boxes: list[BBox]) -> FramePoses:
    return FramePoses(
        frame_index=index,
        timestamp=index / 60.0,
        people=[_pose_with_box(b, 0.9) for b in boxes],
    )


def test_track_id_is_stable_for_steady_motion():
    tracker = IoUTracker(min_hits=1)
    ids = []
    for i in range(20):
        frame = _frame(i, [BBox(100 + 4 * i, 100, 200 + 4 * i, 400)])
        tracker.update(frame)
        ids.append(frame.people[0].track_id)

    assert len(set(ids)) == 1


def test_track_survives_a_box_flipping_aspect_ratio():
    """A keeper going horizontal must not be handed a new id mid-dive."""
    tracker = IoUTracker(min_hits=1)
    ids = []

    # Upright: tall and narrow.
    for i in range(10):
        frame = _frame(i, [BBox(300, 100, 400, 400)])
        tracker.update(frame)
        ids.append(frame.people[0].track_id)

    # Horizontal: short and wide, centred on roughly the same point.
    for i in range(10, 20):
        frame = _frame(i, [BBox(250, 300, 500, 400)])
        tracker.update(frame)
        ids.append(frame.people[0].track_id)

    assert len(set(ids)) == 1, "the dive split the track in two"


def test_two_people_keep_separate_ids():
    tracker = IoUTracker(min_hits=1)
    first, second = [], []
    for i in range(15):
        frame = _frame(i, [BBox(0, 0, 100, 300), BBox(600, 0, 700, 300)])
        tracker.update(frame)
        first.append(frame.people[0].track_id)
        second.append(frame.people[1].track_id)

    assert len(set(first)) == 1
    assert len(set(second)) == 1
    assert first[0] != second[0]


def test_track_is_dropped_after_max_age():
    tracker = IoUTracker(min_hits=1, max_age=3)
    tracker.update(_frame(0, [BBox(0, 0, 100, 300)]))
    assert tracker.active_tracks == 1

    for i in range(1, 8):
        tracker.update(_frame(i, []))

    assert tracker.active_tracks == 0


def test_far_apart_detection_starts_a_new_track():
    """The centre-distance fallback must not link unrelated people."""
    tracker = IoUTracker(min_hits=1)
    first = _frame(0, [BBox(0, 0, 100, 300)])
    tracker.update(first)

    second = _frame(1, [BBox(1500, 600, 1600, 900)])
    tracker.update(second)

    assert first.people[0].track_id != second.people[0].track_id
