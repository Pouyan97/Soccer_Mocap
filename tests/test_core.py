"""Skeleton, types, config and device-preset behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from soccer_mocap import skeleton
from soccer_mocap.config import PipelineConfig
from soccer_mocap.devices.presets import (
    ACTION_CAM,
    BALANCED,
    infer_preset,
    resolve_profile,
)
from soccer_mocap.exceptions import ConfigError
from soccer_mocap.skeleton import CANONICAL, MEDIAPIPE33, J, to_canonical
from soccer_mocap.types import BBox, FramePoses, PersonPose, Track, VideoMetadata

from .synthetic import make_pose, make_video_metadata

# -- skeleton --------------------------------------------------------------


def test_canonical_layout_is_coco17():
    assert len(CANONICAL) == 17
    assert CANONICAL.joint_names[J.LEFT_KNEE] == "left_knee"
    assert CANONICAL.joint_names[J.RIGHT_ANKLE] == "right_ankle"


def test_mediapipe_remaps_to_canonical_by_name():
    """Each canonical joint must pick up the correctly-named source column."""
    xy = np.arange(33 * 2, dtype=float).reshape(33, 2)
    confidence = np.linspace(0.1, 1.0, 33)

    out_xy, out_conf = to_canonical(xy, confidence, MEDIAPIPE33)

    assert out_xy.shape == (17, 2)
    for canonical_index, name in enumerate(CANONICAL.joint_names):
        source = MEDIAPIPE33.index(name)
        np.testing.assert_array_equal(out_xy[canonical_index], xy[source])
        assert out_conf[canonical_index] == pytest.approx(confidence[source])


def test_to_canonical_is_identity_for_canonical_input():
    xy = np.zeros((17, 2))
    out_xy, _ = to_canonical(xy, np.ones(17), CANONICAL)
    np.testing.assert_array_equal(out_xy, xy)


def test_unknown_layout_raises():
    with pytest.raises(KeyError):
        skeleton.get_layout("not_a_layout")


# -- bounding boxes --------------------------------------------------------


def test_iou_of_identical_boxes_is_one():
    box = BBox(0, 0, 10, 10)
    assert box.iou(BBox(0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert BBox(0, 0, 10, 10).iou(BBox(20, 20, 30, 30)) == 0.0


def test_containment_catches_nested_box_that_iou_misses():
    """The exact case that split a goalkeeper's track in two."""
    outer = BBox(0, 0, 100, 100)
    inner = BBox(40, 40, 60, 60)

    assert outer.iou(inner) < 0.1        # IoU says "different objects"
    assert outer.containment(inner) == pytest.approx(1.0)  # containment says otherwise


def test_bbox_from_keypoints_ignores_nan():
    xy = np.array([[10.0, 20.0], [np.nan, np.nan], [30.0, 40.0]])
    box = BBox.from_keypoints(xy)
    assert (box.x1, box.y1, box.x2, box.y2) == (10.0, 20.0, 30.0, 40.0)


# -- poses and tracks ------------------------------------------------------


def test_person_pose_rejects_mismatched_confidence():
    with pytest.raises(ValueError, match="mismatch"):
        PersonPose(xy=np.zeros((17, 2)), confidence=np.ones(5))


def test_visible_requires_both_confidence_and_finite_coordinates():
    xy = np.zeros((17, 2))
    xy[J.NOSE] = [np.nan, np.nan]
    confidence = np.ones(17)

    pose = PersonPose(xy=xy, confidence=confidence)
    visible = pose.visible(0.3)

    assert not visible[J.NOSE]
    assert visible[J.LEFT_KNEE]


def test_track_stack_leaves_nan_for_missing_frames():
    """Gaps must stay as gaps, so the time axis remains uniform."""
    track = Track(track_id=0)
    track.add(0, make_pose())
    track.add(3, make_pose())

    frames, xy, _ = track.stack()

    assert list(frames) == [0, 1, 2, 3]
    assert np.isnan(xy[1]).all()
    assert np.isfinite(xy[0]).all()


def test_build_tracks_groups_by_track_id():
    poses = [make_pose(), make_pose()]
    poses[0].track_id, poses[1].track_id = 7, 9

    from soccer_mocap.types import PoseSequence

    sequence = PoseSequence(
        frames=[FramePoses(0, 0.0, poses)], video=make_video_metadata()
    )
    tracks = sequence.build_tracks()

    assert sorted(tracks) == [7, 9]


# -- video metadata --------------------------------------------------------


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [(0, (1920, 1080)), (90, (1080, 1920)), (180, (1920, 1080)), (270, (1080, 1920))],
)
def test_display_size_accounts_for_rotation(rotation, expected):
    """Phone portrait video is landscape pixels plus a rotation flag."""
    meta = VideoMetadata("x.mov", 1920, 1080, 30.0, 100, rotation=rotation)
    assert meta.display_size == expected


def test_high_speed_detection():
    assert VideoMetadata("x", 1920, 1080, 240.0, 100).is_high_speed
    assert not VideoMetadata("x", 1920, 1080, 30.0, 100).is_high_speed


# -- device presets --------------------------------------------------------


def test_hevc_widescreen_is_inferred_as_action_cam():
    meta = VideoMetadata("x.mp4", 3840, 1600, 60.0, 100, codec="hevc")
    assert infer_preset(meta) is ACTION_CAM


def test_slow_motion_gets_a_frame_stride():
    """240 fps of a 2-second dive is mostly redundant frames."""
    meta = make_video_metadata(fps=240.0)
    profile = resolve_profile(meta, ACTION_CAM)
    assert profile.frame_stride >= 2


def test_explicit_profile_overrides_the_preset():
    meta = make_video_metadata(fps=240.0)
    assert resolve_profile(meta, ACTION_CAM, "accurate").name == "accurate"


def test_long_clips_are_downscaled():
    meta = VideoMetadata("x", 3840, 2160, 30.0, 9000, duration_s=300.0)
    profile = resolve_profile(meta, ACTION_CAM)
    assert profile.target_long_edge is not None
    assert profile.target_long_edge <= 1280


def test_balanced_profile_is_unchanged_for_ordinary_clips():
    meta = make_video_metadata(fps=30.0, num_frames=300)
    from soccer_mocap.devices.presets import GENERIC

    assert resolve_profile(meta, GENERIC).name == BALANCED.name


# -- config ----------------------------------------------------------------


def test_defaults_are_valid():
    config = PipelineConfig()
    assert config.pose.backend == "mediapipe"
    assert config.goalkeeper.enabled is False


def test_nested_sections_load_from_dict():
    config = PipelineConfig.from_dict(
        {"goalkeeper": {"enabled": True, "dive_speed_threshold": 2.0}}
    )
    assert config.goalkeeper.enabled is True
    assert config.goalkeeper.dive_speed_threshold == 2.0
    # Untouched sections keep their defaults.
    assert config.pose.backend == "mediapipe"


def test_unknown_key_is_rejected_rather_than_ignored():
    """A silently-dropped typo is how you believe you ran an ablation you didn't."""
    with pytest.raises(ConfigError, match="Unknown config key"):
        PipelineConfig.from_dict({"goalkeeper": {"enbaled": True}})


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ConfigError, match="Unknown config key"):
        PipelineConfig.from_dict({"goalkeepr": {}})


def test_config_round_trips_through_yaml(tmp_path):
    config = PipelineConfig()
    config.goalkeeper.enabled = True
    config.goalkeeper.goal_posts = [[10.0, 20.0], [30.0, 40.0]]

    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = PipelineConfig.from_yaml(path)

    assert loaded.goalkeeper.enabled is True
    assert loaded.goalkeeper.goal_posts == [[10.0, 20.0], [30.0, 40.0]]
