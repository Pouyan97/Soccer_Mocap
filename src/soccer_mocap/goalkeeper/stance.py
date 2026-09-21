"""Set-position (ready stance) analysis.

The set position is the posture a goalkeeper holds as the striker plants
their standing foot. Coaches care about it because it is the last thing the
keeper controls before the ball is struck, and because the common faults are
visible and fixable: too upright to push off, too narrow to be stable, hands
too low to reach a shot at chest height, weight on the heels.

Every measure here is a ratio, not a length. Ratios survive the two things
that vary most between clips - how far away the camera is and how tall the
keeper is - so numbers from different sessions are comparable without any
calibration at all.

The reference ranges attached to each metric come from coaching convention
rather than from a published normative dataset. They are there to make the
output legible, and they should be replaced with values from your own cohort
before anyone draws a conclusion from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analysis.kinematics import (
    body_height_px,
    center_of_mass,
    joint_angle,
    reference_height_px,
    segment_angle_from_vertical,
)
from ..skeleton import J
from ..types import PersonPose, Track


@dataclass
class StanceMetrics:
    """Ready-position measurements for a single frame.

    All distances are expressed as ratios (see the module docstring); angles
    are in degrees.
    """

    frame_index: int
    #: Ankle separation / shoulder separation. ~1.0-1.5 is a typical ready stance.
    stance_width_ratio: float = float("nan")
    #: Knee angles, 180 = straight.
    left_knee_angle: float = float("nan")
    right_knee_angle: float = float("nan")
    #: Hip angles (shoulder-hip-knee), 180 = straight.
    left_hip_angle: float = float("nan")
    right_hip_angle: float = float("nan")
    #: Trunk lean from vertical. Positive leans toward screen right.
    trunk_lean_deg: float = float("nan")
    #: Forward trunk inclination, magnitude only. ~10-25 degrees is typical.
    trunk_flexion_deg: float = float("nan")
    #: Wrist height above the hips, in body heights. Negative = hands low.
    hand_height_ratio: float = float("nan")
    #: Wrist separation / shoulder separation.
    hand_width_ratio: float = float("nan")
    #: COM height above the ankles, in body heights. Lower = deeper crouch.
    com_height_ratio: float = float("nan")
    #: Scale reference this frame was normalised by.
    body_height_px: float = float("nan")
    #: Notes on anything outside the conventional range.
    flags: list[str] = field(default_factory=list)

    @property
    def mean_knee_angle(self) -> float:
        angles = [a for a in (self.left_knee_angle, self.right_knee_angle) if np.isfinite(a)]
        return float(np.mean(angles)) if angles else float("nan")

    @property
    def is_valid(self) -> bool:
        """Whether enough joints were visible for the result to mean anything."""
        return np.isfinite(self.body_height_px) and np.isfinite(self.stance_width_ratio)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "stance_width_ratio": self.stance_width_ratio,
            "mean_knee_angle": self.mean_knee_angle,
            "left_knee_angle": self.left_knee_angle,
            "right_knee_angle": self.right_knee_angle,
            "trunk_flexion_deg": self.trunk_flexion_deg,
            "trunk_lean_deg": self.trunk_lean_deg,
            "hand_height_ratio": self.hand_height_ratio,
            "hand_width_ratio": self.hand_width_ratio,
            "com_height_ratio": self.com_height_ratio,
            "flags": list(self.flags),
        }


# Conventional ranges, used only to generate the coaching flags.
KNEE_ANGLE_RANGE = (130.0, 165.0)
STANCE_WIDTH_RANGE = (0.9, 1.7)
TRUNK_FLEXION_RANGE = (5.0, 30.0)
HAND_HEIGHT_MIN = -0.02


def analyse_stance(pose: PersonPose, frame_index: int = -1) -> StanceMetrics:
    """Measure the ready position in one frame."""
    xy, confidence = pose.xy, pose.confidence
    metrics = StanceMetrics(frame_index=frame_index)

    height = body_height_px(xy, confidence)
    metrics.body_height_px = height
    if not np.isfinite(height) or height <= 0:
        metrics.flags.append("body height could not be estimated; metrics unavailable")
        return metrics

    def point(index: int) -> np.ndarray | None:
        return pose.joint(index, min_confidence=0.2)

    left_ankle, right_ankle = point(J.LEFT_ANKLE), point(J.RIGHT_ANKLE)
    left_shoulder, right_shoulder = point(J.LEFT_SHOULDER), point(J.RIGHT_SHOULDER)
    left_hip, right_hip = point(J.LEFT_HIP), point(J.RIGHT_HIP)
    left_knee, right_knee = point(J.LEFT_KNEE), point(J.RIGHT_KNEE)
    left_wrist, right_wrist = point(J.LEFT_WRIST), point(J.RIGHT_WRIST)

    shoulder_width = (
        float(np.linalg.norm(left_shoulder - right_shoulder))
        if left_shoulder is not None and right_shoulder is not None
        else float("nan")
    )

    # -- stance width --
    if left_ankle is not None and right_ankle is not None and shoulder_width > 0:
        metrics.stance_width_ratio = (
            float(np.linalg.norm(left_ankle - right_ankle)) / shoulder_width
        )

    # -- joint angles --
    if left_hip is not None and left_knee is not None and left_ankle is not None:
        metrics.left_knee_angle = joint_angle(left_hip, left_knee, left_ankle)
    if right_hip is not None and right_knee is not None and right_ankle is not None:
        metrics.right_knee_angle = joint_angle(right_hip, right_knee, right_ankle)
    if left_shoulder is not None and left_hip is not None and left_knee is not None:
        metrics.left_hip_angle = joint_angle(left_shoulder, left_hip, left_knee)
    if right_shoulder is not None and right_hip is not None and right_knee is not None:
        metrics.right_hip_angle = joint_angle(right_shoulder, right_hip, right_knee)

    # -- trunk --
    if (
        left_shoulder is not None
        and right_shoulder is not None
        and left_hip is not None
        and right_hip is not None
    ):
        mid_shoulder = 0.5 * (left_shoulder + right_shoulder)
        mid_hip = 0.5 * (left_hip + right_hip)
        lean = segment_angle_from_vertical(mid_hip, mid_shoulder)
        metrics.trunk_lean_deg = lean
        metrics.trunk_flexion_deg = abs(lean)

        # -- hands, measured against the hip line --
        wrists = [w for w in (left_wrist, right_wrist) if w is not None]
        if wrists:
            mean_wrist_y = float(np.mean([w[1] for w in wrists]))
            # Image y grows downward, so hips-minus-wrist is "how far above".
            metrics.hand_height_ratio = (float(mid_hip[1]) - mean_wrist_y) / height
        if left_wrist is not None and right_wrist is not None and shoulder_width > 0:
            metrics.hand_width_ratio = (
                float(np.linalg.norm(left_wrist - right_wrist)) / shoulder_width
            )

    # -- crouch depth --
    com = center_of_mass(xy, confidence)
    ankles = [a for a in (left_ankle, right_ankle) if a is not None]
    if np.isfinite(com).all() and ankles:
        ankle_y = float(np.mean([a[1] for a in ankles]))
        metrics.com_height_ratio = (ankle_y - float(com[1])) / height

    metrics.flags.extend(_coaching_flags(metrics))
    return metrics


def _coaching_flags(metrics: StanceMetrics) -> list[str]:
    """Turn the measurements into plain-language notes."""
    flags: list[str] = []

    knee = metrics.mean_knee_angle
    if np.isfinite(knee):
        if knee > KNEE_ANGLE_RANGE[1]:
            flags.append(
                f"legs nearly straight ({knee:.0f} deg) - little loading available to push off"
            )
        elif knee < KNEE_ANGLE_RANGE[0]:
            flags.append(f"very deep crouch ({knee:.0f} deg) - slow to extend upward")

    width = metrics.stance_width_ratio
    if np.isfinite(width):
        if width < STANCE_WIDTH_RANGE[0]:
            flags.append(f"narrow stance ({width:.2f}x shoulders) - limited lateral base")
        elif width > STANCE_WIDTH_RANGE[1]:
            flags.append(f"wide stance ({width:.2f}x shoulders) - slow first step")

    flexion = metrics.trunk_flexion_deg
    if np.isfinite(flexion):
        if flexion < TRUNK_FLEXION_RANGE[0]:
            flags.append(f"upright trunk ({flexion:.0f} deg) - weight likely back on the heels")
        elif flexion > TRUNK_FLEXION_RANGE[1]:
            flags.append(f"heavy forward lean ({flexion:.0f} deg) - vulnerable to shots overhead")

    hands = metrics.hand_height_ratio
    if np.isfinite(hands) and hands < HAND_HEIGHT_MIN:
        flags.append("hands below hip height - late to reach mid-height shots")

    return flags


def find_set_position(
    track: Track,
    fps: float,
    search_window: tuple[int, int] | None = None,
) -> int | None:
    """Find the frame where the keeper is most 'set'.

    Defined as the local minimum of centre-of-mass speed, preferring a lower
    centre of mass among near-ties, since a set keeper is both still and
    loaded.

    Stillness alone is not enough to identify the set position, because a
    keeper lying on the ground after a dive is the stillest they will be all
    clip. So when no explicit window is given, the search is restricted to
    the frames *before* the fastest movement in the track - the set position
    precedes the dive by definition.

    Args:
        track: the goalkeeper's track.
        fps: frame rate.
        search_window: ``(first_frame, last_frame)`` to search within,
            which overrides the automatic restriction. Pass the frames
            before a known shot to find the stance for that shot.

    Returns:
        The frame index, or ``None`` when the track is too short or sparse.
    """
    frames = track.frame_indices
    if search_window is not None:
        low, high = search_window
        frames = [f for f in frames if low <= f <= high]
    if len(frames) < 3:
        return None

    positions, heights, valid_frames = [], [], []
    for frame_index in frames:
        pose = track.poses[frame_index]
        com = center_of_mass(pose.xy, pose.confidence)
        height = body_height_px(pose.xy, pose.confidence)
        if np.isfinite(com).all() and np.isfinite(height) and height > 0:
            positions.append(com)
            heights.append(height)
            valid_frames.append(frame_index)

    if len(valid_frames) < 3:
        return None

    path = np.array(positions)
    scale = reference_height_px(heights)
    if not np.isfinite(scale) or scale <= 0:
        return None

    # Body-heights per frame, so the threshold is resolution-independent.
    step = np.zeros(len(path))
    step[1:] = np.linalg.norm(np.diff(path, axis=0), axis=1) / scale

    # Everything from the fastest frame onward is the dive and its aftermath,
    # and the aftermath is stiller than the set position ever is.
    last_candidate = len(step)
    if search_window is None:
        fastest = int(np.argmax(step))
        if fastest >= 3:
            last_candidate = fastest

    normalised_depth = path[:, 1] / scale  # larger y = lower body

    # Rank by stillness first, break ties toward the lower (more loaded) COM.
    cost = step - 0.05 * (normalised_depth - normalised_depth.mean())
    return valid_frames[int(np.argmin(cost[:last_candidate]))]


def analyse_set_position(
    track: Track,
    fps: float,
    search_window: tuple[int, int] | None = None,
) -> StanceMetrics | None:
    """Locate the set position and measure it. Convenience wrapper."""
    frame_index = find_set_position(track, fps, search_window)
    if frame_index is None:
        return None
    return analyse_stance(track.poses[frame_index], frame_index)
