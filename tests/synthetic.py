"""Synthetic poses and tracks, so tests never need a video file.

Real footage cannot be committed to the repository and cannot be relied on in
CI, but the analysis code still needs something with the right shape and
plausible dynamics to run against. These builders generate COCO-17 skeletons
from a few high-level parameters (where the body is, how tall, how far it is
leaning) and assemble them into tracks with known ground truth.

The figures are kinematically crude - rigid segments, no joint limits - but
they are correct in the properties the analysis actually measures: segment
proportions follow standard anthropometry, the lean is applied about the hip,
and the COM moves the way the caller says it does. That is enough to test
that a dive is detected, pointed the right way, and timed correctly.
"""

from __future__ import annotations

import numpy as np

from soccer_mocap.skeleton import CANONICAL, J
from soccer_mocap.types import FramePoses, PersonPose, Track, VideoMetadata

#: Segment positions as a fraction of stature, measured down from the crown.
#: Rounded from Winter, *Biomechanics and Motor Control of Human Movement*.
_PROPORTIONS = {
    "head": 0.06,
    "shoulder": 0.18,
    "elbow": 0.34,
    "wrist": 0.49,
    "hip": 0.53,
    "knee": 0.75,
    "ankle": 0.97,
}

#: Half-widths, also as a fraction of stature.
_SHOULDER_HALF_WIDTH = 0.10
_HIP_HALF_WIDTH = 0.06


def make_pose(
    com_xy: tuple[float, float] = (320.0, 240.0),
    height_px: float = 300.0,
    lean_deg: float = 0.0,
    knee_flexion_deg: float = 0.0,
    stance_width: float = 1.2,
    arm_raise: float = 0.0,
    confidence: float = 0.95,
) -> PersonPose:
    """Build one plausible COCO-17 pose.

    Args:
        com_xy: where the centre of mass should sit, in pixels.
        height_px: stature in pixels.
        lean_deg: trunk rotation about the hip. Positive leans screen-right,
            matching :func:`~soccer_mocap.analysis.kinematics.segment_angle_from_vertical`.
        knee_flexion_deg: how much to bend the knees. 0 is standing straight.
        stance_width: ankle separation as a multiple of shoulder width.
        arm_raise: 0 keeps the arms down, 1 raises the wrists to head height.
        confidence: per-joint confidence to report.

    Returns:
        A :class:`~soccer_mocap.types.PersonPose` in the canonical layout.
    """
    xy = np.zeros((len(CANONICAL), 2), dtype=float)
    cx, cy = com_xy

    # The hip is the anchor; the COM sits a little above it in a standing
    # adult, so place the hip below the requested COM.
    hip_y = cy + 0.06 * height_px
    hip_half = _HIP_HALF_WIDTH * height_px
    shoulder_half = _SHOULDER_HALF_WIDTH * height_px

    def below_hip(fraction_of_stature: float) -> float:
        """Vertical pixel offset from the hip, downward positive."""
        return (fraction_of_stature - _PROPORTIONS["hip"]) * height_px

    # Legs are built as a two-link chain with an exact included angle, then
    # the whole chain is rotated about the hip to open the stance. Rotating a
    # rigid chain cannot change its internal angle, so the contract this
    # gives tests is crisp:
    #
    #     measured knee angle == 180 - knee_flexion_deg
    #     measured stance width ratio == stance_width
    #
    # An earlier version displaced the knee laterally to fake flexion, which
    # made the knee angle depend on stance width and move non-monotonically
    # with flexion - useless to test against.
    flex = np.radians(knee_flexion_deg)
    thigh = (_PROPORTIONS["knee"] - _PROPORTIONS["hip"]) * height_px
    shank = (_PROPORTIONS["ankle"] - _PROPORTIONS["knee"]) * height_px

    # Thigh tilts forward by half the flexion, shank back by the other half.
    thigh_vec = np.array([np.sin(flex / 2), np.cos(flex / 2)]) * thigh
    shank_vec = np.array([-np.sin(flex / 2), np.cos(flex / 2)]) * shank

    ankle_half = stance_width * shoulder_half
    # Splay each leg outward so the ankles land at the requested separation.
    leg_drop = float((thigh_vec + shank_vec)[1])
    splay = np.arctan2(ankle_half - hip_half, max(leg_drop, 1e-6))

    for side, hip_joint, knee_joint, ankle_joint in (
        (-1, J.LEFT_HIP, J.LEFT_KNEE, J.LEFT_ANKLE),
        (+1, J.RIGHT_HIP, J.RIGHT_KNEE, J.RIGHT_ANKLE),
    ):
        hip_point = np.array([cx + side * hip_half, hip_y])
        # Negated because image y points down, so a positive rotation swings
        # the foot inward rather than out.
        angle = -side * splay
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        knee_point = hip_point + rotation @ thigh_vec
        ankle_point = knee_point + rotation @ shank_vec

        xy[hip_joint] = hip_point
        xy[knee_joint] = knee_point
        xy[ankle_joint] = ankle_point


    # Upper body, rotated about the mid-hip by `lean_deg`.
    lean = np.radians(lean_deg)
    mid_hip = np.array([cx, hip_y])

    def upper(fraction_of_stature: float, lateral: float = 0.0) -> np.ndarray:
        """Place a point above the hip, then rotate it about the hip."""
        vertical = below_hip(fraction_of_stature)  # negative, i.e. above
        local = np.array([lateral, vertical])
        rotation = np.array(
            [[np.cos(lean), -np.sin(lean)], [np.sin(lean), np.cos(lean)]]
        )
        # Negate the x of the rotation so positive lean goes screen-right
        # given that image y points down.
        rotated = rotation @ local
        return mid_hip + np.array([-rotated[0], rotated[1]])

    xy[J.LEFT_SHOULDER] = upper(_PROPORTIONS["shoulder"], +shoulder_half)
    xy[J.RIGHT_SHOULDER] = upper(_PROPORTIONS["shoulder"], -shoulder_half)
    xy[J.NOSE] = upper(_PROPORTIONS["head"], 0.0)
    xy[J.LEFT_EYE] = upper(_PROPORTIONS["head"] - 0.005, +0.02 * height_px)
    xy[J.RIGHT_EYE] = upper(_PROPORTIONS["head"] - 0.005, -0.02 * height_px)
    xy[J.LEFT_EAR] = upper(_PROPORTIONS["head"] + 0.01, +0.04 * height_px)
    xy[J.RIGHT_EAR] = upper(_PROPORTIONS["head"] + 0.01, -0.04 * height_px)

    # Arms: interpolate each joint between hanging down and raised overhead.
    elbow_fraction = _PROPORTIONS["elbow"] - arm_raise * 0.14
    wrist_fraction = _PROPORTIONS["wrist"] - arm_raise * 0.42
    elbow_lateral = shoulder_half * (1.0 + 0.6 * arm_raise)
    wrist_lateral = shoulder_half * (1.0 + 1.0 * arm_raise)

    xy[J.LEFT_ELBOW] = upper(elbow_fraction, +elbow_lateral)
    xy[J.RIGHT_ELBOW] = upper(elbow_fraction, -elbow_lateral)
    xy[J.LEFT_WRIST] = upper(wrist_fraction, +wrist_lateral)
    xy[J.RIGHT_WRIST] = upper(wrist_fraction, -wrist_lateral)

    return PersonPose(
        xy=xy,
        confidence=np.full(len(CANONICAL), confidence),
        layout=CANONICAL,
    )


def make_standing_track(
    num_frames: int = 90,
    track_id: int = 0,
    com_xy: tuple[float, float] = (320.0, 240.0),
    height_px: float = 300.0,
    jitter_px: float = 0.5,
    seed: int = 0,
) -> Track:
    """A keeper standing still, with a little detection jitter.

    Used to confirm that the dive detector does *not* fire on a static
    subject, which is the failure mode that matters most in practice.
    """
    rng = np.random.default_rng(seed)
    track = Track(track_id=track_id)
    for frame in range(num_frames):
        offset = rng.normal(0.0, jitter_px, size=2)
        track.add(
            frame,
            make_pose(
                com_xy=(com_xy[0] + offset[0], com_xy[1] + offset[1]),
                height_px=height_px,
                knee_flexion_deg=25.0,
            ),
        )
    return track


def make_dive_track(
    num_frames: int = 120,
    fps: float = 60.0,
    track_id: int = 0,
    start_xy: tuple[float, float] = (320.0, 240.0),
    height_px: float = 300.0,
    direction: int = 1,
    dive_start_frame: int = 40,
    dive_frames: int = 20,
    lateral_travel_px: float = 260.0,
    peak_rise_px: float = 40.0,
    jitter_px: float = 0.4,
    seed: int = 1,
) -> Track:
    """A keeper who holds a set position, dives, and lands.

    The COM follows a smooth S-curve laterally and a parabola vertically, the
    trunk leans into the dive, and the arms come up. Ground truth the caller
    can assert against: the dive starts at ``dive_start_frame``, runs for
    ``dive_frames``, and goes screen-right when ``direction`` is ``+1``.
    """
    rng = np.random.default_rng(seed)
    track = Track(track_id=track_id)
    dive_end = dive_start_frame + dive_frames
    cx, cy = start_xy

    for frame in range(num_frames):
        if frame < dive_start_frame:
            progress = 0.0
        elif frame >= dive_end:
            progress = 1.0
        else:
            # Smoothstep gives a gentle takeoff and landing rather than the
            # velocity step a linear ramp would produce.
            t = (frame - dive_start_frame) / dive_frames
            progress = t * t * (3.0 - 2.0 * t)

        x = cx + direction * lateral_travel_px * progress
        # Rise then fall: up through the flight phase, down onto the ground.
        arc = np.sin(np.pi * progress)
        y = cy - peak_rise_px * arc + 0.35 * height_px * progress**2

        jitter = rng.normal(0.0, jitter_px, size=2)
        track.add(
            frame,
            make_pose(
                com_xy=(x + jitter[0], y + jitter[1]),
                height_px=height_px,
                lean_deg=direction * 55.0 * progress,
                knee_flexion_deg=25.0 * (1.0 - progress),
                arm_raise=min(1.0, 1.4 * progress),
            ),
        )
    return track


def track_to_frames(*tracks: Track) -> list[FramePoses]:
    """Interleave tracks into the per-frame structure the pipeline uses."""
    all_frames = sorted({f for track in tracks for f in track.frame_indices})
    frames: list[FramePoses] = []
    for frame_index in all_frames:
        people = []
        for track in tracks:
            pose = track.pose_at(frame_index)
            if pose is not None:
                pose.track_id = track.track_id
                people.append(pose)
        frames.append(FramePoses(frame_index=frame_index, timestamp=frame_index, people=people))
    return frames


def make_video_metadata(
    num_frames: int = 120,
    fps: float = 60.0,
    width: int = 1280,
    height: int = 720,
    rotation: int = 0,
) -> VideoMetadata:
    """Metadata for a clip that does not exist on disk."""
    return VideoMetadata(
        path="<synthetic>",
        width=width,
        height=height,
        fps=fps,
        frame_count=num_frames,
        rotation=rotation,
        codec="synthetic",
        duration_s=num_frames / fps,
    )
