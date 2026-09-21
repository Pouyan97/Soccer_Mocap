"""Lifting 2D keypoints to 3D.

What single-camera 3D can and cannot be:

A single view of a moving person is fundamentally ambiguous about depth. Any
3D reconstruction from one camera is the model's prior about human bodies,
not a measurement. That is fine for some questions - is the trunk rotated,
which knee is flexed more, did the keeper get their hips through the dive -
and useless for others. Anything requiring absolute depth accuracy needs a
second camera, and no amount of model sophistication changes that.

So the rule for this project: **report 3D as relative, never as absolute.**
Joint angles in the body's own frame are defensible. A claim that the
keeper's hand was 1.83 m above the grass is not, unless it came through
pitch calibration on the ground plane.

Three sources of 3D are available, in descending order of trustworthiness:

1. **Multi-view**, if a second phone is ever added. Genuinely metric.
2. **A learned lifter** (MotionBERT, VideoPose3D) or a parametric body model
   (SMPL via HMR2/CLIFF). Temporally coherent and anatomically plausible.
3. **MediaPipe world landmarks**, already produced by the default backend at
   no extra cost. Coarse, per-frame, and jittery, but free.

Option 3 is wired up in :class:`PassthroughLifter`. Options 1 and 2 are the
reason this interface exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..skeleton import J
from ..types import PersonPose, Track


@dataclass
class LifterInfo:
    """Static description of a lifting method."""

    key: str
    label: str
    summary: str
    #: True when output is in real metres rather than an arbitrary scale.
    metric: bool
    #: True when the method uses temporal context rather than single frames.
    temporal: bool
    needs_gpu: bool = False
    install_hint: str = ""
    implemented: bool = True


class Lifter(ABC):
    """Produce 3D joint positions from 2D keypoints."""

    info: LifterInfo

    def __init__(self, **options: Any) -> None:
        self.options = options

    @classmethod
    @abstractmethod
    def available(cls) -> tuple[bool, str]:
        """``(usable, reason)``."""

    @abstractmethod
    def lift_track(self, track: Track, fps: float) -> Track:
        """Fill in ``PersonPose.xyz`` for every pose in the track."""


class PassthroughLifter(Lifter):
    """Use the 3D the pose backend already produced.

    MediaPipe's world landmarks are metric-ish and hip-relative, and the
    backend has already remapped them to the canonical skeleton, so there is
    nothing to compute. The only work done here is optional temporal
    smoothing, because per-frame 3D from a monocular model is noticeably
    jittery in the depth axis.
    """

    info = LifterInfo(
        key="passthrough",
        label="Backend-provided 3D",
        summary=(
            "Uses whatever 3D the pose backend emitted, optionally smoothed. "
            "Free with MediaPipe; approximate and hip-relative."
        ),
        metric=False,
        temporal=False,
        implemented=True,
    )

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return True, ""

    def lift_track(self, track: Track, fps: float) -> Track:
        smoothing_hz = self.options.get("smoothing_hz", 6.0)
        frames = track.frame_indices
        available = [f for f in frames if track.poses[f].xyz is not None]

        if not available:
            return track
        if smoothing_hz is None or len(available) < 10:
            return track

        from ..analysis.kinematics import filter_trajectory

        stacked = np.stack([track.poses[f].xyz for f in available])
        smoothed = filter_trajectory(stacked, fps, smoothing_hz)
        for position, frame_index in enumerate(available):
            track.poses[frame_index].xyz = smoothed[position]
        return track


class MonocularLifter(Lifter):
    """Learned 2D-to-3D lifting - planned integration.

    The natural choice is MotionBERT or a similar transformer lifter: they
    take a window of 2D keypoints and output temporally coherent 3D, which
    fixes the frame-to-frame jitter that makes per-frame 3D unusable for
    velocity. A parametric body model (SMPL through HMR2 or CLIFF) is the
    alternative and additionally gives surface geometry and consistent
    segment lengths, at a higher compute cost.

    Implementing this means: install the upstream package, buffer a window
    of normalised 2D keypoints per track, run the model, and write the
    result back into ``PersonPose.xyz``. The interface already assumes a
    whole track, so a windowed model fits without changes.
    """

    info = LifterInfo(
        key="monocular",
        label="Learned monocular lifter (MotionBERT / SMPL)",
        summary=(
            "Temporally coherent 3D from 2D keypoint sequences. "
            "Much smoother than per-frame 3D; still not metric."
        ),
        metric=False,
        temporal=True,
        needs_gpu=True,
        install_hint=(
            "No package is wired up yet. See the notes in "
            "src/soccer_mocap/lifting/base.py for the intended approach."
        ),
        implemented=False,
    )

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return False, "integration not implemented yet"

    def lift_track(self, track: Track, fps: float) -> Track:
        from ..exceptions import BackendNotImplementedError

        raise BackendNotImplementedError(
            self.info.key, "learned lifting is not implemented", self.info.install_hint
        )


LIFTERS: dict[str, type[Lifter]] = {
    "passthrough": PassthroughLifter,
    "monocular": MonocularLifter,
}


def get_lifter(key: str = "passthrough", **options: Any) -> Lifter:
    """Construct a lifter by name."""
    from ..exceptions import BackendUnavailableError

    if key not in LIFTERS:
        raise BackendUnavailableError(
            key,
            "no lifter is registered under that name",
            f"Choose one of: {', '.join(sorted(LIFTERS))}",
        )
    lifter_cls = LIFTERS[key]
    usable, reason = lifter_cls.available()
    if not usable:
        raise BackendUnavailableError(key, reason, lifter_cls.info.install_hint or None)
    return lifter_cls(**options)


def estimate_scale_from_height(pose: PersonPose, stature_m: float = 1.80) -> float | None:
    """Metres per pixel, from an assumed real stature.

    The crudest possible calibration, and the last resort when no goal and no
    pitch homography are available. It is only valid at the subject's own
    depth, and it inherits the full error of the stature guess - a 10 cm
    error in an assumed height is a 5-6% error in every distance derived
    from it.
    """
    from ..analysis.kinematics import body_height_px

    height_px = body_height_px(pose.xy, pose.confidence)
    if not np.isfinite(height_px) or height_px <= 0:
        return None
    return stature_m / height_px


def root_relative(xyz: np.ndarray) -> np.ndarray:
    """Re-centre a 3D pose on the mid-hip.

    Removes global translation so poses can be compared across frames and
    between people.
    """
    array = np.asarray(xyz, dtype=float)
    hips = array[[J.LEFT_HIP, J.RIGHT_HIP]]
    if not np.isfinite(hips).all():
        return array
    return array - hips.mean(axis=0)
