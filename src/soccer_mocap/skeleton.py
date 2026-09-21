"""Skeleton layouts and the mapping between them.

Every pose backend emits keypoints in its own layout: MediaPipe gives 33
landmarks, most academic models give COCO-17, Halpe gives 26, Sapiens goes up
to 308. Downstream analysis should never have to care which one produced the
data, so this module defines a single canonical layout (COCO-17) and the
remapping tables that project each backend layout onto it.

Analysis code works on canonical indices via the ``J`` accessor; raw
backend-native keypoints are still carried alongside for anything that needs
the extra detail (e.g. Sapiens foot joints for ankle kinematics).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SkeletonLayout:
    """Names and connectivity for one backend's keypoint set."""

    name: str
    joint_names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...] = ()
    #: Maps this layout's joint name -> canonical (COCO-17) joint name.
    #: Only joints that have a canonical equivalent need an entry.
    canonical_aliases: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.joint_names)

    def index(self, joint_name: str) -> int:
        """Return the column index of ``joint_name`` in this layout."""
        try:
            return self.joint_names.index(joint_name)
        except ValueError as exc:
            raise KeyError(
                f"Joint {joint_name!r} is not part of layout {self.name!r}. "
                f"Available: {', '.join(self.joint_names)}"
            ) from exc

    def has(self, joint_name: str) -> bool:
        return joint_name in self.joint_names


# --------------------------------------------------------------------------
# Canonical layout: COCO-17.
# --------------------------------------------------------------------------
# Chosen as canonical because every backend under consideration can produce it
# either natively or by subsetting, which keeps the conversion lossless in the
# direction we care about.

COCO17_JOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO17_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
)

COCO17 = SkeletonLayout(name="coco17", joint_names=COCO17_JOINTS, edges=COCO17_EDGES)

#: Canonical layout used by all analysis code.
CANONICAL = COCO17


class J:
    """Canonical joint indices, for readable analysis code.

    ``kp[J.LEFT_KNEE]`` beats ``kp[13]`` when you are reading a dive-detection
    routine six months from now.
    """

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# --------------------------------------------------------------------------
# MediaPipe Pose: 33 landmarks.
# --------------------------------------------------------------------------

MEDIAPIPE33_JOINTS: tuple[str, ...] = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

MEDIAPIPE33 = SkeletonLayout(
    name="mediapipe33",
    joint_names=MEDIAPIPE33_JOINTS,
    edges=(
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
        (27, 29), (29, 31), (28, 30), (30, 32),
    ),
    # MediaPipe's names already match COCO for the 17 we need, so the alias
    # table is an identity mapping over the shared subset.
    canonical_aliases={name: name for name in COCO17_JOINTS},
)


# --------------------------------------------------------------------------
# Halpe-26: what RTMPose and several PosePipeline algorithms emit.
# --------------------------------------------------------------------------

HALPE26_JOINTS: tuple[str, ...] = (
    *COCO17_JOINTS,
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe",
    "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
)

HALPE26 = SkeletonLayout(
    name="halpe26",
    joint_names=HALPE26_JOINTS,
    edges=COCO17_EDGES,
    canonical_aliases={name: name for name in COCO17_JOINTS},
)


LAYOUTS: dict[str, SkeletonLayout] = {
    layout.name: layout for layout in (COCO17, MEDIAPIPE33, HALPE26)
}


def get_layout(name: str) -> SkeletonLayout:
    """Look up a registered layout by name."""
    try:
        return LAYOUTS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown skeleton layout {name!r}. Registered: {', '.join(sorted(LAYOUTS))}"
        ) from exc


def build_canonical_index(layout: SkeletonLayout) -> np.ndarray:
    """Return an index array that gathers ``layout`` columns into canonical order.

    The result has one entry per canonical joint. An entry of ``-1`` marks a
    canonical joint this layout cannot supply, which the caller fills with NaN
    and zero confidence.
    """
    index = np.full(len(CANONICAL), -1, dtype=np.int64)
    reverse = {dst: src for src, dst in layout.canonical_aliases.items()}
    for canonical_pos, canonical_name in enumerate(CANONICAL.joint_names):
        source_name = reverse.get(canonical_name)
        if source_name is None and layout.has(canonical_name):
            source_name = canonical_name
        if source_name is not None and layout.has(source_name):
            index[canonical_pos] = layout.index(source_name)
    return index


def to_canonical(
    xy: np.ndarray,
    confidence: np.ndarray,
    layout: SkeletonLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap backend-native keypoints into the canonical COCO-17 layout.

    Args:
        xy: ``(J, 2)`` or ``(T, J, 2)`` coordinates in the layout's own order.
        confidence: ``(J,)`` or ``(T, J)`` matching scores.
        layout: the layout ``xy`` is expressed in.

    Returns:
        ``(xy_canonical, confidence_canonical)``. Joints the source layout
        cannot provide come back as NaN with zero confidence.
    """
    if layout.name == CANONICAL.name:
        return np.asarray(xy, dtype=float), np.asarray(confidence, dtype=float)

    index = build_canonical_index(layout)
    valid = index >= 0
    safe_index = np.where(valid, index, 0)

    out_xy = np.take(np.asarray(xy, dtype=float), safe_index, axis=-2)
    out_conf = np.take(np.asarray(confidence, dtype=float), safe_index, axis=-1)

    # Blank out the joints this layout does not carry.
    out_xy[..., ~valid, :] = np.nan
    out_conf[..., ~valid] = 0.0
    return out_xy, out_conf
