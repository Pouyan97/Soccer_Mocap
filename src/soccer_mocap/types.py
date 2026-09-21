"""Core data structures passed between pipeline stages.

The whole pipeline speaks in these types. A pose backend returns
``FramePoses``; the tracker turns a list of those into ``Track`` objects; the
analysis and goalkeeper modules consume ``Track``. Keeping the contract in one
place is what makes the backends swappable.

Coordinates follow two conventions, and mixing them up is the easiest bug to
write here:

* **Image space** - pixels, origin at top-left, ``y`` increasing downward.
  Everything a 2D pose backend emits lives here.
* **Pitch space** - metres, origin at the centre spot, ``x`` along the
  touchline toward the attacking goal, ``y`` across the pitch. Produced by
  :mod:`soccer_mocap.calibration` once a homography is available.

Attribute names carry the space: ``xy`` is image space, ``xy_pitch`` is pitch
space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .skeleton import CANONICAL, SkeletonLayout


@dataclass(frozen=True)
class VideoMetadata:
    """What we know about a source clip before decoding any pixels.

    Phone footage makes several of these fields load-bearing rather than
    informational: ``rotation`` is frequently non-zero because phones record
    in the sensor's native orientation and set a container flag instead of
    rotating pixels, and ``fps`` is often fractional (29.97) or very high
    (240 for slow-motion) which changes every velocity we compute.
    """

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    #: Container rotation flag in degrees (0/90/180/270), as phones set it.
    rotation: int = 0
    codec: str = "unknown"
    duration_s: float | None = None

    @property
    def display_size(self) -> tuple[int, int]:
        """``(width, height)`` after the container rotation is applied."""
        if self.rotation in (90, 270):
            return self.height, self.width
        return self.width, self.height

    @property
    def is_portrait(self) -> bool:
        width, height = self.display_size
        return height > width

    @property
    def is_high_speed(self) -> bool:
        """True for slow-motion capture, where event timing gets interesting."""
        return self.fps >= 100.0


@dataclass
class BBox:
    """Axis-aligned bounding box in image space (pixels)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=float)

    def intersection(self, other: BBox) -> float:
        """Overlapping area with ``other``, in square pixels."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    def iou(self, other: BBox) -> float:
        """Intersection-over-union with ``other``."""
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def containment(self, other: BBox) -> float:
        """Overlap as a fraction of the *smaller* box's area.

        1.0 means one box sits entirely inside the other. IoU misses this
        case badly - a small box nested in one four times its size scores
        only 0.25 - which is exactly the shape of the spurious partial
        detections that pose models emit during fast motion.
        """
        smaller = min(self.area, other.area)
        return self.intersection(other) / smaller if smaller > 0 else 0.0

    @classmethod
    def from_keypoints(cls, xy: np.ndarray, padding: float = 0.0) -> BBox:
        """Tight box around the finite keypoints, optionally padded by a ratio."""
        finite = xy[np.isfinite(xy).all(axis=-1)]
        if len(finite) == 0:
            return cls(0.0, 0.0, 0.0, 0.0, score=0.0)
        x1, y1 = finite.min(axis=0)
        x2, y2 = finite.max(axis=0)
        pad_x, pad_y = padding * (x2 - x1), padding * (y2 - y1)
        return cls(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)


@dataclass
class PersonPose:
    """One detected person in one frame."""

    #: ``(J, 2)`` keypoints in image space, NaN where not estimated.
    xy: np.ndarray
    #: ``(J,)`` per-joint confidence in ``[0, 1]``.
    confidence: np.ndarray
    layout: SkeletonLayout = CANONICAL
    bbox: BBox | None = None
    track_id: int | None = None
    #: ``(J, 3)`` root-relative 3D joints in metres, when a lifter has run.
    xyz: np.ndarray | None = None
    #: ``(2,)`` position on the pitch in metres, when calibration is available.
    xy_pitch: np.ndarray | None = None
    #: Backend-native keypoints, kept when the layout was downsampled to canonical.
    raw_xy: np.ndarray | None = None
    raw_layout: SkeletonLayout | None = None

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=float)
        self.confidence = np.asarray(self.confidence, dtype=float)
        if self.xy.shape[0] != self.confidence.shape[0]:
            raise ValueError(
                f"keypoint/confidence length mismatch: {self.xy.shape[0]} vs "
                f"{self.confidence.shape[0]}"
            )
        if self.bbox is None:
            self.bbox = BBox.from_keypoints(self.xy)

    @property
    def num_joints(self) -> int:
        return self.xy.shape[0]

    def visible(self, min_confidence: float = 0.3) -> np.ndarray:
        """Boolean mask of joints that are both finite and confident enough."""
        return (self.confidence >= min_confidence) & np.isfinite(self.xy).all(axis=-1)

    def joint(self, index: int, min_confidence: float = 0.0) -> np.ndarray | None:
        """Return one joint's ``(x, y)``, or ``None`` if it is not usable."""
        if index >= self.num_joints or self.confidence[index] < min_confidence:
            return None
        point = self.xy[index]
        return point if np.isfinite(point).all() else None

    def mean_confidence(self) -> float:
        finite = self.confidence[np.isfinite(self.confidence)]
        return float(finite.mean()) if finite.size else 0.0


@dataclass
class FramePoses:
    """Every person found in a single frame."""

    frame_index: int
    timestamp: float
    people: list[PersonPose] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.people)

    def by_track(self, track_id: int) -> PersonPose | None:
        return next((p for p in self.people if p.track_id == track_id), None)

    def largest(self) -> PersonPose | None:
        """The person with the biggest bounding box.

        A decent proxy for 'the subject' in single-player phone footage, where
        the person being filmed is nearest the camera.
        """
        if not self.people:
            return None
        return max(self.people, key=lambda p: p.bbox.area if p.bbox else 0.0)


@dataclass
class Track:
    """One person followed across frames.

    Frames where the track was not detected are simply absent from
    ``poses``; use :meth:`pose_at` rather than indexing by position.
    """

    track_id: int
    poses: dict[int, PersonPose] = field(default_factory=dict)
    #: Free-form labels attached by later stages, e.g. ``{"role": "goalkeeper"}``.
    attributes: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def frame_indices(self) -> list[int]:
        return sorted(self.poses)

    @property
    def first_frame(self) -> int:
        return min(self.poses) if self.poses else -1

    @property
    def last_frame(self) -> int:
        return max(self.poses) if self.poses else -1

    def pose_at(self, frame_index: int) -> PersonPose | None:
        return self.poses.get(frame_index)

    def add(self, frame_index: int, pose: PersonPose) -> None:
        pose.track_id = self.track_id
        self.poses[frame_index] = pose

    def stack(self, min_confidence: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dense ``(frames, xy, confidence)`` arrays over this track's span.

        Gaps inside the span come back as NaN rather than being dropped, so the
        time axis stays uniform and velocities stay meaningful. Joints below
        ``min_confidence`` are blanked the same way.
        """
        if not self.poses:
            return np.empty(0, dtype=int), np.empty((0, 0, 2)), np.empty((0, 0))

        span = np.arange(self.first_frame, self.last_frame + 1)
        num_joints = next(iter(self.poses.values())).num_joints
        xy = np.full((len(span), num_joints, 2), np.nan)
        conf = np.zeros((len(span), num_joints))

        for row, frame_index in enumerate(span):
            pose = self.poses.get(int(frame_index))
            if pose is None:
                continue
            keep = pose.confidence >= min_confidence
            xy[row][keep] = pose.xy[keep]
            conf[row][keep] = pose.confidence[keep]
        return span, xy, conf


@dataclass
class PoseSequence:
    """The full result of running a video through the pose stage."""

    frames: list[FramePoses]
    video: VideoMetadata
    layout: SkeletonLayout = CANONICAL
    backend: str = "unknown"
    tracks: dict[int, Track] = field(default_factory=dict)
    #: Anything a stage wants to record about how this result was produced.
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def fps(self) -> float:
        return self.video.fps

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.fps if self.fps > 0 else 0.0

    def frame(self, frame_index: int) -> FramePoses | None:
        return next((f for f in self.frames if f.frame_index == frame_index), None)

    def build_tracks(self) -> dict[int, Track]:
        """Group per-frame detections into :class:`Track` objects by ``track_id``.

        Call after a tracker has assigned ids. Detections still carrying
        ``track_id is None`` are skipped.
        """
        tracks: dict[int, Track] = {}
        for frame in self.frames:
            for person in frame.people:
                if person.track_id is None:
                    continue
                track = tracks.setdefault(person.track_id, Track(track_id=person.track_id))
                track.add(frame.frame_index, person)
        self.tracks = tracks
        return tracks

    def longest_track(self) -> Track | None:
        """The track present in the most frames - usually the intended subject."""
        tracks = self.tracks or self.build_tracks()
        return max(tracks.values(), key=len) if tracks else None
