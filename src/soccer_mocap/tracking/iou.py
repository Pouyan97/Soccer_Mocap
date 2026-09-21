"""IoU-based multi-person tracking.

A greedy IoU tracker with a small amount of motion prediction. This is
deliberately not a re-identification system: it links detections across
frames by overlap, tolerates short gaps, and gives up when a player is
occluded for long enough. That is sufficient for the two cases this project
actually needs -

* a single-subject drill video, where there is one player to follow, and
* a goalkeeper, who stays near the goal and is rarely occluded for long

and it is not sufficient for tracking 22 players through a crowded box. When
that becomes the requirement, replace this with ByteTrack or BoT-SORT rather
than adding heuristics here; the :class:`Tracker` interface is the seam.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..types import BBox, FramePoses, PersonPose


class Tracker(ABC):
    """Assign persistent ids to per-frame detections."""

    @abstractmethod
    def update(self, frame: FramePoses) -> FramePoses:
        """Set ``track_id`` on each person in ``frame`` and return it."""

    def run(self, frames: list[FramePoses]) -> list[FramePoses]:
        """Track a whole sequence in order."""
        return [self.update(frame) for frame in frames]

    def reset(self) -> None:  # noqa: B027 - optional hook, stateless trackers need none
        """Forget all state, ready for a new clip."""


@dataclass
class _TrackState:
    """Bookkeeping for one live track."""

    track_id: int
    bbox: BBox
    #: Per-frame centre displacement, used to predict where the box moves next.
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hits: int = 1
    age: int = 0
    #: Frames since the last successful match.
    misses: int = 0
    confirmed: bool = False

    def predict(self) -> BBox:
        """Where this track's box is expected to be in the next frame.

        A constant-velocity guess. Crude, but it is the difference between
        keeping and losing a sprinting player between two frames at 30 fps.
        """
        dx, dy = self.velocity
        return BBox(
            self.bbox.x1 + dx,
            self.bbox.y1 + dy,
            self.bbox.x2 + dx,
            self.bbox.y2 + dy,
            self.bbox.score,
        )

    def update(self, bbox: BBox, smoothing: float = 0.5) -> None:
        old_cx, old_cy = self.bbox.center
        new_cx, new_cy = bbox.center
        measured = np.array([new_cx - old_cx, new_cy - old_cy])
        # Exponential smoothing: a single noisy detection should not send the
        # prediction flying, but a real sprint should be picked up quickly.
        self.velocity = smoothing * measured + (1.0 - smoothing) * self.velocity
        self.bbox = bbox
        self.hits += 1
        self.misses = 0

    def mark_missed(self) -> None:
        self.misses += 1
        # Coast along the last known velocity while unmatched, decaying it so
        # a long-lost track does not drift across the frame.
        self.bbox = self.predict()
        self.velocity *= 0.8


class IoUTracker(Tracker):
    """Greedy IoU tracker with constant-velocity prediction.

    Args:
        iou_threshold: minimum overlap for a detection to continue a track.
        max_age: frames a track survives unmatched before being dropped.
        min_hits: matches before a track is reported as confirmed. Filters
            one-frame false positives.
        min_confidence: detections below this mean confidence are ignored.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        min_confidence: float = 0.0,
        max_center_distance_ratio: float = 1.5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.min_confidence = min_confidence
        self.max_center_distance_ratio = max_center_distance_ratio
        self._tracks: dict[int, _TrackState] = {}
        self._next_id = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def update(self, frame: FramePoses) -> FramePoses:
        detections = [
            person
            for person in frame.people
            if person.bbox is not None and person.mean_confidence() >= self.min_confidence
        ]

        for state in self._tracks.values():
            state.age += 1

        matches, unmatched = self._match(detections)

        for track_id, person in matches.items():
            assert person.bbox is not None
            state = self._tracks[track_id]
            state.update(person.bbox)
            if state.hits >= self.min_hits:
                state.confirmed = True
            person.track_id = track_id

        matched_ids = set(matches)
        for track_id, state in list(self._tracks.items()):
            if track_id in matched_ids:
                continue
            state.mark_missed()
            if state.misses > self.max_age:
                del self._tracks[track_id]

        for person in unmatched:
            assert person.bbox is not None
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox=person.bbox,
                confirmed=self.min_hits <= 1,
            )
            person.track_id = track_id

        return frame

    def _match(
        self, detections: list[PersonPose]
    ) -> tuple[dict[int, PersonPose], list[PersonPose]]:
        """Pair tracks with detections: IoU first, then centre distance.

        The second pass exists because IoU fails on exactly the event this
        project cares about. A goalkeeper going from upright to horizontal
        turns a tall narrow box into a short wide one in a few frames; the
        two barely overlap, IoU drops below any sensible threshold, and the
        track breaks in the middle of the dive. Centre distance does not
        care about aspect ratio, so it recovers the association.
        """
        if not self._tracks or not detections:
            return {}, list(detections)

        track_ids = list(self._tracks)
        predicted = {tid: self._tracks[tid].predict() for tid in track_ids}

        matches: dict[int, PersonPose] = {}
        used_detections: set[int] = set()
        used_tracks: set[int] = set()

        # -- pass 1: IoU --
        iou = np.zeros((len(track_ids), len(detections)))
        for i, track_id in enumerate(track_ids):
            for j, person in enumerate(detections):
                assert person.bbox is not None
                iou[i, j] = predicted[track_id].iou(person.bbox)

        # Greedy rather than Hungarian: with a handful of players the optimal
        # assignment and the greedy one agree, and greedy has no scipy
        # dependency in the hot loop.
        for i, j in _descending_pairs(iou):
            if iou[i, j] < self.iou_threshold:
                break
            if i in used_tracks or j in used_detections:
                continue
            used_tracks.add(i)
            used_detections.add(j)
            matches[track_ids[i]] = detections[j]

        # -- pass 2: centre distance, for whatever pass 1 left over --
        remaining_tracks = [i for i in range(len(track_ids)) if i not in used_tracks]
        remaining_detections = [j for j in range(len(detections)) if j not in used_detections]

        if remaining_tracks and remaining_detections:
            affinity = np.zeros((len(remaining_tracks), len(remaining_detections)))
            for a, i in enumerate(remaining_tracks):
                box = predicted[track_ids[i]]
                for b, j in enumerate(remaining_detections):
                    person = detections[j]
                    assert person.bbox is not None
                    affinity[a, b] = _center_affinity(
                        box, person.bbox, self.max_center_distance_ratio
                    )

            for a, b in _descending_pairs(affinity):
                if affinity[a, b] <= 0:
                    break
                i, j = remaining_tracks[a], remaining_detections[b]
                if i in used_tracks or j in used_detections:
                    continue
                used_tracks.add(i)
                used_detections.add(j)
                matches[track_ids[i]] = detections[j]

        unmatched = [p for j, p in enumerate(detections) if j not in used_detections]
        return matches, unmatched


def _descending_pairs(matrix: np.ndarray) -> list[tuple[int, int]]:
    """``(row, col)`` index pairs ordered by descending value."""
    flat_order = np.argsort(-matrix, axis=None)
    rows, cols = np.unravel_index(flat_order, matrix.shape)
    return [(int(r), int(c)) for r, c in zip(rows, cols, strict=True)]


def _center_affinity(a: BBox, b: BBox, max_ratio: float) -> float:
    """Closeness of two boxes' centres, in ``[0, 1]``, 0 when too far apart.

    The distance is scaled by the boxes' own size so the same threshold
    works for a keeper filling the frame and one thirty metres away.
    """
    ax, ay = a.center
    bx, by = b.center
    distance = float(np.hypot(ax - bx, ay - by))

    # Mean diagonal of the two boxes, a size measure that does not collapse
    # when a body goes horizontal the way height alone would.
    scale = 0.5 * (np.hypot(a.width, a.height) + np.hypot(b.width, b.height))
    if scale <= 0:
        return 0.0

    ratio = distance / scale
    return max(0.0, 1.0 - ratio / max_ratio) if ratio < max_ratio else 0.0


def suppress_duplicates(
    people: list[PersonPose],
    iou_threshold: float = 0.6,
    containment_threshold: float = 0.8,
) -> list[PersonPose]:
    """Drop detections that duplicate a more confident one.

    Multi-person pose models routinely emit a second, partial skeleton for
    one person during rapid motion, when the internal tracker and the
    detector disagree. Left alone the duplicate starts its own track, and a
    single player's dive is split across two ids - which is exactly what the
    goalkeeper analysis must not see.

    Two tests, because one is not enough:

    * **IoU**, which catches two detections of similar size sitting on top
      of each other.
    * **Containment**, which catches the far more common case of a small
      partial skeleton nested inside the real one. Such a pair can have an
      IoU as low as 0.2 while being unambiguously the same person, so an
      IoU-only filter lets it through.

    Detections are considered highest-confidence first, so the survivor of
    each duplicate pair is the better one.
    """
    candidates = sorted(people, key=lambda p: p.mean_confidence(), reverse=True)
    kept: list[PersonPose] = []
    for person in candidates:
        if person.bbox is None:
            kept.append(person)
            continue
        duplicate = any(
            other.bbox is not None
            and (
                person.bbox.iou(other.bbox) > iou_threshold
                or person.bbox.containment(other.bbox) > containment_threshold
            )
            for other in kept
        )
        if not duplicate:
            kept.append(person)
    return kept


def track_sequence(
    frames: list[FramePoses],
    iou_threshold: float = 0.3,
    max_age: int = 30,
    min_hits: int = 3,
) -> list[FramePoses]:
    """Convenience wrapper: track a whole clip with one call."""
    tracker = IoUTracker(
        iou_threshold=iou_threshold, max_age=max_age, min_hits=min_hits
    )
    return tracker.run(frames)
