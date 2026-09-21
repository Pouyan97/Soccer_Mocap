"""Deciding which track is the goalkeeper.

Three strategies, in descending order of how much they can be trusted:

1. **Manual.** The user says which track id. Always correct, and the demo app
   makes it one click, so it is the default when a person is present.
2. **Geometric.** With the goal marked, the keeper is the track that stays
   nearest the goal and covers the least ground. This is reliable for
   keeper-focused footage and is what the CLI uses.
3. **Kit colour.** Goalkeepers are required to wear a colour distinct from
   both teams and from the referee, which makes the keeper the colour
   outlier among tracks. Useful when no goal is marked, but it fails on
   training footage where everyone wears the same bib.

Each returns a :class:`GoalkeeperCandidate` with a score and the reasoning,
rather than a bare id, so a wrong pick is visible instead of mysterious.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analysis.kinematics import center_of_mass
from ..types import Track
from .goal import GoalGeometry


@dataclass
class GoalkeeperCandidate:
    """One track, scored for how goalkeeper-like it is."""

    track_id: int
    score: float
    #: Human-readable justification, shown in the demo app.
    reasons: list[str] = field(default_factory=list)
    #: Raw measurements behind the score.
    metrics: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"track {self.track_id}: {self.score:.2f} ({'; '.join(self.reasons)})"


def _track_com_path(track: Track) -> np.ndarray:
    """Centre-of-mass position per observed frame, as ``(N, 2)``."""
    points = []
    for frame_index in track.frame_indices:
        pose = track.poses[frame_index]
        com = center_of_mass(pose.xy, pose.confidence)
        if np.isfinite(com).all():
            points.append(com)
    return np.array(points) if points else np.empty((0, 2))


def score_by_geometry(
    tracks: dict[int, Track],
    goal: GoalGeometry,
    min_frames: int = 10,
) -> list[GoalkeeperCandidate]:
    """Score tracks by how they sit relative to the goal.

    Three signals, combined:

    * **Proximity.** Median distance from the goal line. Keepers live within
      a few metres of it; outfield players pass through.
    * **Containment.** Fraction of frames spent within the goal's width.
    * **Stillness.** Total ground covered, normalised by duration. A keeper
      shuffles; a winger sprints.
    """
    candidates: list[GoalkeeperCandidate] = []

    for track_id, track in tracks.items():
        path = _track_com_path(track)
        if len(path) < min_frames:
            continue

        depths = np.array([goal.distance_off_line_m(p) for p in path])
        laterals = np.array([abs(goal.lateral_offset_m(p)) for p in path])

        median_depth = float(np.median(depths))
        # Keepers operate roughly between the line and the edge of the box.
        # Score peaks on the line and decays over the box depth.
        proximity = float(np.exp(-max(0.0, median_depth) / 8.0))

        within_mouth = float(np.mean(laterals <= (7.32 / 2 + 5.5)))

        step = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.zeros(1)
        distance_m = float(step.sum()) / goal.px_per_m
        per_frame = distance_m / max(1, len(path))
        stillness = float(np.exp(-per_frame / 0.05))

        # Proximity dominates: it is the single most diagnostic signal, and
        # the other two mostly break ties between a keeper and a centre-back
        # who is camped on the line for a corner.
        score = 0.55 * proximity + 0.25 * within_mouth + 0.20 * stillness

        reasons = [
            f"median {median_depth:.1f} m off the goal line",
            f"{100 * within_mouth:.0f}% of frames within the goal area width",
            f"covered {distance_m:.1f} m over {len(path)} frames",
        ]
        candidates.append(
            GoalkeeperCandidate(
                track_id=track_id,
                score=score,
                reasons=reasons,
                metrics={
                    "median_depth_m": median_depth,
                    "within_mouth_fraction": within_mouth,
                    "distance_covered_m": distance_m,
                    "frames": float(len(path)),
                },
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def score_by_kit_colour(
    tracks: dict[int, Track],
    frames: dict[int, np.ndarray],
    min_frames: int = 5,
) -> list[GoalkeeperCandidate]:
    """Score tracks by how far their kit colour sits from everyone else's.

    Samples the torso region of each track, takes the median hue, and ranks
    tracks by distance from the population median. The keeper is required by
    the laws to be distinguishable, so they are usually the outlier.

    Args:
        tracks: tracks to score.
        frames: ``{frame_index: RGB image}``. Only a sample is needed; pass
            every tenth frame rather than the whole clip.
    """
    torso_colours: dict[int, np.ndarray] = {}

    for track_id, track in tracks.items():
        samples = []
        for frame_index in track.frame_indices:
            image = frames.get(frame_index)
            if image is None:
                continue
            pose = track.poses[frame_index]
            patch = _torso_patch(image, pose.xy)
            if patch is not None and patch.size:
                samples.append(np.median(patch.reshape(-1, 3), axis=0))
        if len(samples) >= min_frames:
            torso_colours[track_id] = np.median(np.vstack(samples), axis=0)

    if len(torso_colours) < 2:
        return []

    population = np.median(np.vstack(list(torso_colours.values())), axis=0)
    distances = {
        track_id: float(np.linalg.norm(colour - population))
        for track_id, colour in torso_colours.items()
    }
    furthest = max(distances.values()) or 1.0

    candidates = [
        GoalkeeperCandidate(
            track_id=track_id,
            score=distance / furthest,
            reasons=[
                f"kit colour RGB{tuple(int(c) for c in torso_colours[track_id])} "
                f"is {distance:.0f} from the squad median"
            ],
            metrics={"colour_distance": distance},
        )
        for track_id, distance in distances.items()
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _torso_patch(image: np.ndarray, xy: np.ndarray) -> np.ndarray | None:
    """Crop the shirt region between the shoulders and the hips."""
    from ..skeleton import J

    corners = xy[[J.LEFT_SHOULDER, J.RIGHT_SHOULDER, J.LEFT_HIP, J.RIGHT_HIP]]
    if not np.isfinite(corners).all():
        return None

    height, width = image.shape[:2]
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    # Inset by 20% so the patch is shirt rather than background at the edges.
    inset_x, inset_y = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
    x1, x2 = int(max(0, x1 + inset_x)), int(min(width, x2 - inset_x))
    y1, y2 = int(max(0, y1 + inset_y)), int(min(height, y2 - inset_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def identify_goalkeeper(
    tracks: dict[int, Track],
    goal: GoalGeometry | None = None,
    frames: dict[int, np.ndarray] | None = None,
    method: str = "auto",
    track_id: int | None = None,
) -> tuple[int | None, list[GoalkeeperCandidate]]:
    """Pick the goalkeeper track.

    Args:
        tracks: candidate tracks.
        goal: goal geometry, required for the geometric method.
        frames: sampled frames, required for the colour method.
        method: ``"manual"``, ``"geometry"``, ``"colour"``, or ``"auto"``,
            which uses the best strategy the available inputs support.
        track_id: the answer, when ``method="manual"``.

    Returns:
        ``(track_id, ranked_candidates)``. The id is ``None`` when no
        strategy could be applied, which the caller should surface rather
        than guess past.
    """
    if method == "manual" or track_id is not None:
        if track_id is None:
            return None, []
        return track_id, [
            GoalkeeperCandidate(track_id, 1.0, ["selected manually"])
        ]

    if not tracks:
        return None, []

    if method in ("auto", "geometry") and goal is not None:
        candidates = score_by_geometry(tracks, goal)
        if candidates:
            return candidates[0].track_id, candidates

    if method in ("auto", "colour") and frames:
        candidates = score_by_kit_colour(tracks, frames)
        if candidates:
            return candidates[0].track_id, candidates

    if method == "auto":
        # Nothing diagnostic available. The longest track is the best
        # remaining guess for single-subject footage, and it is flagged as a
        # weak inference so the caller can warn.
        longest = max(tracks.values(), key=len)
        return longest.track_id, [
            GoalkeeperCandidate(
                longest.track_id,
                0.3,
                ["no goal marked and no colour cue; fell back to the longest track"],
            )
        ]

    return None, []
