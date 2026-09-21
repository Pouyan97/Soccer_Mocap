"""Goal geometry: the one object of known size in the frame.

A goal is 7.32 m wide and 2.44 m high by law, everywhere, always. In a scene
where nothing else has a reliable dimension, that makes the goal mouth the
most valuable calibration target available - two clicked posts give a metric
scale, an approximate ground plane, and the reference line that every
goalkeeper positioning metric is measured against.

This module deliberately asks for the posts rather than detecting them.
Automatic goal detection from a single frame is its own research problem, and
getting it subtly wrong silently corrupts every downstream metric. Two clicks
in the demo app is a better trade. :func:`detect_goal_posts` is left as the
seam for a detector when one is warranted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import CalibrationError

#: Laws of the Game: 8 yards between the posts, 8 feet to the crossbar.
GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44
#: Six-yard box, measured from each post and out from the goal line.
GOAL_AREA_DEPTH_M = 5.5
#: Eighteen-yard box.
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_SPOT_DISTANCE_M = 11.0


@dataclass
class GoalGeometry:
    """The goal mouth in image space, plus the scale it implies.

    Args:
        left_post: ``(x, y)`` of the left post where it meets the ground.
        right_post: ``(x, y)`` of the right post where it meets the ground.
        crossbar_y: image ``y`` of the crossbar. Optional; when supplied it
            gives a second, independent scale estimate that is a useful
            sanity check on the first.

    "Left" and "right" are as seen by the camera, not by the keeper.
    """

    left_post: np.ndarray
    right_post: np.ndarray
    crossbar_y: float | None = None

    def __post_init__(self) -> None:
        self.left_post = np.asarray(self.left_post, dtype=float)
        self.right_post = np.asarray(self.right_post, dtype=float)
        if self.left_post.shape != (2,) or self.right_post.shape != (2,):
            raise CalibrationError("Goal posts must each be an (x, y) pair.")
        # Enforce the naming so downstream sign conventions hold.
        if self.left_post[0] > self.right_post[0]:
            self.left_post, self.right_post = self.right_post, self.left_post

    # -- basic geometry ---------------------------------------------------

    @property
    def width_px(self) -> float:
        """Distance between the posts in pixels."""
        return float(np.linalg.norm(self.right_post - self.left_post))

    @property
    def center(self) -> np.ndarray:
        """Midpoint of the goal line, in image space."""
        return 0.5 * (self.left_post + self.right_post)

    @property
    def px_per_m(self) -> float:
        """Image scale at the goal line, from the known 7.32 m goal width.

        Valid only near the goal line. Perspective makes it wrong anywhere
        else, which is why positioning metrics that need accuracy off the
        line should go through a full homography instead.
        """
        width = self.width_px
        if width <= 0:
            raise CalibrationError("Goal posts coincide; cannot derive a scale.")
        return width / GOAL_WIDTH_M

    @property
    def height_px(self) -> float:
        """Crossbar height in pixels, measured if known, else inferred."""
        if self.crossbar_y is not None:
            return abs(float(self.center[1] - self.crossbar_y))
        return GOAL_HEIGHT_M * self.px_per_m

    @property
    def goal_line_direction(self) -> np.ndarray:
        """Unit vector along the goal line, pointing left post -> right post."""
        delta = self.right_post - self.left_post
        norm = np.linalg.norm(delta)
        if norm == 0:
            raise CalibrationError("Goal posts coincide; goal line is undefined.")
        return delta / norm

    @property
    def goal_line_normal(self) -> np.ndarray:
        """Unit normal to the goal line, pointing out into the field of play.

        Image ``y`` grows downward and the field is below the goal line in a
        typical behind-the-goal or side view, so the normal is taken as the
        direction with positive ``y``.
        """
        along = self.goal_line_direction
        normal = np.array([-along[1], along[0]])
        return normal if normal[1] > 0 else -normal

    def scale_consistency(self) -> float | None:
        """Ratio of the crossbar-derived scale to the post-derived one.

        ``1.0`` means the two agree. Values far from 1 mean the clicked
        points are wrong or the lens distortion is severe. ``None`` when no
        crossbar was given.
        """
        if self.crossbar_y is None:
            return None
        measured = abs(float(self.center[1] - self.crossbar_y))
        expected = GOAL_HEIGHT_M * self.px_per_m
        return measured / expected if expected > 0 else None

    # -- measurements against the goal ------------------------------------

    def signed_lateral_offset_px(self, point: np.ndarray) -> float:
        """Distance from the goal centre along the goal line, in pixels.

        Negative toward the left post, positive toward the right. This is the
        raw material for "is the keeper covering their near post".
        """
        return float(np.dot(np.asarray(point, dtype=float) - self.center,
                            self.goal_line_direction))

    def distance_off_line_px(self, point: np.ndarray) -> float:
        """Perpendicular distance out from the goal line, in pixels.

        Positive means off the line into the field, which is the normal case
        for a keeper; negative means behind it, i.e. inside the goal.
        """
        return float(np.dot(np.asarray(point, dtype=float) - self.center,
                            self.goal_line_normal))

    def lateral_offset_m(self, point: np.ndarray) -> float:
        """:meth:`signed_lateral_offset_px` converted to metres."""
        return self.signed_lateral_offset_px(point) / self.px_per_m

    def distance_off_line_m(self, point: np.ndarray) -> float:
        """:meth:`distance_off_line_px` in metres.

        Least trustworthy of the metric conversions: it applies a scale
        measured at the goal line to a displacement away from it, so it grows
        optimistic as the keeper advances. Good to a metre or so for a keeper
        on their line; use a homography past that.
        """
        return self.distance_off_line_px(point) / self.px_per_m

    def coverage_fraction(self, point: np.ndarray) -> float:
        """Where the keeper stands across the mouth, as a fraction.

        ``0.0`` at the left post, ``0.5`` at centre, ``1.0`` at the right
        post. Values outside ``[0, 1]`` mean they are wider than the frame of
        the goal.
        """
        return 0.5 + self.lateral_offset_m(point) / GOAL_WIDTH_M

    def is_in_goal_area(self, point: np.ndarray) -> bool:
        """Whether a point falls inside the six-yard box.

        An approximation: it treats the box as a rectangle in image space,
        which perspective does not respect. Adequate for tagging, not for
        measurement.
        """
        lateral = abs(self.lateral_offset_m(point))
        depth = self.distance_off_line_m(point)
        return lateral <= (GOAL_WIDTH_M / 2 + GOAL_AREA_DEPTH_M) and (
            0 <= depth <= GOAL_AREA_DEPTH_M
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "left_post": self.left_post.tolist(),
            "right_post": self.right_post.tolist(),
            "crossbar_y": self.crossbar_y,
            "width_px": self.width_px,
            "px_per_m": self.px_per_m,
            "scale_consistency": self.scale_consistency(),
        }

    @classmethod
    def from_config(cls, posts: list[list[float]] | None, crossbar_y: float | None = None):
        """Build from the ``goal_posts`` entry of a config file."""
        if not posts or len(posts) != 2:
            raise CalibrationError(
                "goalkeeper.goal_posts needs exactly two [x, y] points: "
                "the foot of each post."
            )
        return cls(np.array(posts[0], dtype=float), np.array(posts[1], dtype=float), crossbar_y)


def detect_goal_posts(image: np.ndarray) -> GoalGeometry | None:
    """Automatic goal detection - not implemented.

    The seam for a real detector. A workable approach is a Hough transform
    over the white-pixel mask to find the two near-vertical post lines and
    the horizontal crossbar, then take their intersections. It is left out
    because a confidently wrong goal silently corrupts every metric that
    depends on it, and the demo app can ask for two clicks instead.

    Returns:
        Always ``None``, meaning "no automatic detection available".
    """
    return None
