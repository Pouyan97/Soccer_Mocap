"""Pitch calibration: mapping image pixels onto pitch metres.

A single fixed camera looking at a flat pitch is the one case where monocular
geometry is genuinely easy. The pitch is a plane, so a homography - eight
numbers, recoverable from four point correspondences - maps any image point
on the ground to a pitch coordinate exactly. Pitch markings give those
correspondences for free, because their real dimensions are fixed by the
Laws of the Game.

The catch, and it is the important one: **a homography maps the ground plane
only.** A player's feet are on the plane and transform correctly; their head
is two metres above it and does not. Always project the feet. Projecting a
centre of mass gives a position that is wrong by roughly the player's height
times the tangent of the camera's elevation angle, which at a typical
touchline camera is a couple of metres.

Two things break this model, and both are common in phone footage:

* **Panning or handheld motion.** The homography is per-camera-pose, so any
  movement invalidates it. Use a tripod.
* **Wide-angle distortion.** Barrel distortion bends the straight lines the
  homography assumes. Undistort first, or film at a narrower FOV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..exceptions import CalibrationError

#: Standard pitch, in metres. The Laws permit a range; these are the FIFA
#: preferred dimensions and what broadcast pitches use.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

#: Named landmarks in pitch coordinates: origin at the centre spot, +x toward
#: the right-hand goal, +y toward the far touchline. These are the points a
#: user clicks when calibrating manually.
PITCH_LANDMARKS: dict[str, tuple[float, float]] = {
    "centre_spot": (0.0, 0.0),
    "centre_circle_top": (0.0, 9.15),
    "centre_circle_bottom": (0.0, -9.15),
    "halfway_top": (0.0, PITCH_WIDTH_M / 2),
    "halfway_bottom": (0.0, -PITCH_WIDTH_M / 2),
    # Left goal (defending -x).
    "left_goal_centre": (-PITCH_LENGTH_M / 2, 0.0),
    "left_post_top": (-PITCH_LENGTH_M / 2, 3.66),
    "left_post_bottom": (-PITCH_LENGTH_M / 2, -3.66),
    "left_penalty_spot": (-PITCH_LENGTH_M / 2 + 11.0, 0.0),
    "left_box_top_corner": (-PITCH_LENGTH_M / 2, 20.16),
    "left_box_bottom_corner": (-PITCH_LENGTH_M / 2, -20.16),
    "left_box_top_edge": (-PITCH_LENGTH_M / 2 + 16.5, 20.16),
    "left_box_bottom_edge": (-PITCH_LENGTH_M / 2 + 16.5, -20.16),
    "left_six_top_corner": (-PITCH_LENGTH_M / 2, 9.16),
    "left_six_bottom_corner": (-PITCH_LENGTH_M / 2, -9.16),
    "left_six_top_edge": (-PITCH_LENGTH_M / 2 + 5.5, 9.16),
    "left_six_bottom_edge": (-PITCH_LENGTH_M / 2 + 5.5, -9.16),
    # Corners.
    "corner_top_left": (-PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2),
    "corner_bottom_left": (-PITCH_LENGTH_M / 2, -PITCH_WIDTH_M / 2),
    "corner_top_right": (PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2),
    "corner_bottom_right": (PITCH_LENGTH_M / 2, -PITCH_WIDTH_M / 2),
}


@dataclass
class PitchCalibration:
    """A homography from image pixels to pitch metres.

    Build it with :meth:`from_points` rather than constructing directly.
    """

    #: 3x3 homography mapping homogeneous image points to pitch points.
    homography: np.ndarray
    #: Mean reprojection error over the calibration points, in pixels.
    reprojection_error_px: float = float("nan")
    #: The correspondences used, kept so a calibration can be inspected.
    image_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    pitch_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    def __post_init__(self) -> None:
        self.homography = np.asarray(self.homography, dtype=float)
        if self.homography.shape != (3, 3):
            raise CalibrationError("A homography must be a 3x3 matrix.")

    @property
    def is_reliable(self) -> bool:
        """Whether the fit is tight enough to act on.

        Five pixels is a pragmatic bar: below it the dominant error is the
        user's clicking precision rather than the model.
        """
        return np.isfinite(self.reprojection_error_px) and self.reprojection_error_px < 5.0

    @classmethod
    def from_points(
        cls,
        image_points: np.ndarray | list[list[float]],
        pitch_points: np.ndarray | list[list[float]],
    ) -> PitchCalibration:
        """Fit a homography from at least four correspondences.

        Args:
            image_points: ``(N, 2)`` pixel coordinates, on the ground plane.
            pitch_points: ``(N, 2)`` matching pitch coordinates in metres.

        Raises:
            CalibrationError: with fewer than four points, or if the points
                are degenerate (three or more collinear).
        """
        image = np.asarray(image_points, dtype=float)
        pitch = np.asarray(pitch_points, dtype=float)

        if image.shape != pitch.shape or image.ndim != 2 or image.shape[1] != 2:
            raise CalibrationError(
                f"Point arrays must both be (N, 2); got {image.shape} and {pitch.shape}."
            )
        if len(image) < 4:
            raise CalibrationError(
                f"A homography needs at least 4 points, got {len(image)}. "
                "Pitch corners, penalty box corners and the centre circle are "
                "the easiest to click accurately."
            )

        homography = _solve_homography(image, pitch)
        error = _reprojection_error(homography, image, pitch)
        return cls(
            homography=homography,
            reprojection_error_px=error,
            image_points=image,
            pitch_points=pitch,
        )

    @classmethod
    def from_landmarks(
        cls, correspondences: dict[str, tuple[float, float]]
    ) -> PitchCalibration:
        """Fit from ``{landmark_name: (image_x, image_y)}``.

        Convenient for the demo app, where the user picks a named landmark
        from a list and clicks it.
        """
        unknown = set(correspondences) - set(PITCH_LANDMARKS)
        if unknown:
            raise CalibrationError(
                f"Unknown pitch landmark(s): {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(PITCH_LANDMARKS))}"
            )
        names = list(correspondences)
        image = np.array([correspondences[n] for n in names], dtype=float)
        pitch = np.array([PITCH_LANDMARKS[n] for n in names], dtype=float)
        return cls.from_points(image, pitch)

    # -- projection -------------------------------------------------------

    def to_pitch(self, image_xy: np.ndarray) -> np.ndarray:
        """Project ground-plane image points to pitch metres.

        Args:
            image_xy: ``(2,)`` or ``(N, 2)`` pixel coordinates. These must be
                points on the ground - a player's feet, not their head.

        Returns:
            Matching pitch coordinates in metres.
        """
        return _apply_homography(self.homography, image_xy)

    def to_image(self, pitch_xy: np.ndarray) -> np.ndarray:
        """Project pitch metres back into image pixels."""
        return _apply_homography(np.linalg.inv(self.homography), pitch_xy)

    def distance_m(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
        """Ground distance in metres between two image points."""
        a, b = self.to_pitch(image_a), self.to_pitch(image_b)
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def to_dict(self) -> dict[str, object]:
        return {
            "homography": self.homography.tolist(),
            "reprojection_error_px": self.reprojection_error_px,
            "is_reliable": self.is_reliable,
            "num_points": len(self.image_points),
        }


def _solve_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Direct Linear Transform, with normalisation.

    Uses OpenCV's RANSAC fit when available, since it rejects a mis-clicked
    point; falls back to a plain normalised DLT otherwise, so the package
    still works without cv2.
    """
    try:
        import cv2

        homography, _ = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        if homography is not None:
            return np.asarray(homography, dtype=float)
    except ImportError:
        pass

    # Hartley normalisation: without it the DLT is badly conditioned, because
    # pixel coordinates are ~1e3 while pitch coordinates are ~1e1.
    source_norm, transform_s = _normalise(source)
    target_norm, transform_t = _normalise(target)

    rows = []
    for (x, y), (u, v) in zip(source_norm, target_norm, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    _, _, vt = np.linalg.svd(np.array(rows))
    homography_norm = vt[-1].reshape(3, 3)

    homography = np.linalg.inv(transform_t) @ homography_norm @ transform_s
    if abs(homography[2, 2]) < 1e-12:
        raise CalibrationError(
            "Degenerate point configuration; the homography is singular. "
            "Check that no three calibration points are collinear."
        )
    return homography / homography[2, 2]


def _normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate to the centroid and scale so the mean distance is sqrt(2)."""
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_distance = float(np.mean(np.linalg.norm(centred, axis=1)))
    if mean_distance < 1e-12:
        raise CalibrationError("All calibration points are coincident.")
    scale = np.sqrt(2) / mean_distance
    transform = np.array(
        [
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1],
        ]
    )
    return centred * scale, transform


def _apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to ``(2,)`` or ``(N, 2)`` points."""
    array = np.asarray(points, dtype=float)
    single = array.ndim == 1
    if single:
        array = array[None, :]

    homogeneous = np.hstack([array, np.ones((len(array), 1))])
    projected = homogeneous @ homography.T

    w = projected[:, 2:3]
    # Points on the camera's horizon project to infinity; report them as NaN
    # rather than as a very large, very wrong coordinate.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(w) > 1e-9, projected[:, :2] / w, np.nan)

    return result[0] if single else result


def _reprojection_error(
    homography: np.ndarray, image: np.ndarray, pitch: np.ndarray
) -> float:
    """Mean error in pixels when pitch points are mapped back to the image."""
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return float("nan")
    reprojected = _apply_homography(inverse, pitch)
    return float(np.nanmean(np.linalg.norm(reprojected - image, axis=1)))


def feet_position(xy: np.ndarray, confidence: np.ndarray | None = None) -> np.ndarray:
    """The ground-contact point of a pose, for projecting onto the pitch.

    Midpoint of the ankles, which is the lowest reliably-tracked landmark and
    the only one actually on the ground plane.
    """
    from ..skeleton import J

    points = []
    for index in (J.LEFT_ANKLE, J.RIGHT_ANKLE):
        if index >= len(xy):
            continue
        point = xy[index]
        if not np.isfinite(point).all():
            continue
        if confidence is not None and confidence[index] < 0.2:
            continue
        points.append(point)

    if not points:
        return np.array([np.nan, np.nan])
    return np.mean(points, axis=0)
