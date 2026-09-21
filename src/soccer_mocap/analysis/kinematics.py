"""Joint angles, velocities, and the scale normalisation that makes them comparable.

Two things here are worth understanding before trusting any number that comes
out of this module.

**Everything is in pixels unless calibration has run.** A single uncalibrated
camera cannot recover metric scale. Rather than pretend otherwise, speeds and
displacements are normalised by the subject's own pixel height, giving
*body-heights per second*. That unit is dimensionless, comparable between
clips shot at different distances, and converts to m/s by multiplying by the
player's real height. Functions that return pixel units say so in the name.

**Differentiating noisy positions amplifies the noise.** Pose keypoints
jitter by a few pixels per frame; a raw central difference turns that into
large spurious velocities, and a second difference makes accelerations
meaningless. Filter before differentiating - :func:`filter_trajectory` is
there for that, and the velocity helpers apply it by default.
"""

from __future__ import annotations

import numpy as np

from ..skeleton import J


def filter_trajectory(
    xy: np.ndarray,
    fps: float,
    cutoff_hz: float | None = 6.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-lag low-pass filter over time.

    A 4th-order Butterworth applied forwards and backwards, which is the
    standard choice in gait analysis: zero phase shift means event timing is
    preserved, which matters when the thing being measured is when a foot
    struck a ball.

    6 Hz is the conventional cutoff for whole-body human movement. Raise it
    for fast limb segments - a kicking shank has real content well above
    6 Hz, and over-filtering will clip the peak.

    Args:
        xy: ``(T, ...)`` array; time must be the first axis.
        fps: sampling rate.
        cutoff_hz: ``None`` returns the input untouched.
        order: filter order, applied twice by filtfilt.

    Returns:
        Array of the same shape. NaN runs are interpolated before filtering
        and restored afterwards, since filtfilt propagates NaN across the
        whole signal.
    """
    if cutoff_hz is None or len(xy) < 2:
        return xy

    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        return xy

    nyquist = 0.5 * fps
    if cutoff_hz >= nyquist:
        # Nothing to remove; filtering here would only distort the signal.
        return xy

    # filtfilt needs more samples than 3 * max(len(a), len(b)).
    if len(xy) <= 3 * (order + 1):
        return xy

    original_shape = xy.shape
    flat = xy.reshape(len(xy), -1).astype(float)
    filled, missing = _interpolate_gaps(flat)

    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    smoothed = filtfilt(b, a, filled, axis=0)

    smoothed[missing] = np.nan
    return smoothed.reshape(original_shape)


def _interpolate_gaps(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill NaN runs column-wise; return the filled array and the mask."""
    filled = flat.copy()
    missing = ~np.isfinite(filled)
    time_index = np.arange(len(filled))

    for column in range(filled.shape[1]):
        column_missing = missing[:, column]
        if not column_missing.any():
            continue
        valid = ~column_missing
        if valid.sum() < 2:
            # Not enough to interpolate; hold a constant so filtfilt runs.
            filled[:, column] = filled[valid, column][0] if valid.any() else 0.0
            continue
        filled[column_missing, column] = np.interp(
            time_index[column_missing], time_index[valid], filled[valid, column]
        )
    return filled, missing


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at ``b`` in the chain a-b-c, in degrees.

    For a knee that is ``angle(hip, knee, ankle)``: 180 degrees is fully
    extended, smaller values are more flexed.
    """
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return float("nan")

    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def joint_angle_series(
    xy: np.ndarray, a_index: int, b_index: int, c_index: int
) -> np.ndarray:
    """:func:`joint_angle` over a ``(T, J, 2)`` trajectory."""
    return np.array(
        [joint_angle(frame[a_index], frame[b_index], frame[c_index]) for frame in xy]
    )


def segment_angle_from_vertical(proximal: np.ndarray, distal: np.ndarray) -> float:
    """Angle of a body segment away from vertical, in degrees.

    Zero is upright. Sign follows image space: positive leans toward +x
    (screen right). Used for torso lean, which is what separates a
    goalkeeper's dive from a sidestep.
    """
    if not (np.isfinite(proximal).all() and np.isfinite(distal).all()):
        return float("nan")
    dx = float(distal[0] - proximal[0])
    # Image y grows downward, so flip it to get a normal upward axis.
    dy = float(proximal[1] - distal[1])
    if dx == 0 and dy == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(dx, abs(dy) if dy != 0 else 1e-9)))


#: Segment lengths as a fraction of stature, from Winter (2009).
#: Keyed by the segment they span along the head-to-foot chain.
_SEGMENT_FRACTIONS = {
    "nose_shoulder": 0.12,
    "shoulder_hip": 0.29,
    "hip_knee": 0.245,
    "knee_ankle": 0.246,
}

#: Below this, a "person" is too small for any derived quantity to mean
#: anything, and dividing by it produces the kind of number that looks like
#: a bug report.
_MIN_PLAUSIBLE_HEIGHT_PX = 20.0


def body_height_px(xy: np.ndarray, confidence: np.ndarray | None = None) -> float:
    """Estimate stature in pixels, for scale normalisation.

    Measured as the summed length of the body's segments - nose to shoulder
    to hip to knee to ankle - rather than as the vertical extent of the
    bounding box. Two reasons, both of which bite hard in this application:

    * **Orientation.** A goalkeeper mid-dive is horizontal. Vertical extent
      collapses toward zero exactly when the interesting things are
      happening, and every speed normalised by it explodes.
    * **Posture.** A keeper in a set position is crouched, so even upright
      their vertical extent understates their stature by 10-20%.

    Segment lengths are invariant to both. Whichever segments are visible
    are summed and divided by the fraction of stature they represent, so a
    partial skeleton still gives a usable estimate - which matters, because
    phone footage cuts feet off constantly.

    Returns:
        Stature in pixels, or NaN when too little of the body is visible or
        the result is implausibly small.
    """

    def usable(index: int) -> np.ndarray | None:
        if index >= len(xy):
            return None
        point = xy[index]
        if not np.isfinite(point).all():
            return None
        if confidence is not None and confidence[index] < 0.2:
            return None
        return point

    def midpoint(a: int, b: int) -> np.ndarray | None:
        left, right = usable(a), usable(b)
        if left is not None and right is not None:
            return 0.5 * (left + right)
        return left if left is not None else right

    nose = usable(J.NOSE)
    shoulder = midpoint(J.LEFT_SHOULDER, J.RIGHT_SHOULDER)
    hip = midpoint(J.LEFT_HIP, J.RIGHT_HIP)
    knee = midpoint(J.LEFT_KNEE, J.RIGHT_KNEE)
    ankle = midpoint(J.LEFT_ANKLE, J.RIGHT_ANKLE)

    chain = (
        (nose, shoulder, _SEGMENT_FRACTIONS["nose_shoulder"]),
        (shoulder, hip, _SEGMENT_FRACTIONS["shoulder_hip"]),
        (hip, knee, _SEGMENT_FRACTIONS["hip_knee"]),
        (knee, ankle, _SEGMENT_FRACTIONS["knee_ankle"]),
    )

    total_length = 0.0
    total_fraction = 0.0
    for proximal, distal, fraction in chain:
        if proximal is None or distal is None:
            continue
        total_length += float(np.linalg.norm(distal - proximal))
        total_fraction += fraction

    if total_fraction <= 0:
        return float("nan")

    height = total_length / total_fraction
    return height if height >= _MIN_PLAUSIBLE_HEIGHT_PX else float("nan")


def reference_height_px(
    heights: np.ndarray | list[float], min_samples: int = 3
) -> float:
    """Collapse per-frame stature estimates into one value for a track.

    A person's stature does not change during a clip, so normalising each
    frame by its own noisy estimate is strictly worse than normalising every
    frame by one robust estimate: it injects the pose model's jitter into
    the denominator, and a frame where the model briefly loses the legs
    produces a velocity spike of a hundred body-heights per second.

    The median is used rather than the mean because the error distribution
    is one-sided - occlusion and self-overlap shorten the apparent body,
    they never lengthen it - so the mean is dragged low by a tail of bad
    frames while the median is not.

    Returns:
        A single stature in pixels, or NaN if too few frames were usable.
    """
    array = np.asarray(heights, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size < min_samples:
        return float(np.median(finite)) if finite.size else float("nan")
    return float(np.median(finite))


def center_of_mass(xy: np.ndarray, confidence: np.ndarray | None = None) -> np.ndarray:
    """Approximate whole-body centre of mass in image space.

    A weighted mean of the trunk landmarks rather than a full segmental
    model: with 17 keypoints and no segment masses, the extra machinery of a
    proper Dempster model would not buy accuracy it can justify. The trunk
    carries most of body mass, so this tracks the true COM closely enough for
    the displacement and velocity measures used here.
    """
    indices = [J.LEFT_SHOULDER, J.RIGHT_SHOULDER, J.LEFT_HIP, J.RIGHT_HIP]
    weights = np.array([0.15, 0.15, 0.35, 0.35])

    points, used_weights = [], []
    for index, weight in zip(indices, weights, strict=True):
        if index >= len(xy):
            continue
        point = xy[index]
        if not np.isfinite(point).all():
            continue
        if confidence is not None and confidence[index] < 0.2:
            continue
        points.append(point)
        used_weights.append(weight)

    if not points:
        return np.array([np.nan, np.nan])

    stacked = np.vstack(points)
    weight_array = np.array(used_weights)
    return (stacked * weight_array[:, None]).sum(axis=0) / weight_array.sum()


def com_series(xy: np.ndarray, confidence: np.ndarray | None = None) -> np.ndarray:
    """:func:`center_of_mass` over a ``(T, J, 2)`` trajectory -> ``(T, 2)``."""
    return np.array(
        [
            center_of_mass(xy[t], confidence[t] if confidence is not None else None)
            for t in range(len(xy))
        ]
    )


def velocity(
    positions: np.ndarray,
    fps: float,
    filter_cutoff_hz: float | None = 6.0,
) -> np.ndarray:
    """Central-difference velocity in units of ``positions`` per second.

    Filtering happens before differentiating, not after, which is the order
    that matters: differentiating first has already turned the noise into
    signal by the time a filter sees it.
    """
    if len(positions) < 2:
        return np.zeros_like(positions)
    smoothed = filter_trajectory(positions, fps, filter_cutoff_hz)
    return np.gradient(smoothed, 1.0 / fps, axis=0)


def speed(
    positions: np.ndarray,
    fps: float,
    filter_cutoff_hz: float | None = 6.0,
) -> np.ndarray:
    """Scalar speed, i.e. the magnitude of :func:`velocity`."""
    return np.linalg.norm(velocity(positions, fps, filter_cutoff_hz), axis=-1)


def normalize_by_height(
    values: np.ndarray, height_px: float | np.ndarray
) -> np.ndarray:
    """Convert pixel-unit values to body-heights, the scale-free unit.

    Multiply the result by the player's real stature in metres to get SI
    units. ``height_px`` may be a per-frame array.
    """
    height = np.asarray(height_px, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = values / height if height.ndim == 0 else values / height[..., None]
    return np.where(np.isfinite(out), out, np.nan)
