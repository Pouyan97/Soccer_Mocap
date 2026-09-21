"""Event detection for outfield play: kicks and ground contacts.

Both events are inferred from the feet alone, with no ball tracking. That is
a deliberate limitation rather than an oversight - a ball in phone footage is
small, fast, frequently motion-blurred into invisibility, and often occluded
by the player striking it, so a ball detector is its own project. Foot
kinematics are already available from the pose stage and are enough to time
the events, which is what the biomechanics needs.

What this costs you: a kick is detected from the *swing*, so a mimed strike
looks identical to a real one, and the exact instant of ball contact is
inferred from peak foot speed rather than measured. Expect timing good to
about a frame, and treat the events as candidates to review on the overlay
rather than as ground truth.

Everything is normalised by the player's stature, so thresholds are in
body-heights per second and work at any camera distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..skeleton import J
from ..types import Track
from .kinematics import body_height_px, filter_trajectory, reference_height_px


class Foot(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class KickEvent:
    """A candidate ball strike, timed from peak foot speed."""

    frame_index: int
    time_s: float
    foot: Foot
    #: Peak foot speed, in body-heights per second.
    peak_speed_bh_s: float
    #: How sharply the foot decelerated afterwards. A real contact stops the
    #: foot; a mimed swing lets it follow through, so this separates them
    #: better than peak speed alone.
    deceleration_bh_s2: float
    #: Mean joint confidence around the event.
    confidence: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "foot": self.foot.value,
            "peak_speed_bh_s": self.peak_speed_bh_s,
            "deceleration_bh_s2": self.deceleration_bh_s2,
            "confidence": self.confidence,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        return (
            f"{self.foot.value} foot strike at {self.time_s:.2f} s, "
            f"peak {self.peak_speed_bh_s:.1f} bh/s"
        )


@dataclass
class ContactEvent:
    """A foot touching down, for step timing and cadence."""

    frame_index: int
    time_s: float
    foot: Foot

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "foot": self.foot.value,
        }


def _foot_series(
    track: Track, fps: float, joint: int, filter_cutoff_hz: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Return ``(frames, position_px, confidence, reference_height)`` for one ankle."""
    frames, positions, confidences, heights = [], [], [], []

    for frame_index in track.frame_indices:
        pose = track.poses[frame_index]
        point = pose.joint(joint, min_confidence=0.2)
        height = body_height_px(pose.xy, pose.confidence)
        if point is None or not np.isfinite(height) or height <= 0:
            continue
        frames.append(frame_index)
        positions.append(point)
        confidences.append(float(pose.confidence[joint]))
        heights.append(height)

    if len(frames) < 8:
        return None

    reference = reference_height_px(heights)
    if not np.isfinite(reference) or reference <= 0:
        return None

    # A kicking shank carries real content well above the 6 Hz used for
    # whole-body motion, so the default cutoff here is higher; filtering at
    # 6 Hz would clip the very peak the detector is looking for.
    smoothed = filter_trajectory(np.array(positions), fps, filter_cutoff_hz)
    return np.array(frames), smoothed, np.array(confidences), reference


def detect_kicks(
    track: Track,
    fps: float,
    speed_threshold: float = 3.0,
    min_separation_s: float = 0.4,
    filter_cutoff_hz: float | None = 12.0,
) -> list[KickEvent]:
    """Find candidate ball strikes from foot speed peaks.

    Args:
        track: the player's track.
        fps: frame rate. At 30 fps a strike spans two or three frames, so
            peak speed is underestimated; 120 fps or better is wanted for
            anything quantitative about the strike itself.
        speed_threshold: minimum peak foot speed in body-heights per second.
            3.0 separates a strike from a running stride for an adult.
        min_separation_s: refractory period, so one strike is not reported
            several times from adjacent frames of the same peak.
        filter_cutoff_hz: low-pass before differentiating.

    Returns:
        Events in chronological order.
    """
    events: list[KickEvent] = []

    for foot, joint in ((Foot.LEFT, J.LEFT_ANKLE), (Foot.RIGHT, J.RIGHT_ANKLE)):
        series = _foot_series(track, fps, joint, filter_cutoff_hz)
        if series is None:
            continue
        frames, positions, confidences, reference = series

        time_s = frames / fps
        velocity = np.gradient(positions / reference, time_s, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        acceleration = np.gradient(speed, time_s)

        for index in _local_peaks(speed, speed_threshold):
            # Deceleration is measured over the frames after the peak: a real
            # contact arrests the foot, a mimed swing does not.
            tail = slice(index, min(len(speed), index + max(2, int(0.05 * fps))))
            deceleration = float(np.min(acceleration[tail])) if tail.stop > tail.start else 0.0

            events.append(
                KickEvent(
                    frame_index=int(frames[index]),
                    time_s=float(time_s[index]),
                    foot=foot,
                    peak_speed_bh_s=float(speed[index]),
                    deceleration_bh_s2=deceleration,
                    confidence=(
                        float(np.mean(confidences[tail]))
                        if tail.stop > tail.start
                        else float("nan")
                    ),
                )
            )

    events.sort(key=lambda e: e.time_s)
    return _enforce_separation(events, min_separation_s)


def _local_peaks(signal: np.ndarray, threshold: float) -> list[int]:
    """Indices of local maxima above ``threshold``."""
    peaks: list[int] = []
    for i in range(1, len(signal) - 1):
        if signal[i] < threshold:
            continue
        if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)
    return peaks


def _enforce_separation(events: list[KickEvent], min_separation_s: float) -> list[KickEvent]:
    """Keep the fastest event within each refractory window."""
    kept: list[KickEvent] = []
    for event in events:
        if kept and (event.time_s - kept[-1].time_s) < min_separation_s:
            if event.peak_speed_bh_s > kept[-1].peak_speed_bh_s:
                kept[-1] = event
            continue
        kept.append(event)
    return kept


def detect_foot_contacts(
    track: Track,
    fps: float,
    filter_cutoff_hz: float | None = 8.0,
) -> list[ContactEvent]:
    """Find ground contacts, for step timing and cadence.

    A contact is taken as a local minimum of the ankle's height - that is, a
    local *maximum* of image ``y``, since image y grows downward - where the
    foot is also moving slowly. Without a calibrated ground plane there is no
    absolute height to threshold against, so this detects the relative
    lowest point of each swing rather than true touchdown.
    """
    contacts: list[ContactEvent] = []

    for foot, joint in ((Foot.LEFT, J.LEFT_ANKLE), (Foot.RIGHT, J.RIGHT_ANKLE)):
        series = _foot_series(track, fps, joint, filter_cutoff_hz)
        if series is None:
            continue
        frames, positions, _, reference = series

        y = positions[:, 1]
        speed = np.abs(np.gradient(y / reference, frames / fps))
        slow = speed < np.percentile(speed, 40)

        for i in range(1, len(y) - 1):
            if y[i] >= y[i - 1] and y[i] > y[i + 1] and slow[i]:
                contacts.append(
                    ContactEvent(
                        frame_index=int(frames[i]),
                        time_s=float(frames[i] / fps),
                        foot=foot,
                    )
                )

    contacts.sort(key=lambda c: c.time_s)
    return contacts


#: Human running cadence tops out around 220 steps/min even for sprinters.
#: Anything outside this band means the detector fired on something that is
#: not gait - a stationary player's pose jitter, most often - and the number
#: should not be shown at all rather than shown as a record-breaking figure.
PLAUSIBLE_CADENCE_SPM = (40.0, 260.0)


def cadence_spm(contacts: list[ContactEvent]) -> float:
    """Steps per minute from a list of contacts.

    Returns NaN with fewer than two contacts, or when the result falls
    outside :data:`PLAUSIBLE_CADENCE_SPM`, which indicates the contacts were
    noise rather than steps.
    """
    if len(contacts) < 2:
        return float("nan")
    span = contacts[-1].time_s - contacts[0].time_s
    if span <= 0:
        return float("nan")

    cadence = 60.0 * (len(contacts) - 1) / span
    low, high = PLAUSIBLE_CADENCE_SPM
    return cadence if low <= cadence <= high else float("nan")


def summarise_events(
    kicks: list[KickEvent], contacts: list[ContactEvent]
) -> list[str]:
    """Plain-text lines for the CLI report."""
    lines = [f"Strikes detected: {len(kicks)}"]
    lines.extend(f"  {i}. {k.summary()}" for i, k in enumerate(kicks, start=1))

    if contacts:
        cadence = cadence_spm(contacts)
        lines.append(f"Foot contacts: {len(contacts)}")
        if np.isfinite(cadence):
            lines.append(f"  Cadence: {cadence:.0f} steps/min")
        else:
            lines.append(
                "  Cadence not reported: the contacts do not look like gait, "
                "which usually means the player was not running."
            )
    if kicks:
        lines.append(
            "  Note: strikes are inferred from foot swing, not from the ball. "
            "Check them on the overlay before using them."
        )
    return lines
