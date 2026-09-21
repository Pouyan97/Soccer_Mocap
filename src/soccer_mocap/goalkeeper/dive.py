"""Dive detection, phase segmentation, and classification.

A dive is detected from the goalkeeper's own motion, with no ball tracking
required. The signal is a burst of lateral centre-of-mass speed combined with
the trunk going off vertical - the combination is what separates a dive from
a sidestep (fast but upright) and from a crouch (leaning but slow).

Everything is normalised by the keeper's pixel height, giving body-heights
per second. That makes one threshold work across camera distances and
resolutions, which is the whole problem with phone footage. Multiply by the
keeper's real stature to read the numbers in m/s.

The phases follow the usual coaching breakdown:

``set``
    Still, loaded, waiting.
``preparatory``
    The counter-movement: a small drop and often a step toward the ball.
``takeoff``
    Ground contact ends. Peak COM acceleration lives here.
``flight``
    Airborne and ballistic. Nothing the keeper does now changes their COM
    path, which is why takeoff quality is what coaches actually work on.
``landing``
    Ground contact resumes.

Phase boundaries come from COM speed and height, so they are reliable to
about one frame at 30 fps and better at 60. At 30 fps a fast dive is only
~8 frames long, which is the main argument for filming keepers at 60 or
higher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..analysis.kinematics import (
    body_height_px,
    center_of_mass,
    filter_trajectory,
    reference_height_px,
    segment_angle_from_vertical,
)
from ..skeleton import J
from ..types import Track
from .goal import GoalGeometry


class DiveDirection(str, Enum):
    """Which way the keeper went, from the camera's point of view."""

    LEFT = "left"
    RIGHT = "right"
    CENTRE = "centre"


class DiveHeight(str, Enum):
    """How high the save attempt was, relative to the keeper's own body."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"


@dataclass
class DivePhases:
    """Frame indices marking the phase boundaries. ``None`` when not resolved."""

    set_frame: int | None = None
    preparatory_start: int | None = None
    takeoff: int | None = None
    flight_start: int | None = None
    peak: int | None = None
    landing: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "set": self.set_frame,
            "preparatory_start": self.preparatory_start,
            "takeoff": self.takeoff,
            "flight_start": self.flight_start,
            "peak": self.peak,
            "landing": self.landing,
        }


@dataclass
class DiveEvent:
    """One detected dive."""

    start_frame: int
    end_frame: int
    fps: float
    direction: DiveDirection
    height: DiveHeight
    phases: DivePhases = field(default_factory=DivePhases)

    #: Peak COM speed during the dive, in body-heights per second.
    peak_speed_bh_s: float = float("nan")
    #: Lateral COM displacement, in body heights. Signed: + is screen right.
    lateral_displacement_bh: float = float("nan")
    #: Maximum trunk deviation from vertical, in degrees.
    max_trunk_lean_deg: float = float("nan")
    #: Highest reach achieved by either wrist, in body heights above the hips.
    peak_reach_bh: float = float("nan")
    #: Seconds from the first movement to the peak of the dive.
    time_to_peak_s: float = float("nan")
    #: Seconds from a supplied shot event to the first movement. NaN if no
    #: shot time was given - this is reaction time and needs an external cue.
    reaction_time_s: float = float("nan")
    #: Lateral distance in metres, when goal geometry made a scale available.
    lateral_displacement_m: float | None = None
    #: Mean joint confidence over the dive; low values mean don't trust it.
    confidence: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps if self.fps > 0 else float("nan")

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_s": self.duration_s,
            "direction": self.direction.value,
            "height": self.height.value,
            "peak_speed_bh_s": self.peak_speed_bh_s,
            "lateral_displacement_bh": self.lateral_displacement_bh,
            "lateral_displacement_m": self.lateral_displacement_m,
            "max_trunk_lean_deg": self.max_trunk_lean_deg,
            "peak_reach_bh": self.peak_reach_bh,
            "time_to_peak_s": self.time_to_peak_s,
            "reaction_time_s": self.reaction_time_s,
            "confidence": self.confidence,
            "phases": self.phases.to_dict(),
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """One line for a report or the demo app."""
        speed = f"{self.peak_speed_bh_s:.1f} bh/s" if np.isfinite(self.peak_speed_bh_s) else "?"
        distance = (
            f"{self.lateral_displacement_m:.2f} m"
            if self.lateral_displacement_m is not None
            else f"{abs(self.lateral_displacement_bh):.2f} bh"
        )
        return (
            f"{self.direction.value} {self.height.value} dive, "
            f"{self.duration_s:.2f} s, peak {speed}, travelled {distance}"
        )


@dataclass
class _DiveSignals:
    """Per-frame arrays used by both detection and classification."""

    frames: np.ndarray
    com: np.ndarray            # (T, 2) filtered, pixels
    speed_bh_s: np.ndarray     # (T,)
    lateral_bh_s: np.ndarray   # (T,) signed, screen-x
    trunk_lean: np.ndarray     # (T,) degrees
    reach_bh: np.ndarray       # (T,) wrist height above hips, body heights
    com_height_bh: np.ndarray  # (T,) COM above ankles, body heights
    ankle_bh: np.ndarray       # (T,) mean ankle height in image y, body heights
    height_px: np.ndarray      # (T,) per-frame stature estimate, pixels
    reference_height_px: float # single robust stature for the whole track
    confidence: np.ndarray     # (T,)


def _build_signals(track: Track, fps: float, filter_cutoff_hz: float | None) -> _DiveSignals | None:
    """Extract the time series a dive is detected from."""
    frame_indices = track.frame_indices
    if len(frame_indices) < 5:
        return None

    com_list, lean_list, reach_list, com_h_list, ankle_list, height_list, conf_list, kept = (
        [], [], [], [], [], [], [], []
    )

    for frame_index in frame_indices:
        pose = track.poses[frame_index]
        com = center_of_mass(pose.xy, pose.confidence)
        height = body_height_px(pose.xy, pose.confidence)
        if not (np.isfinite(com).all() and np.isfinite(height) and height > 0):
            continue

        left_shoulder = pose.joint(J.LEFT_SHOULDER, 0.2)
        right_shoulder = pose.joint(J.RIGHT_SHOULDER, 0.2)
        left_hip = pose.joint(J.LEFT_HIP, 0.2)
        right_hip = pose.joint(J.RIGHT_HIP, 0.2)

        lean = float("nan")
        mid_hip = None
        if all(p is not None for p in (left_shoulder, right_shoulder, left_hip, right_hip)):
            mid_shoulder = 0.5 * (left_shoulder + right_shoulder)
            mid_hip = 0.5 * (left_hip + right_hip)
            lean = segment_angle_from_vertical(mid_hip, mid_shoulder)

        reach = float("nan")
        wrists = [
            w
            for w in (pose.joint(J.LEFT_WRIST, 0.2), pose.joint(J.RIGHT_WRIST, 0.2))
            if w is not None
        ]
        if wrists and mid_hip is not None:
            highest = min(float(w[1]) for w in wrists)  # smallest y = highest
            reach = float(mid_hip[1]) - highest  # pixels; normalised below

        ankles = [
            a
            for a in (pose.joint(J.LEFT_ANKLE, 0.2), pose.joint(J.RIGHT_ANKLE, 0.2))
            if a is not None
        ]
        mean_ankle_y = (
            float(np.mean([a[1] for a in ankles])) if ankles else float("nan")
        )

        kept.append(frame_index)
        com_list.append(com)
        lean_list.append(lean)
        # Stored in pixels here and normalised below, once the track's single
        # reference stature is known.
        reach_list.append(reach)
        com_h_list.append(mean_ankle_y - float(com[1]))
        ankle_list.append(mean_ankle_y)
        height_list.append(height)
        conf_list.append(pose.mean_confidence())

    if len(kept) < 5:
        return None

    frames = np.array(kept)
    com = filter_trajectory(np.array(com_list), fps, filter_cutoff_hz)
    height_px = np.array(height_list)

    # One stature for the whole track. Normalising each frame by its own
    # estimate would put the pose model's jitter in the denominator, and a
    # single frame with badly-estimated legs then reads as a velocity spike
    # of tens of body-heights per second.
    reference = reference_height_px(height_px)
    if not np.isfinite(reference) or reference <= 0:
        return None

    # Differentiate in body-heights so the threshold is scale-free.
    com_bh = com / reference
    # np.gradient handles the non-uniform spacing that dropped frames create.
    time_s = frames / fps
    velocity = np.gradient(com_bh, time_s, axis=0)

    return _DiveSignals(
        frames=frames,
        com=com,
        speed_bh_s=np.linalg.norm(velocity, axis=1),
        lateral_bh_s=velocity[:, 0],
        trunk_lean=np.array(lean_list),
        reach_bh=np.array(reach_list) / reference,
        com_height_bh=np.array(com_h_list) / reference,
        ankle_bh=np.array(ankle_list) / reference,
        height_px=height_px,
        reference_height_px=reference,
        confidence=np.array(conf_list),
    )


def detect_dives(
    track: Track,
    fps: float,
    speed_threshold: float = 1.2,
    min_dive_frames: int = 4,
    lean_threshold_deg: float = 25.0,
    filter_cutoff_hz: float | None = 8.0,
    goal: GoalGeometry | None = None,
    shot_times_s: list[float] | None = None,
) -> list[DiveEvent]:
    """Find every dive in a goalkeeper's track.

    Args:
        track: the goalkeeper's track.
        fps: frame rate. Getting this wrong scales every speed reported.
        speed_threshold: COM speed in body-heights/s that starts a dive. The
            1.2 default separates a committed dive from a quick sidestep for
            an adult keeper; lower it for youth footage, where body heights
            per second run higher for the same absolute speed.
        min_dive_frames: reject bursts shorter than this. At 30 fps, 4 frames
            is 130 ms, which is about the shortest a real dive can be.
        lean_threshold_deg: trunk deviation from vertical required somewhere
            in the burst. This is the test that rejects a fast sidestep.
        filter_cutoff_hz: low-pass applied before differentiating. 8 Hz keeps
            the takeoff transient that 6 Hz would round off.
        goal: supplies the pixel-to-metre scale for the metric outputs.
        shot_times_s: times of ball strikes, if known from another source.
            Used only to compute reaction time.

    Returns:
        Dives in chronological order. Empty when the keeper never dives,
        which is the common case and not an error.
    """
    signals = _build_signals(track, fps, filter_cutoff_hz)
    if signals is None:
        return []

    above = signals.speed_bh_s >= speed_threshold
    events: list[DiveEvent] = []

    for start, end in _contiguous_runs(above):
        if (end - start + 1) < min_dive_frames:
            continue

        window = slice(start, end + 1)
        leans = signals.trunk_lean[window]
        finite_leans = leans[np.isfinite(leans)]
        max_lean = float(np.max(np.abs(finite_leans))) if finite_leans.size else float("nan")

        # A dive leans. A sidestep does not. Skip the test entirely when the
        # trunk was never visible rather than silently dropping a real dive.
        if np.isfinite(max_lean) and max_lean < lean_threshold_deg:
            continue

        events.append(
            _build_event(signals, start, end, fps, max_lean, goal, shot_times_s)
        )

    return events


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` index pairs for each run of True."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _build_event(
    signals: _DiveSignals,
    start: int,
    end: int,
    fps: float,
    max_lean: float,
    goal: GoalGeometry | None,
    shot_times_s: list[float] | None,
) -> DiveEvent:
    """Measure and classify one detected burst."""
    window = slice(start, end + 1)
    frames = signals.frames
    start_frame, end_frame = int(frames[start]), int(frames[end])

    peak_local = start + int(np.argmax(signals.speed_bh_s[window]))
    peak_speed = float(signals.speed_bh_s[peak_local])

    lateral_bh = float(
        (signals.com[end, 0] - signals.com[start, 0]) / signals.reference_height_px
    )

    reaches = signals.reach_bh[window]
    finite_reaches = reaches[np.isfinite(reaches)]
    peak_reach = float(np.max(finite_reaches)) if finite_reaches.size else float("nan")

    direction = _classify_direction(lateral_bh)
    height_class = _classify_height(signals, window, peak_reach)
    phases = _segment_phases(signals, start, end, peak_local)

    event = DiveEvent(
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
        direction=direction,
        height=height_class,
        phases=phases,
        peak_speed_bh_s=peak_speed,
        lateral_displacement_bh=lateral_bh,
        max_trunk_lean_deg=max_lean,
        peak_reach_bh=peak_reach,
        time_to_peak_s=float(frames[peak_local] - frames[start]) / fps,
        confidence=float(np.nanmean(signals.confidence[window])),
    )

    if goal is not None:
        event.lateral_displacement_m = (
            lateral_bh * signals.reference_height_px / goal.px_per_m
        )

    if shot_times_s:
        movement_time = start_frame / fps
        # The most recent shot at or before the dive started.
        prior = [t for t in shot_times_s if t <= movement_time]
        if prior:
            event.reaction_time_s = movement_time - max(prior)
            if event.reaction_time_s < 0.10:
                event.notes.append(
                    "reaction under 100 ms - the keeper likely moved before the strike "
                    "(anticipation), or the shot time is misaligned"
                )

    if not np.isfinite(max_lean):
        event.notes.append("trunk not visible; lean test skipped for this dive")
    if event.confidence < 0.5:
        event.notes.append("low pose confidence; treat these numbers as indicative")

    return event


def _classify_direction(lateral_bh: float, deadband: float = 0.15) -> DiveDirection:
    """Left, right, or neither, from signed lateral displacement."""
    if not np.isfinite(lateral_bh) or abs(lateral_bh) < deadband:
        return DiveDirection.CENTRE
    return DiveDirection.RIGHT if lateral_bh > 0 else DiveDirection.LEFT


def _classify_height(signals: _DiveSignals, window: slice, peak_reach: float) -> DiveHeight:
    """Low, mid, or high, from how far the COM dropped and how high the hands went.

    Uses the COM drop as the primary cue because the hands are the joints
    most often lost to motion blur at the moment of the save.
    """
    com_heights = signals.com_height_bh[window]
    finite = com_heights[np.isfinite(com_heights)]

    if finite.size:
        minimum = float(np.min(finite))
        if minimum < 0.20:
            return DiveHeight.LOW
        if minimum > 0.40 and (not np.isfinite(peak_reach) or peak_reach > 0.35):
            return DiveHeight.HIGH
        return DiveHeight.MID

    if np.isfinite(peak_reach):
        if peak_reach > 0.40:
            return DiveHeight.HIGH
        if peak_reach < 0.10:
            return DiveHeight.LOW
    return DiveHeight.MID


def _segment_phases(
    signals: _DiveSignals, start: int, end: int, peak_local: int
) -> DivePhases:
    """Locate the phase boundaries around a detected burst."""
    phases = DivePhases()
    frames = signals.frames
    speed = signals.speed_bh_s

    # Set: the stillest frame in the second before movement began.
    lookback = max(0, start - int(round(len(frames) * 0.1)) - 1)
    if start > lookback:
        phases.set_frame = int(frames[lookback + int(np.argmin(speed[lookback:start]))])

    # Preparatory: walk back from the burst to where speed first lifted off
    # its baseline. This catches the counter-movement, which sits below the
    # detection threshold by definition.
    baseline = float(np.median(speed[:start])) if start > 0 else 0.0
    prep = start
    while prep > 0 and speed[prep - 1] > baseline * 1.5:
        prep -= 1
    phases.preparatory_start = int(frames[prep])

    phases.takeoff = int(frames[start])
    phases.peak = int(frames[peak_local])

    # Flight begins when the feet leave the ground. Detected as the ankles
    # rising clear of the level they held before the dive - an absolute COM
    # height would not work, because a diving body's COM falls even while it
    # is airborne.
    phases.flight_start = _find_takeoff_frame(signals, start, end)

    phases.landing = int(frames[end])
    return phases


#: Ankles must rise this far above their pre-dive level, in body heights,
#: before the keeper is called airborne. Large enough to ignore pose jitter.
_FOOT_LIFT_THRESHOLD_BH = 0.04


def _find_takeoff_frame(signals: _DiveSignals, start: int, end: int) -> int | None:
    """First frame in the burst where both feet have left the ground."""
    ankle = signals.ankle_bh
    baseline_window = ankle[:start]
    finite_baseline = baseline_window[np.isfinite(baseline_window)]
    if finite_baseline.size == 0:
        return None

    # Image y grows downward, so "risen" means a smaller value.
    ground_level = float(np.median(finite_baseline))
    threshold = ground_level - _FOOT_LIFT_THRESHOLD_BH

    for i in range(start, end + 1):
        if np.isfinite(ankle[i]) and ankle[i] < threshold:
            return int(signals.frames[i])
    return None


def dive_summary_table(dives: list[DiveEvent]) -> list[dict[str, object]]:
    """Flatten dives into rows for a table or a DataFrame."""
    return [
        {
            "#": i + 1,
            "start_s": round(d.start_frame / d.fps, 3) if d.fps else None,
            "duration_s": round(d.duration_s, 3),
            "direction": d.direction.value,
            "height": d.height.value,
            "peak_speed_bh_s": round(d.peak_speed_bh_s, 2),
            "lateral_m": (
                round(d.lateral_displacement_m, 2)
                if d.lateral_displacement_m is not None
                else None
            ),
            "lateral_bh": round(d.lateral_displacement_bh, 2),
            "max_lean_deg": round(d.max_trunk_lean_deg, 1),
            "confidence": round(d.confidence, 2),
        }
        for i, d in enumerate(dives)
    ]
