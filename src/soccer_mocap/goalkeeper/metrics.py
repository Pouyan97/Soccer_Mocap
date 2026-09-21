"""Aggregate goalkeeper report.

Pulls positioning, stance, and dives into one object that the CLI, the demo
app, and the JSON export all render from, so the three never drift apart.

Positioning is the section to read with the most care. Distances off the
goal line come from a scale measured *at* the goal line, so they get
progressively more optimistic as the keeper advances. Within a few metres of
the line they are good to roughly 10%; for anything further out, calibrate a
full pitch homography and use that instead. Every affected number carries a
warning in :attr:`GoalkeeperReport.warnings` rather than being silently
reported as fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..analysis.kinematics import (
    body_height_px,
    center_of_mass,
    filter_trajectory,
    reference_height_px,
)
from ..types import Track
from .dive import DiveEvent, detect_dives, dive_summary_table
from .goal import GOAL_WIDTH_M, GoalGeometry
from .identify import GoalkeeperCandidate
from .stance import StanceMetrics, analyse_set_position


@dataclass
class PositioningMetrics:
    """Where the keeper stood over the clip."""

    #: Median distance out from the goal line, metres. None without a goal.
    median_depth_m: float | None = None
    #: Range of depth over the clip, metres.
    depth_range_m: float | None = None
    #: Median position across the mouth: 0 = left post, 1 = right post.
    median_coverage: float | None = None
    #: Fraction of frames spent within the goal area.
    time_in_goal_area: float | None = None
    #: Ground covered, metres.
    distance_covered_m: float | None = None
    #: Ground covered in body heights, always available.
    distance_covered_bh: float = float("nan")
    #: Peak COM speed in body-heights per second.
    peak_speed_bh_s: float = float("nan")
    frames_analysed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "median_depth_m": self.median_depth_m,
            "depth_range_m": self.depth_range_m,
            "median_coverage": self.median_coverage,
            "time_in_goal_area": self.time_in_goal_area,
            "distance_covered_m": self.distance_covered_m,
            "distance_covered_bh": self.distance_covered_bh,
            "peak_speed_bh_s": self.peak_speed_bh_s,
            "frames_analysed": self.frames_analysed,
        }


@dataclass
class GoalkeeperReport:
    """Everything the goalkeeper analysis produced for one clip."""

    track_id: int
    fps: float
    duration_s: float
    positioning: PositioningMetrics = field(default_factory=PositioningMetrics)
    set_position: StanceMetrics | None = None
    dives: list[DiveEvent] = field(default_factory=list)
    #: Ranked identification candidates, so a wrong pick is auditable.
    candidates: list[GoalkeeperCandidate] = field(default_factory=list)
    #: Caveats that apply to the numbers above.
    warnings: list[str] = field(default_factory=list)
    #: True when a goal was marked, which gates every metric output.
    calibrated: bool = False

    @property
    def dive_count(self) -> int:
        return len(self.dives)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "calibrated": self.calibrated,
            "positioning": self.positioning.to_dict(),
            "set_position": self.set_position.to_dict() if self.set_position else None,
            "dives": [d.to_dict() for d in self.dives],
            "dive_count": self.dive_count,
            "candidates": [
                {"track_id": c.track_id, "score": c.score, "reasons": c.reasons}
                for c in self.candidates
            ],
            "warnings": list(self.warnings),
        }

    def summary_lines(self) -> list[str]:
        """Plain-text report, shared by the CLI and the demo app."""
        lines = [
            f"Goalkeeper: track {self.track_id}",
            f"Clip: {self.duration_s:.1f} s at {self.fps:.1f} fps",
            "",
        ]

        pos = self.positioning
        lines.append("Positioning")
        if pos.median_depth_m is not None:
            lines.append(f"  Median distance off line : {pos.median_depth_m:.2f} m")
            lines.append(f"  Depth range              : {pos.depth_range_m:.2f} m")
        if pos.median_coverage is not None:
            side = _coverage_description(pos.median_coverage)
            lines.append(f"  Median mouth position    : {pos.median_coverage:.2f} ({side})")
        if pos.time_in_goal_area is not None:
            lines.append(f"  Time in goal area        : {100 * pos.time_in_goal_area:.0f}%")
        if pos.distance_covered_m is not None:
            lines.append(f"  Distance covered         : {pos.distance_covered_m:.1f} m")
        else:
            lines.append(
                f"  Distance covered         : {pos.distance_covered_bh:.1f} body heights"
            )
        lines.append(f"  Peak speed               : {pos.peak_speed_bh_s:.2f} body heights/s")
        lines.append("")

        if self.set_position is not None:
            stance = self.set_position
            lines.append(f"Set position (frame {stance.frame_index})")
            lines.append(
                f"  Stance width             : {stance.stance_width_ratio:.2f} x shoulders"
            )
            lines.append(f"  Knee angle               : {stance.mean_knee_angle:.0f} deg")
            lines.append(f"  Trunk flexion            : {stance.trunk_flexion_deg:.0f} deg")
            lines.append(
                f"  Hand height              : {stance.hand_height_ratio:+.2f}"
                " body heights above hips"
            )
            for flag in stance.flags:
                lines.append(f"  ! {flag}")
            lines.append("")

        lines.append(f"Dives detected: {self.dive_count}")
        for i, dive in enumerate(self.dives, start=1):
            lines.append(f"  {i}. {dive.summary()}")
            for note in dive.notes:
                lines.append(f"     ! {note}")
        lines.append("")

        if self.warnings:
            lines.append("Caveats")
            lines.extend(f"  - {w}" for w in self.warnings)

        return lines

    def __str__(self) -> str:
        return "\n".join(self.summary_lines())

    def dive_table(self) -> list[dict[str, Any]]:
        return dive_summary_table(self.dives)


def _coverage_description(coverage: float) -> str:
    if coverage < 0.35:
        return "toward the left post"
    if coverage > 0.65:
        return "toward the right post"
    return "central"


def analyse_positioning(
    track: Track,
    fps: float,
    goal: GoalGeometry | None,
    filter_cutoff_hz: float | None = 6.0,
) -> PositioningMetrics:
    """Measure where the keeper stood and how much ground they covered."""
    metrics = PositioningMetrics()

    positions, heights = [], []
    for frame_index in track.frame_indices:
        pose = track.poses[frame_index]
        com = center_of_mass(pose.xy, pose.confidence)
        height = body_height_px(pose.xy, pose.confidence)
        if np.isfinite(com).all() and np.isfinite(height) and height > 0:
            positions.append(com)
            heights.append(height)

    metrics.frames_analysed = len(positions)
    if len(positions) < 2:
        return metrics

    path = filter_trajectory(np.array(positions), fps, filter_cutoff_hz)
    # One stature for the whole track, not one per frame: see
    # kinematics.reference_height_px for why that distinction matters.
    reference = reference_height_px(heights)
    if not np.isfinite(reference) or reference <= 0:
        return metrics

    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    metrics.distance_covered_bh = float(np.sum(steps / reference))

    velocity = np.gradient(path / reference, 1.0 / fps, axis=0)
    metrics.peak_speed_bh_s = float(np.max(np.linalg.norm(velocity, axis=1)))

    if goal is None:
        return metrics

    depths = np.array([goal.distance_off_line_m(p) for p in path])
    coverage = np.array([goal.coverage_fraction(p) for p in path])

    metrics.median_depth_m = float(np.median(depths))
    metrics.depth_range_m = float(np.percentile(depths, 95) - np.percentile(depths, 5))
    metrics.median_coverage = float(np.median(coverage))
    metrics.time_in_goal_area = float(
        np.mean([goal.is_in_goal_area(p) for p in path])
    )
    metrics.distance_covered_m = float(np.sum(steps)) / goal.px_per_m
    return metrics


def build_report(
    track: Track,
    fps: float,
    goal: GoalGeometry | None = None,
    candidates: list[GoalkeeperCandidate] | None = None,
    shot_times_s: list[float] | None = None,
    speed_threshold: float = 1.2,
    min_dive_frames: int = 4,
    lean_threshold_deg: float = 25.0,
    filter_cutoff_hz: float | None = 6.0,
) -> GoalkeeperReport:
    """Run the whole goalkeeper analysis over one track."""
    duration = len(track) / fps if fps > 0 else 0.0
    report = GoalkeeperReport(
        track_id=track.track_id,
        fps=fps,
        duration_s=duration,
        candidates=candidates or [],
        calibrated=goal is not None,
    )

    report.positioning = analyse_positioning(track, fps, goal, filter_cutoff_hz)
    report.set_position = analyse_set_position(track, fps)
    report.dives = detect_dives(
        track,
        fps,
        speed_threshold=speed_threshold,
        min_dive_frames=min_dive_frames,
        lean_threshold_deg=lean_threshold_deg,
        # Dive detection wants a higher cutoff than positioning: the takeoff
        # transient is real signal, and 6 Hz rounds the peak off.
        filter_cutoff_hz=(filter_cutoff_hz or 6.0) + 2.0,
        goal=goal,
        shot_times_s=shot_times_s,
    )

    report.warnings.extend(_collect_warnings(report, track, fps, goal, shot_times_s))
    return report


def _collect_warnings(
    report: GoalkeeperReport,
    track: Track,
    fps: float,
    goal: GoalGeometry | None,
    shot_times_s: list[float] | None,
) -> list[str]:
    """Assemble the caveats that apply to this particular run."""
    warnings: list[str] = []

    if goal is None:
        warnings.append(
            "No goal was marked, so positioning is in body heights rather than metres. "
            "Mark the two posts to get metric distances."
        )
    else:
        consistency = goal.scale_consistency()
        if consistency is not None and not (0.8 <= consistency <= 1.25):
            warnings.append(
                f"Goal post and crossbar scales disagree by {abs(1 - consistency) * 100:.0f}%. "
                "Check the marked points, or the lens distortion."
            )
        warnings.append(
            "Distances off the goal line use the scale measured at the line, so they "
            "become optimistic as the keeper advances. Good to about 10% on the line."
        )

    if fps < 50 and report.dives:
        warnings.append(
            f"Filmed at {fps:.0f} fps. A dive lasts only a handful of frames at this rate, "
            "so phase timings are coarse. 60 fps or higher is strongly preferred."
        )

    if shot_times_s is None and report.dives:
        warnings.append(
            "No shot times were supplied, so reaction times are not reported. "
            "Ball tracking is not implemented; supply strike times to get them."
        )

    coverage = len(track) / max(1, (track.last_frame - track.first_frame + 1))
    if coverage < 0.8:
        warnings.append(
            f"The keeper was detected in only {100 * coverage:.0f}% of frames across their "
            "span. Gaps were interpolated; check for occlusion."
        )

    if report.positioning.frames_analysed < 15:
        warnings.append(
            "Very few usable frames. Treat every number here as indicative only."
        )

    stance = report.set_position
    if stance is not None and not stance.is_valid:
        warnings.append("The set position could not be measured reliably.")

    if goal is not None and report.positioning.median_coverage is not None:
        offset = abs(report.positioning.median_coverage - 0.5) * GOAL_WIDTH_M
        if offset > 1.5:
            warnings.append(
                f"The keeper's median position is {offset:.1f} m off centre. If they were not "
                "defending an angle, check that the posts were marked correctly."
            )

    return warnings
