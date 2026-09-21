"""Goalkeeper-specific tracking and analysis.

The goalkeeper is a different measurement problem from an outfield player.
They occupy a known, fixed region; the goal beside them is the one object in
frame whose real dimensions are known exactly; and the movements that matter
are short, explosive, and rare rather than continuous. That is enough of a
difference to justify its own package rather than a flag on the general
analysis path.

Typical use:

    from soccer_mocap.goalkeeper import GoalGeometry, build_report, identify_goalkeeper

    goal = GoalGeometry(left_post=(410, 520), right_post=(980, 528))
    track_id, candidates = identify_goalkeeper(tracks, goal=goal)
    report = build_report(tracks[track_id], fps=60.0, goal=goal, candidates=candidates)
    print(report)
"""

from __future__ import annotations

from .dive import (
    DiveDirection,
    DiveEvent,
    DiveHeight,
    DivePhases,
    detect_dives,
    dive_summary_table,
)
from .goal import (
    GOAL_HEIGHT_M,
    GOAL_WIDTH_M,
    GoalGeometry,
    detect_goal_posts,
)
from .identify import (
    GoalkeeperCandidate,
    identify_goalkeeper,
    score_by_geometry,
    score_by_kit_colour,
)
from .metrics import (
    GoalkeeperReport,
    PositioningMetrics,
    analyse_positioning,
    build_report,
)
from .stance import StanceMetrics, analyse_set_position, analyse_stance, find_set_position

__all__ = [
    "GOAL_HEIGHT_M",
    "GOAL_WIDTH_M",
    "DiveDirection",
    "DiveEvent",
    "DiveHeight",
    "DivePhases",
    "GoalGeometry",
    "GoalkeeperCandidate",
    "GoalkeeperReport",
    "PositioningMetrics",
    "StanceMetrics",
    "analyse_positioning",
    "analyse_set_position",
    "analyse_stance",
    "build_report",
    "detect_dives",
    "detect_goal_posts",
    "dive_summary_table",
    "find_set_position",
    "identify_goalkeeper",
    "score_by_geometry",
    "score_by_kit_colour",
]
