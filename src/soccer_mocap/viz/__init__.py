"""Drawing skeletons, goals and events onto frames."""

from __future__ import annotations

from .overlay import (
    draw_frame,
    draw_goal,
    draw_label,
    draw_pose,
    render_overlay_frames,
    track_colour,
)

__all__ = [
    "draw_frame",
    "draw_goal",
    "draw_label",
    "draw_pose",
    "render_overlay_frames",
    "track_colour",
]
