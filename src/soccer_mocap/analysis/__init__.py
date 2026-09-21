"""Kinematics, events, and general match metrics."""

from __future__ import annotations

from .events import (
    ContactEvent,
    Foot,
    KickEvent,
    cadence_spm,
    detect_foot_contacts,
    detect_kicks,
    summarise_events,
)
from .kinematics import (
    body_height_px,
    center_of_mass,
    com_series,
    filter_trajectory,
    joint_angle,
    joint_angle_series,
    normalize_by_height,
    segment_angle_from_vertical,
    speed,
    velocity,
)

__all__ = [
    "ContactEvent",
    "Foot",
    "KickEvent",
    "body_height_px",
    "cadence_spm",
    "detect_foot_contacts",
    "detect_kicks",
    "summarise_events",
    "center_of_mass",
    "com_series",
    "filter_trajectory",
    "joint_angle",
    "joint_angle_series",
    "normalize_by_height",
    "reference_height_px",
    "segment_angle_from_vertical",
    "speed",
    "velocity",
]
