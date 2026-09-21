"""Pose estimation: the swappable markerless backends."""

from __future__ import annotations

from .base import BackendInfo, PoseBackend, StubBackend
from .registry import (
    availability,
    default_backend_key,
    describe,
    get_backend,
    register,
    registered,
)

__all__ = [
    "BackendInfo",
    "PoseBackend",
    "StubBackend",
    "availability",
    "default_backend_key",
    "describe",
    "get_backend",
    "register",
    "registered",
]
