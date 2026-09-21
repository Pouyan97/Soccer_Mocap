"""Phone capture profiles and the processing settings that suit them."""

from __future__ import annotations

from .presets import (
    DEVICE_PRESETS,
    PROFILES,
    CaptureAdvice,
    DevicePreset,
    ProcessingProfile,
    capture_advice,
    get_preset,
    get_profile,
    infer_preset,
    resolve_profile,
)

__all__ = [
    "DEVICE_PRESETS",
    "PROFILES",
    "CaptureAdvice",
    "DevicePreset",
    "ProcessingProfile",
    "capture_advice",
    "get_preset",
    "get_profile",
    "infer_preset",
    "resolve_profile",
]
