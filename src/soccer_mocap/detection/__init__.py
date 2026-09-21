"""Person detection, for top-down pose backends."""

from __future__ import annotations

from .person import PersonDetector, YOLOPersonDetector, crop_to_box, get_detector

__all__ = ["PersonDetector", "YOLOPersonDetector", "crop_to_box", "get_detector"]
