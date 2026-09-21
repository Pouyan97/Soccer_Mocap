"""Person detection.

Only needed for top-down pose backends, which expect a cropped person box
rather than a whole frame. MediaPipe does its own detection internally, so
this stage is skipped entirely on the default path; Sapiens and RTMPose
(in its two-stage form) both need it.

Ultralytics is in the environment already, so the YOLO detector here is
implemented rather than stubbed. Weights download themselves on first use.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..exceptions import BackendUnavailableError
from ..types import BBox

#: COCO class index for "person". Every model in this family shares it.
PERSON_CLASS_ID = 0


class PersonDetector(ABC):
    """Find people in a frame."""

    @classmethod
    @abstractmethod
    def available(cls) -> tuple[bool, str]:
        """``(usable, reason)`` without importing the heavy dependency."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[BBox]:
        """Return person boxes for one RGB frame."""

    def setup(self) -> None:  # noqa: B027 - optional hook, not every detector needs it
        """Load the model. Called once before the first detect."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release the model."""


class YOLOPersonDetector(PersonDetector):
    """Person boxes from an Ultralytics YOLO model.

    Args:
        model: weights name or path. The nano model is the default because
            person detection is the easy case for these models and the extra
            capacity of the larger ones is not worth the latency here.
        confidence: minimum detection score.
        device: ``"cuda"``, ``"cpu"``, or ``None`` to let ultralytics choose.
        min_height_fraction: discard boxes shorter than this fraction of the
            frame. Filters the crowd behind the goal, which otherwise
            produces dozens of unusable detections per frame.
    """

    def __init__(
        self,
        model: str = "yolo11n.pt",
        confidence: float = 0.35,
        device: str | None = None,
        min_height_fraction: float = 0.05,
        **options: Any,
    ) -> None:
        self.model_name = model
        self.confidence = confidence
        self.device = device
        self.min_height_fraction = min_height_fraction
        self.options = options
        self._model = None

    @classmethod
    def available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("ultralytics") is None:
            return False, "the 'ultralytics' package is not installed"
        return True, ""

    def setup(self) -> None:
        usable, reason = self.available()
        if not usable:
            raise BackendUnavailableError(
                "yolo", reason, "mamba run -n soccer-mocap pip install ultralytics"
            )
        from ultralytics import YOLO

        self._model = YOLO(self.model_name)
        if self.device:
            self._model.to(self.device)

    def close(self) -> None:
        self._model = None

    def detect(self, image: np.ndarray) -> list[BBox]:
        if self._model is None:
            self.setup()

        results = self._model.predict(
            image,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence,
            verbose=False,
            **self.options,
        )
        if not results:
            return []

        frame_height = image.shape[0]
        min_height = self.min_height_fraction * frame_height

        boxes: list[BBox] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            if (y2 - y1) < min_height:
                continue
            boxes.append(BBox(x1, y1, x2, y2, score=float(box.conf[0])))

        boxes.sort(key=lambda b: b.area, reverse=True)
        return boxes


def get_detector(name: str = "yolo", **options: Any) -> PersonDetector:
    """Construct a detector by name."""
    detectors: dict[str, type[PersonDetector]] = {"yolo": YOLOPersonDetector}
    if name not in detectors:
        raise BackendUnavailableError(
            name,
            "no detector is registered under that name",
            f"Choose one of: {', '.join(sorted(detectors))}",
        )
    return detectors[name](**options)


def crop_to_box(
    image: np.ndarray, bbox: BBox, padding: float = 0.15
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a padded person box, for feeding a top-down pose model.

    Returns the crop and its ``(x, y)`` origin, so keypoints estimated in
    crop coordinates can be shifted back into frame coordinates.
    """
    height, width = image.shape[:2]
    pad_x, pad_y = padding * bbox.width, padding * bbox.height

    x1 = max(0, int(bbox.x1 - pad_x))
    y1 = max(0, int(bbox.y1 - pad_y))
    x2 = min(width, int(bbox.x2 + pad_x))
    y2 = min(height, int(bbox.y2 + pad_y))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=image.dtype), (0, 0)
    return image[y1:y2, x1:x2], (x1, y1)
