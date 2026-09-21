"""The interface every pose backend implements.

Adding a new markerless system means writing one subclass of
:class:`PoseBackend` and registering it. Nothing downstream changes, because
the backend is responsible for converting its own keypoint layout to the
canonical one before returning.

A backend is expected to be cheap to construct and expensive to warm up, so
model loading belongs in :meth:`PoseBackend.setup`, which the pipeline calls
once before the first frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..skeleton import CANONICAL, SkeletonLayout, to_canonical
from ..types import PersonPose


@dataclass
class BackendInfo:
    """Static description of a backend, readable without installing it.

    The demo app lists every registered backend with this metadata and greys
    out the ones whose dependencies are missing, rather than hiding them -
    seeing that Sapiens exists is how a user learns it is an option.
    """

    key: str
    label: str
    layout: SkeletonLayout
    #: One-line summary shown in the UI.
    summary: str
    #: Rough speed class on a consumer GPU: "realtime", "fast", "slow".
    speed: str
    #: Whether it natively handles several people per frame.
    multi_person: bool
    #: Whether a GPU is effectively required.
    needs_gpu: bool
    #: Exact command that makes this backend work.
    install_hint: str = ""
    #: Upstream project, so the citation trail stays intact.
    homepage: str = ""
    #: False while the integration is still a stub.
    implemented: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


class PoseBackend(ABC):
    """Estimate 2D keypoints for every person in a frame."""

    #: Filled in by subclasses.
    info: BackendInfo

    def __init__(self, **options: Any) -> None:
        self.options = options
        self._ready = False

    # -- lifecycle --------------------------------------------------------

    @classmethod
    @abstractmethod
    def available(cls) -> tuple[bool, str]:
        """Whether this backend can run here.

        Returns ``(True, "")`` when usable, or ``(False, reason)`` when not.
        Must not import the heavy dependency - check for it with
        ``importlib.util.find_spec`` so listing backends stays cheap.
        """

    def setup(self) -> None:
        """Load models. Called once before the first :meth:`estimate`."""
        self._ready = True

    def close(self) -> None:
        """Release models and device memory."""
        self._ready = False

    def __enter__(self) -> PoseBackend:
        self.setup()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- inference --------------------------------------------------------

    @abstractmethod
    def estimate_native(self, image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Run the model on one RGB frame.

        Returns one ``(xy, confidence)`` pair per person, in the backend's own
        keypoint layout and in the pixel coordinates of ``image``.
        """

    def estimate(self, image: np.ndarray, scale: float = 1.0) -> list[PersonPose]:
        """Estimate poses and return them in canonical layout.

        Args:
            image: ``(H, W, 3)`` uint8 RGB frame.
            scale: the factor ``image`` was downscaled by. Keypoints are
                divided by it so the result is in original-video pixels.
        """
        if not self._ready:
            self.setup()

        people: list[PersonPose] = []
        for native_xy, native_conf in self.estimate_native(image):
            xy, confidence = to_canonical(native_xy, native_conf, self.info.layout)
            if scale != 1.0:
                xy = xy / scale
                native_xy = native_xy / scale
            people.append(
                PersonPose(
                    xy=xy,
                    confidence=confidence,
                    layout=CANONICAL,
                    raw_xy=native_xy if self.info.layout.name != CANONICAL.name else None,
                    raw_layout=self.info.layout
                    if self.info.layout.name != CANONICAL.name
                    else None,
                )
            )
        return people

    def __repr__(self) -> str:
        return f"<{type(self).__name__} key={self.info.key!r} ready={self._ready}>"


class StubBackend(PoseBackend):
    """Base for backends that are registered but not yet wired up.

    These exist so the architecture, the docs and the UI all agree on what is
    planned, and so the error a user hits names the work that is missing
    rather than looking like a crash.
    """

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return False, "integration not implemented yet"

    def setup(self) -> None:
        from ..exceptions import BackendNotImplementedError

        raise BackendNotImplementedError(
            self.info.key,
            f"the {self.info.label} integration is a planned stub with no inference code yet",
            self.info.install_hint or None,
        )

    def estimate_native(self, image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        self.setup()  # always raises
        return []
