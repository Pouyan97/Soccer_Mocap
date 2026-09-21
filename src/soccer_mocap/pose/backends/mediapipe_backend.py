"""MediaPipe Pose backend - the default, and the one that runs out of the box.

MediaPipe is not the most accurate option here; Sapiens and RTMPose both beat
it on joint error. It is the default because it runs in real time on a CPU
and was trained largely on phone-camera imagery, which is exactly the footage
this project targets. It is the right tool for validating the pipeline end to
end and for the demo app; swap in a heavier backend when you need publishable
joint angles.

**MediaPipe 1.0 removed the old ``mp.solutions`` API**, which used to bundle
its weights inside the wheel. Only the Tasks API remains, and it loads a
``.task`` file from disk, so the model download is now a setup step rather
than an optional extra:

    python scripts/download_models.py mediapipe

One genuine advantage of the Tasks API: it returns ``pose_world_landmarks``,
a metric 3D estimate in metres relative to the mid-hip. That is a real
monocular 3D signal for free, and it is carried through to
:attr:`~soccer_mocap.types.PersonPose.xyz`. Treat it as approximate - it
comes from a model prior about human proportions, not from the camera
geometry - but it is good enough to tell a dive from a step.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from ...skeleton import MEDIAPIPE33, build_canonical_index
from ..base import BackendInfo, PoseBackend
from ..registry import register

#: Where `scripts/download_models.py` puts the task files, relative to the repo root.
MODELS_DIR = Path(__file__).resolve().parents[4] / "models"

#: Searched in order when no explicit ``model_path`` is given.
MODEL_CANDIDATES = (
    "pose_landmarker_heavy.task",
    "pose_landmarker_full.task",
    "pose_landmarker_lite.task",
)

#: Maps the processing profile's 0/1/2 complexity knob onto the task files,
#: so device presets keep working without knowing MediaPipe's file names.
COMPLEXITY_TO_VARIANT = {0: "lite", 1: "full", 2: "heavy"}


def find_model(explicit: str | Path | None = None, complexity: int | None = None) -> Path | None:
    """Locate a ``.task`` file, or return ``None`` if none is installed."""
    if explicit is not None:
        path = Path(explicit)
        return path if path.exists() else None

    if complexity is not None:
        variant = COMPLEXITY_TO_VARIANT.get(int(complexity))
        if variant:
            preferred = MODELS_DIR / f"pose_landmarker_{variant}.task"
            if preferred.exists():
                return preferred

    for name in MODEL_CANDIDATES:
        candidate = MODELS_DIR / name
        if candidate.exists():
            return candidate
    return None


@register
class MediaPipeBackend(PoseBackend):
    """2D (and approximate 3D) pose from MediaPipe Tasks.

    Options:
        model_path: explicit ``.task`` file. Defaults to the best one in
            ``models/``, honouring ``model_complexity`` when several exist.
        model_complexity: 0 lite / 1 full / 2 heavy. Selects the task file.
        num_poses: maximum people to return per frame.
        min_detection_confidence: person-detection threshold.
        min_presence_confidence: per-landmark presence threshold.
        min_tracking_confidence: threshold for reusing the previous frame's
            landmarks instead of re-detecting.
        static_image_mode: process frames independently (IMAGE mode) instead
            of as a video. Wanted when frames are strided far apart, because
            VIDEO mode's tracking assumes near-continuous motion.
    """

    info = BackendInfo(
        key="mediapipe",
        label="MediaPipe Pose",
        layout=MEDIAPIPE33,
        summary=(
            "Real-time 33-landmark pose with approximate metric 3D. "
            "CPU-friendly and trained on phone imagery; least accurate of the four."
        ),
        speed="realtime",
        multi_person=True,
        needs_gpu=False,
        install_hint=(
            "mamba run -n soccer-mocap pip install mediapipe\n"
            "    python scripts/download_models.py mediapipe"
        ),
        homepage="https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker",
        implemented=True,
        extras={"native_keypoints": 33, "provides": ["pose", "approx_3d", "segmentation"]},
    )

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self._landmarker = None
        self._model_path: Path | None = None
        self._video_mode = True
        self._call_index = 0
        #: Metric 3D from the last frame, one ``(33, 3)`` array per person.
        self._last_world: list[np.ndarray] = []

    @classmethod
    def available(cls) -> tuple[bool, str]:
        if importlib.util.find_spec("mediapipe") is None:
            return False, "the 'mediapipe' package is not installed"
        if find_model() is None:
            return False, (
                "no pose_landmarker .task model found in models/ "
                "(MediaPipe 1.0 no longer bundles weights)"
            )
        return True, ""

    # -- lifecycle --------------------------------------------------------

    def setup(self) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = find_model(
            self.options.get("model_path"),
            self.options.get("model_complexity"),
        )
        if model_path is None:
            raise FileNotFoundError(
                "No MediaPipe pose model found. Fetch one with:\n"
                "    python scripts/download_models.py mediapipe"
            )
        self._model_path = model_path

        self._video_mode = not bool(self.options.get("static_image_mode", False))
        running_mode = (
            mp_vision.RunningMode.VIDEO if self._video_mode else mp_vision.RunningMode.IMAGE
        )

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_poses=int(self.options.get("num_poses", 4)),
            min_pose_detection_confidence=float(
                self.options.get("min_detection_confidence", 0.5)
            ),
            min_pose_presence_confidence=float(self.options.get("min_presence_confidence", 0.5)),
            min_tracking_confidence=float(self.options.get("min_tracking_confidence", 0.5)),
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._call_index = 0
        self._ready = True

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._ready = False

    # -- inference --------------------------------------------------------

    def estimate_native(self, image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        import mediapipe as mp

        height, width = image.shape[:2]
        # MediaPipe needs contiguous uint8 RGB and misbehaves quietly on a
        # non-contiguous view, which is what rotating a frame produces.
        rgb = np.ascontiguousarray(image, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._video_mode:
            # VIDEO mode requires strictly increasing timestamps. Counting
            # calls is safer than deriving from a frame index, which is not
            # monotonic once frames are strided or a clip is re-run.
            timestamp_ms = self._call_index * 33
            self._call_index += 1
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        landmark_sets = getattr(result, "pose_landmarks", None) or []
        world_sets = getattr(result, "pose_world_landmarks", None) or []

        self._last_world = [_world_to_array(w) for w in world_sets]
        return [_landmarks_to_arrays(lm, width, height) for lm in landmark_sets]

    def estimate(self, image: np.ndarray, scale: float = 1.0):  # type: ignore[override]
        """Estimate poses, attaching MediaPipe's metric 3D to each person."""
        people = super().estimate(image, scale=scale)
        for person, world in zip(people, self._last_world, strict=False):
            # World landmarks are already metric and hip-relative, so the
            # image downscale factor does not apply to them. They do need the
            # same 33 -> 17 remap as the 2D points, or xyz and xy would be
            # indexed by different skeletons.
            person.xyz = _world_to_canonical(world)
        return people


def _landmarks_to_arrays(landmarks: Any, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalised landmarks to pixel ``(xy, confidence)`` arrays.

    MediaPipe normalises to ``[0, 1]`` and will report values outside that
    range for joints it extrapolates past the frame edge. Those are kept - a
    foot just below the bottom of frame still constrains a stride estimate -
    and their low ``visibility`` is what flags them downstream.
    """
    count = len(landmarks)
    xy = np.empty((count, 2), dtype=float)
    confidence = np.empty(count, dtype=float)
    for i, landmark in enumerate(landmarks):
        xy[i, 0] = landmark.x * width
        xy[i, 1] = landmark.y * height
        # `visibility` is the landmark-level score; `presence` exists too but
        # is less discriminative in practice for occluded limbs.
        confidence[i] = getattr(landmark, "visibility", None) or 0.0
    return xy, confidence


def _world_to_array(landmarks: Any) -> np.ndarray:
    """Convert world landmarks to a ``(33, 3)`` array in metres, hip-relative."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)


#: Gathers MediaPipe's 33 columns into canonical COCO-17 order. Computed once.
_CANONICAL_INDEX = build_canonical_index(MEDIAPIPE33)


def _world_to_canonical(world: np.ndarray) -> np.ndarray:
    """Reorder a ``(33, 3)`` world array into the canonical 17-joint layout."""
    valid = _CANONICAL_INDEX >= 0
    gathered = world[np.where(valid, _CANONICAL_INDEX, 0)]
    gathered[~valid] = np.nan
    return gathered
