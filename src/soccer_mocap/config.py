"""Configuration objects and YAML loading.

One nested dataclass tree describes a whole pipeline run. Configs are plain
YAML so an experiment is reproducible from a file in ``configs/``, and every
field has a default so a bare ``PipelineConfig()`` is a valid run.

Unknown keys raise rather than being ignored: a silently-dropped typo in a
config is how you end up believing you ran an ablation you never ran.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from .exceptions import ConfigError

T = TypeVar("T")


@dataclass
class VideoConfig:
    """Decoding and preprocessing of the source clip."""

    #: Device preset key from :mod:`soccer_mocap.devices.presets`.
    device_preset: str = "generic"
    #: Processing profile name, or ``None`` to take the preset's recommendation.
    profile: str | None = None
    #: Honour the container rotation flag. Effectively always wanted for phones.
    apply_rotation: bool = True
    #: Trim to ``[start_s, end_s)``. ``None`` means the clip's own bounds.
    start_s: float | None = None
    end_s: float | None = None
    #: Decoder to prefer; "auto" tries PyAV then falls back to OpenCV.
    decoder: str = "auto"


@dataclass
class PoseConfig:
    """Which pose backend to run and how to filter its output."""

    backend: str = "mediapipe"
    #: Drop joints below this confidence before any analysis.
    min_joint_confidence: float = 0.3
    #: Drop whole detections whose mean confidence is below this.
    min_person_confidence: float = 0.4
    #: Extra options forwarded verbatim to the backend constructor.
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackingConfig:
    """Linking per-frame detections into tracks."""

    enabled: bool = True
    method: str = "iou"
    #: Minimum IoU to consider two boxes the same person.
    iou_threshold: float = 0.3
    #: Frames a track survives without a detection before it is closed.
    max_age: int = 30
    #: Consecutive detections before a track is considered confirmed.
    min_hits: int = 3
    #: Two detections overlapping by more than this are treated as the same
    #: person, and the less confident one is discarded before tracking.
    duplicate_iou: float = 0.6


@dataclass
class CalibrationConfig:
    """Mapping image pixels onto pitch metres."""

    enabled: bool = False
    #: "manual" uses points from ``pitch_points``; "auto" needs a line detector.
    method: str = "manual"
    #: Image-space points, paired with ``pitch_points`` by position.
    image_points: list[list[float]] = field(default_factory=list)
    #: Pitch-space points in metres, origin at the centre spot.
    pitch_points: list[list[float]] = field(default_factory=list)
    #: Fallback scale when no homography is available: player height in metres.
    assumed_player_height_m: float = 1.80


@dataclass
class GoalkeeperConfig:
    """Goalkeeper-specific analysis.

    ``enabled`` gates the whole section; the rest only matter when it is on.
    """

    enabled: bool = False
    #: How to pick the keeper out of the tracks. See :mod:`goalkeeper.identify`.
    identification: str = "auto"
    #: Used when ``identification == "manual"``.
    track_id: int | None = None
    #: Goal mouth in image space as ``[[x1, y1], [x2, y2]]`` (left post, right post).
    goal_posts: list[list[float]] | None = None
    #: Standard goal dimensions in metres. Gives absolute scale near the goal.
    goal_width_m: float = 7.32
    goal_height_m: float = 2.44
    #: Wrist speed that marks the start of a dive, in body-heights per second.
    dive_speed_threshold: float = 1.2
    #: Minimum frames a dive must last to be reported, filters twitches.
    min_dive_frames: int = 4
    #: Torso-angle change that distinguishes a dive from a step, in degrees.
    dive_lean_threshold_deg: float = 25.0


@dataclass
class AnalysisConfig:
    """General (non-goalkeeper) match analysis."""

    #: Butterworth cutoff for joint trajectories, in Hz. None disables filtering.
    filter_cutoff_hz: float | None = 6.0
    filter_order: int = 4
    #: Compute joint angles, velocities and accelerations.
    kinematics: bool = True
    #: Detect ball contacts / strikes from foot kinematics.
    events: bool = True


@dataclass
class OutputConfig:
    """What to write and where."""

    directory: str = "outputs"
    #: Formats to export: any of json, csv, parquet, trc.
    formats: list[str] = field(default_factory=lambda: ["json"])
    #: Render an overlay video with the skeleton drawn on.
    overlay_video: bool = True
    #: Write the per-frame plots and the summary report.
    report: bool = True


@dataclass
class PipelineConfig:
    """Everything one run of the pipeline needs."""

    name: str = "default"
    video: VideoConfig = field(default_factory=VideoConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    goalkeeper: GoalkeeperConfig = field(default_factory=GoalkeeperConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return _build(cls, data, path="")

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load a config from a YAML file."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ConfigError(
                "Reading YAML configs needs PyYAML.\n"
                "    mamba install -n soccer-mocap -c conda-forge pyyaml"
            ) from exc

        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"Config root must be a mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        """Write this config out, so a run can record exactly what it used."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=False)


def _build(target: type[T], data: dict[str, Any], path: str) -> T:
    """Recursively construct a dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping at {path or 'root'}, got {type(data).__name__}")

    known = {f.name: f for f in fields(target)}  # type: ignore[arg-type]
    unknown = set(data) - set(known)
    if unknown:
        where = f" in section '{path}'" if path else ""
        raise ConfigError(
            f"Unknown config key(s){where}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )

    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        spec = known[key]
        child_path = f"{path}.{key}" if path else key
        nested = _nested_dataclass_type(spec)
        if nested is not None and isinstance(value, dict):
            kwargs[key] = _build(nested, value, child_path)
        else:
            kwargs[key] = value
    return target(**kwargs)  # type: ignore[return-value]


def _nested_dataclass_type(spec: dataclasses.Field) -> type | None:
    """Return the dataclass type behind a field, or ``None`` if it is a leaf.

    ``from __future__ import annotations`` turns every annotation into a
    string, so the type is recovered by instantiating the field's
    ``default_factory`` rather than by inspecting ``spec.type``.
    """
    if spec.default_factory is dataclasses.MISSING:  # type: ignore[misc]
        return spec.type if is_dataclass(spec.type) else None
    default = spec.default_factory()  # type: ignore[misc]
    return type(default) if is_dataclass(default) else None


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load ``path``, or return the built-in defaults when it is ``None``."""
    if path is None:
        return PipelineConfig()
    return PipelineConfig.from_yaml(path)
