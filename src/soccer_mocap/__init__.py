"""Single-camera markerless motion capture and analysis for soccer.

The package is organised as a pipeline of swappable stages:

    video -> pose -> tracking -> lifting -> calibration -> analysis -> export

The pose stage is the one that matters most and the one most likely to be
replaced, so it sits behind a registry: MediaPipe runs today, and Sapiens,
RTMPose and PosePipeline are registered as stubs with their integration
notes. Nothing downstream knows which backend produced a pose, because every
backend converts to one canonical COCO-17 skeleton on the way out.

A goalkeeper-specific package sits alongside the general analysis, because
the keeper is a different measurement problem: a known region of the pitch,
a goal of exactly known size to calibrate against, and short explosive
actions rather than continuous running.

Quick start:

    from soccer_mocap import Pipeline, PipelineConfig

    config = PipelineConfig()
    config.goalkeeper.enabled = True
    config.goalkeeper.goal_posts = [[410, 520], [980, 528]]

    result = Pipeline(config).run("keeper.mov")
    print(result.goalkeeper)

Or from a terminal:

    soccer-mocap run keeper.mov --goalkeeper --goal-posts 410,520 980,528
    soccer-mocap demo          # the phone-friendly web app
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import (
    AnalysisConfig,
    CalibrationConfig,
    GoalkeeperConfig,
    OutputConfig,
    PipelineConfig,
    PoseConfig,
    TrackingConfig,
    VideoConfig,
    load_config,
)
from .exceptions import (
    BackendNotImplementedError,
    BackendUnavailableError,
    CalibrationError,
    ConfigError,
    SoccerMocapError,
    VideoReadError,
)
from .pipeline import Pipeline, PipelineResult, run_pipeline
from .skeleton import CANONICAL, COCO17, J, SkeletonLayout, get_layout
from .types import BBox, FramePoses, PersonPose, PoseSequence, Track, VideoMetadata

__all__ = [
    "AnalysisConfig",
    "BBox",
    "BackendNotImplementedError",
    "BackendUnavailableError",
    "CANONICAL",
    "COCO17",
    "CalibrationConfig",
    "CalibrationError",
    "ConfigError",
    "FramePoses",
    "GoalkeeperConfig",
    "J",
    "OutputConfig",
    "PersonPose",
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "PoseConfig",
    "PoseSequence",
    "SkeletonLayout",
    "SoccerMocapError",
    "Track",
    "TrackingConfig",
    "VideoConfig",
    "VideoMetadata",
    "VideoReadError",
    "__version__",
    "get_layout",
    "load_config",
    "run_pipeline",
]
