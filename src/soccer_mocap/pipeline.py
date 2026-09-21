"""The end-to-end pipeline.

Stages, in order:

1. **Probe** the clip and pick a processing profile for the phone it came from.
2. **Pose** - run the chosen backend over the frames.
3. **Track** - link detections into per-person tracks.
4. **Lift** - attach 3D, where the backend or a lifter can supply it.
5. **Calibrate** - fit a pitch homography, if points were given.
6. **Analyse** - goalkeeper report and/or general kinematics.
7. **Export** - write the requested formats and the overlay video.

Each stage is optional and independently testable. The pipeline holds no
state of its own beyond the config, so running two clips concurrently means
constructing two pipelines.

Frames are streamed rather than loaded up front: a two-minute 4K clip is
about 45 GB of decoded RGB, so the only frames kept in memory are the ones
the overlay renderer asked for.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration.pitch import PitchCalibration, feet_position
from .config import PipelineConfig
from .devices.presets import get_preset, infer_preset, resolve_profile
from .goalkeeper import GoalGeometry, GoalkeeperReport, build_report, identify_goalkeeper
from .io.video import VideoReader, probe, write_video
from .io.writers import write_json, write_report
from .pose.registry import default_backend_key, get_backend
from .skeleton import CANONICAL
from .tracking.iou import IoUTracker, suppress_duplicates
from .types import FramePoses, PoseSequence, VideoMetadata

#: Called with ``(fraction_done, message)``. The demo app uses it for a
#: progress bar; the CLI prints it.
ProgressCallback = Callable[[float, str], None]


@dataclass
class PipelineResult:
    """Everything one run produced."""

    sequence: PoseSequence
    video: VideoMetadata
    config: PipelineConfig
    goalkeeper: GoalkeeperReport | None = None
    #: Candidate ball strikes, when general event analysis ran.
    kicks: list = field(default_factory=list)
    #: Ground contacts, for step timing.
    contacts: list = field(default_factory=list)
    calibration: PitchCalibration | None = None
    goal: GoalGeometry | None = None
    #: Rendered overlay frames, when the config asked for them.
    overlay_frames: list[np.ndarray] = field(default_factory=list)
    #: Paths of everything written to disk.
    outputs: dict[str, Path] = field(default_factory=dict)
    #: Per-stage wall-clock seconds.
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def num_people(self) -> int:
        return len(self.sequence.tracks)

    def summary_lines(self) -> list[str]:
        """Plain-text summary for the CLI."""
        lines = [
            f"Video      : {Path(self.video.path).name}",
            f"Resolution : {self.video.display_size[0]}x{self.video.display_size[1]}"
            f"  ({'portrait' if self.video.is_portrait else 'landscape'})",
            f"Frame rate : {self.video.fps:.2f} fps",
            f"Frames     : {len(self.sequence.frames)} processed",
            f"Backend    : {self.sequence.backend}",
            f"People     : {self.num_people} track(s)",
        ]
        if self.calibration is not None:
            lines.append(
                f"Calibration: {self.calibration.reprojection_error_px:.1f} px error"
                f" ({'reliable' if self.calibration.is_reliable else 'POOR'})"
            )
        if self.timings:
            total = sum(self.timings.values())
            stages = ", ".join(f"{k} {v:.1f}s" for k, v in self.timings.items())
            lines.append(f"Timing     : {total:.1f}s total ({stages})")
        if self.warnings:
            lines.append("")
            lines.append("Warnings")
            lines.extend(f"  - {w}" for w in self.warnings)
        if self.kicks or self.contacts:
            from .analysis.events import summarise_events

            lines.append("")
            lines.extend(summarise_events(self.kicks, self.contacts))
        if self.goalkeeper is not None:
            lines.append("")
            lines.extend(self.goalkeeper.summary_lines())
        return lines

    def __str__(self) -> str:
        return "\n".join(self.summary_lines())


class Pipeline:
    """Run a clip through the configured stages.

    Example:
        >>> from soccer_mocap import Pipeline, PipelineConfig
        >>> config = PipelineConfig()
        >>> config.goalkeeper.enabled = True
        >>> config.goalkeeper.goal_posts = [[410, 520], [980, 528]]
        >>> result = Pipeline(config).run("keeper.mov")
        >>> print(result.goalkeeper)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(
        self,
        video_path: str | Path,
        progress: ProgressCallback | None = None,
        keep_frames_for_overlay: bool = True,
    ) -> PipelineResult:
        """Process one clip."""
        config = self.config
        report_progress = progress or (lambda fraction, message: None)

        timings: dict[str, float] = {}
        warnings: list[str] = []

        # -- 1. probe -----------------------------------------------------
        started = time.perf_counter()
        report_progress(0.02, "Reading video metadata")
        video = probe(video_path)

        preset = (
            infer_preset(video)
            if config.video.device_preset in ("auto", "")
            else get_preset(config.video.device_preset)
        )
        profile = resolve_profile(video, preset, config.video.profile)
        timings["probe"] = time.perf_counter() - started

        warnings.extend(_capture_warnings(video, preset))

        # -- 2. pose ------------------------------------------------------
        started = time.perf_counter()
        backend_key = config.pose.backend
        if backend_key in ("auto", ""):
            backend_key = default_backend_key()

        backend_options = {
            "model_complexity": profile.model_complexity,
            "num_poses": 6 if profile.multi_person else 1,
            **config.pose.backend_options,
        }
        backend = get_backend(backend_key, **backend_options)

        frames: list[FramePoses] = []
        kept_images: list[np.ndarray] = []
        overlay_scale = 1.0
        want_overlay = keep_frames_for_overlay and config.output.overlay_video

        expected = _expected_frame_count(video, profile.frame_stride, config)

        with backend:
            reader = VideoReader(
                video_path,
                target_long_edge=profile.target_long_edge,
                stride=profile.frame_stride,
                start_s=config.video.start_s,
                end_s=config.video.end_s,
                apply_rotation_flag=config.video.apply_rotation,
                decoder=config.video.decoder,
            )
            with reader:
                for position, frame in enumerate(reader):
                    people = backend.estimate(frame.image, scale=frame.scale)
                    people = [
                        p
                        for p in people
                        if p.mean_confidence() >= config.pose.min_person_confidence
                    ]
                    # Multi-person models double-detect during fast motion,
                    # and a duplicate would start its own track and split the
                    # subject in two. Collapse them before tracking sees them.
                    people = suppress_duplicates(people, config.tracking.duplicate_iou)
                    frames.append(
                        FramePoses(
                            frame_index=frame.index,
                            timestamp=frame.timestamp,
                            people=people,
                        )
                    )
                    if want_overlay:
                        # Kept at inference resolution to bound memory. The
                        # scale is recorded so the renderer can map keypoints
                        # (which are in original-video pixels) back down.
                        kept_images.append(frame.image.copy())
                        overlay_scale = frame.scale

                    if expected and position % 10 == 0:
                        report_progress(
                            0.05 + 0.65 * min(1.0, position / expected),
                            f"Pose: frame {position}/{expected}",
                        )

        timings["pose"] = time.perf_counter() - started

        if not frames:
            raise RuntimeError(
                "No frames were decoded. The file may be empty, or the trim "
                "window may lie outside the clip."
            )

        sequence = PoseSequence(
            frames=frames,
            video=video,
            layout=CANONICAL,
            backend=backend_key,
            meta={
                "device_preset": preset.key,
                "profile": profile.name,
                "target_long_edge": profile.target_long_edge,
                "frame_stride": profile.frame_stride,
            },
        )

        detections = sum(len(f) for f in frames)
        if detections == 0:
            warnings.append(
                "No people were detected in any frame. Check that the subject is "
                "large enough in frame, and that the clip is not rotated."
            )

        # -- 3. track -----------------------------------------------------
        started = time.perf_counter()
        if config.tracking.enabled:
            report_progress(0.72, "Tracking")
            tracker = IoUTracker(
                iou_threshold=config.tracking.iou_threshold,
                max_age=config.tracking.max_age,
                min_hits=config.tracking.min_hits,
                min_confidence=config.pose.min_person_confidence,
            )
            tracker.run(frames)
        else:
            # Without tracking, treat every frame's people as one track each
            # so downstream code has something coherent to consume.
            for frame in frames:
                for index, person in enumerate(frame.people):
                    person.track_id = index
        sequence.build_tracks()
        timings["tracking"] = time.perf_counter() - started

        # -- 4. lift ------------------------------------------------------
        started = time.perf_counter()
        from .lifting.base import PassthroughLifter

        lifter = PassthroughLifter(smoothing_hz=config.analysis.filter_cutoff_hz)
        for track in sequence.tracks.values():
            lifter.lift_track(track, video.fps)
        timings["lifting"] = time.perf_counter() - started

        # -- 5. calibrate -------------------------------------------------
        calibration = None
        if config.calibration.enabled and config.calibration.image_points:
            started = time.perf_counter()
            report_progress(0.78, "Calibrating pitch")
            try:
                calibration = PitchCalibration.from_points(
                    config.calibration.image_points, config.calibration.pitch_points
                )
                if not calibration.is_reliable:
                    warnings.append(
                        f"Pitch calibration reprojection error is "
                        f"{calibration.reprojection_error_px:.1f} px. Re-check the "
                        "clicked points before trusting metric positions."
                    )
                _project_to_pitch(sequence, calibration)
            except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
                warnings.append(f"Pitch calibration failed: {exc}")
            timings["calibration"] = time.perf_counter() - started

        # -- 6. goalkeeper ------------------------------------------------
        goalkeeper_report = None
        goal = None
        if config.goalkeeper.enabled:
            started = time.perf_counter()
            report_progress(0.82, "Analysing goalkeeper")
            goalkeeper_report, goal, gk_warnings = self._run_goalkeeper(sequence, video)
            warnings.extend(gk_warnings)
            timings["goalkeeper"] = time.perf_counter() - started

        # -- 6b. general events -------------------------------------------
        kicks: list = []
        contacts: list = []
        if config.analysis.events and not config.goalkeeper.enabled:
            started = time.perf_counter()
            report_progress(0.85, "Detecting events")
            subject = sequence.longest_track()
            if subject is not None:
                from .analysis.events import detect_foot_contacts, detect_kicks

                kicks = detect_kicks(subject, video.fps)
                contacts = detect_foot_contacts(subject, video.fps)
            timings["events"] = time.perf_counter() - started

        result = PipelineResult(
            sequence=sequence,
            video=video,
            config=config,
            goalkeeper=goalkeeper_report,
            kicks=kicks,
            contacts=contacts,
            calibration=calibration,
            goal=goal,
            timings=timings,
            warnings=warnings,
        )

        # -- 7. export ----------------------------------------------------
        started = time.perf_counter()
        report_progress(0.88, "Writing outputs")
        highlight = goalkeeper_report.track_id if goalkeeper_report else None

        if want_overlay and kept_images:
            from .viz.overlay import render_overlay_frames

            result.overlay_frames = render_overlay_frames(
                kept_images,
                frames,
                goal=goal,
                dives=goalkeeper_report.dives if goalkeeper_report else None,
                highlight_track=highlight,
                min_confidence=config.pose.min_joint_confidence,
                scale=overlay_scale,
            )

        result.outputs = self._write_outputs(result)
        timings["export"] = time.perf_counter() - started

        report_progress(1.0, "Done")
        return result

    # -- stage helpers ----------------------------------------------------

    def _run_goalkeeper(
        self, sequence: PoseSequence, video: VideoMetadata
    ) -> tuple[GoalkeeperReport | None, GoalGeometry | None, list[str]]:
        """Identify the keeper and build their report."""
        settings = self.config.goalkeeper
        warnings: list[str] = []

        goal = None
        if settings.goal_posts:
            try:
                goal = GoalGeometry.from_config(settings.goal_posts)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Could not use the supplied goal posts: {exc}")

        tracks = sequence.tracks or sequence.build_tracks()
        if not tracks:
            warnings.append("No tracks available, so no goalkeeper analysis was run.")
            return None, goal, warnings

        track_id, candidates = identify_goalkeeper(
            tracks,
            goal=goal,
            method=settings.identification,
            track_id=settings.track_id,
        )
        if track_id is None or track_id not in tracks:
            warnings.append(
                "Could not identify a goalkeeper track. Set goalkeeper.track_id "
                "explicitly, or mark the goal posts."
            )
            return None, goal, warnings

        if candidates and len(candidates) > 1 and candidates[0].score < 0.5:
            warnings.append(
                f"Goalkeeper identification is uncertain (score "
                f"{candidates[0].score:.2f}). Verify track {track_id} in the overlay."
            )

        report = build_report(
            tracks[track_id],
            fps=video.fps,
            goal=goal,
            candidates=candidates,
            speed_threshold=settings.dive_speed_threshold,
            min_dive_frames=settings.min_dive_frames,
            lean_threshold_deg=settings.dive_lean_threshold_deg,
            filter_cutoff_hz=self.config.analysis.filter_cutoff_hz,
        )
        return report, goal, warnings

    def _write_outputs(self, result: PipelineResult) -> dict[str, Path]:
        """Write every requested export. Failures warn rather than abort."""
        config = self.config
        outputs: dict[str, Path] = {}

        directory = Path(config.output.directory) / _run_name(result.video.path)
        directory.mkdir(parents=True, exist_ok=True)

        for fmt in config.output.formats:
            try:
                if fmt == "json":
                    outputs["json"] = write_json(
                        directory / "results.json",
                        result.sequence,
                        goalkeeper_report=result.goalkeeper,
                        config=config,
                        extra={"warnings": result.warnings},
                    )
                elif fmt == "csv":
                    from .io.writers import write_csv

                    outputs["csv"] = write_csv(directory / "keypoints.csv", result.sequence)
                elif fmt == "parquet":
                    from .io.writers import write_parquet

                    outputs["parquet"] = write_parquet(
                        directory / "keypoints.parquet", result.sequence
                    )
                elif fmt == "trc":
                    from .io.writers import write_trc

                    outputs["trc"] = write_trc(
                        directory / "motion.trc",
                        result.sequence,
                        track_id=result.goalkeeper.track_id if result.goalkeeper else None,
                    )
                else:
                    result.warnings.append(f"Unknown output format {fmt!r}; skipped.")
            except Exception as exc:  # noqa: BLE001 - one bad format must not lose the rest
                result.warnings.append(f"Could not write {fmt}: {exc}")

        if config.output.overlay_video and result.overlay_frames:
            try:
                # The overlay is written at the stride the pipeline ran at, so
                # its playback rate has to match or the motion looks wrong.
                stride = result.sequence.meta.get("frame_stride", 1)
                outputs["overlay"] = write_video(
                    directory / "overlay.mp4",
                    result.overlay_frames,
                    fps=result.video.fps / max(1, stride),
                )
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(f"Could not write the overlay video: {exc}")

        if config.output.report:
            outputs["report"] = write_report(
                directory / "report.txt", result.summary_lines()
            )

        return outputs


def _project_to_pitch(sequence: PoseSequence, calibration: PitchCalibration) -> None:
    """Attach pitch coordinates to every detection, projecting from the feet."""
    for frame in sequence.frames:
        for person in frame.people:
            feet = feet_position(person.xy, person.confidence)
            if np.isfinite(feet).all():
                person.xy_pitch = calibration.to_pitch(feet)


def _expected_frame_count(
    video: VideoMetadata, stride: int, config: PipelineConfig
) -> int:
    """How many frames the reader will emit, for the progress bar."""
    total = video.frame_count or 0
    if config.video.start_s or config.video.end_s:
        start = config.video.start_s or 0.0
        end = config.video.end_s or (video.duration_s or 0.0)
        total = int(max(0.0, end - start) * video.fps)
    return max(1, total // max(1, stride))


def _capture_warnings(video: VideoMetadata, preset) -> list[str]:
    """Device-specific caveats, shared with the demo app."""
    from .devices.presets import capture_advice

    return list(capture_advice(preset, video).warnings)


def _run_name(video_path: str) -> str:
    """Directory name for one run: the clip's stem plus a timestamp."""
    stem = Path(video_path).stem or "clip"
    return f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"


def run_pipeline(
    video_path: str | Path,
    config: PipelineConfig | None = None,
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    """One-call convenience wrapper around :class:`Pipeline`."""
    return Pipeline(config).run(video_path, progress=progress)
