"""Phone capture profiles and the processing settings that suit them.

The demo app is meant to be handed to someone with whatever phone they own,
and phone footage varies in ways that change the result rather than just the
file size:

* **Orientation.** Phones record in the sensor's native landscape orientation
  and mark portrait footage with a container rotation flag. Decoders that
  ignore the flag hand you sideways frames, and a sideways person is the one
  thing a pose model reliably fails on.
* **Frame rate.** 30, 60, and 240 fps all show up. Every velocity, dive
  duration, and reaction time is divided by this number, so guessing it wrong
  scales the biomechanics by the same factor.
* **Rolling shutter.** Fast limb motion skews on CMOS sensors. Worse on
  budget phones, and it biases ankle/wrist positions at exactly the moments
  (ball strike, dive) we care about.
* **Field of view.** Ultra-wide lenses have heavy barrel distortion that
  breaks the pitch homography unless it is undistorted first.

A preset records what to expect from a given class of phone; a
:class:`ProcessingProfile` records what to do about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import VideoMetadata


@dataclass(frozen=True)
class ProcessingProfile:
    """How hard to work per frame.

    ``target_long_edge`` is the single biggest runtime lever: pose models are
    resized to a fixed input anyway, so downscaling 4K footage before
    inference costs almost no accuracy and saves most of the decode time.
    """

    name: str
    description: str
    #: Longest image edge fed to the model, in pixels. ``None`` keeps native.
    target_long_edge: int | None
    #: Process every Nth frame. 1 = every frame.
    frame_stride: int = 1
    #: Backend-specific quality knob (MediaPipe model_complexity 0/1/2).
    model_complexity: int = 1
    #: Whether to run the (slower) multi-person detect-then-pose path.
    multi_person: bool = True
    #: Temporal smoothing window in frames, 0 to disable.
    smoothing_window: int = 5


FAST = ProcessingProfile(
    name="fast",
    description="Preview quality. Runs on any laptop, good for checking framing.",
    target_long_edge=640,
    frame_stride=2,
    model_complexity=0,
    multi_person=False,
    smoothing_window=3,
)

BALANCED = ProcessingProfile(
    name="balanced",
    description="Default. Full frame rate at 720p-equivalent inference.",
    target_long_edge=1280,
    frame_stride=1,
    model_complexity=1,
    multi_person=True,
    smoothing_window=5,
)

ACCURATE = ProcessingProfile(
    name="accurate",
    description="Native resolution, heaviest model. For biomechanics, not for demos.",
    target_long_edge=None,
    frame_stride=1,
    model_complexity=2,
    multi_person=True,
    smoothing_window=3,
)

PROFILES: dict[str, ProcessingProfile] = {p.name: p for p in (FAST, BALANCED, ACCURATE)}


@dataclass(frozen=True)
class DevicePreset:
    """Expected capture characteristics for one class of phone."""

    key: str
    label: str
    os: str
    #: Frame rates the camera app commonly produces, slow-motion included.
    common_fps: tuple[float, ...]
    #: Typical recorded resolution as ``(width, height)`` in landscape.
    typical_resolution: tuple[int, int]
    #: Container codec the default camera app writes.
    codec: str
    #: Horizontal field of view of the main lens, degrees. Used as the seed
    #: for pitch calibration when no intrinsics are supplied.
    fov_deg: float
    #: How strongly this sensor skews fast motion. Drives the warning shown
    #: in the demo app, not a correction - correcting it needs the readout time.
    rolling_shutter: str = "moderate"
    recommended_profile: str = "balanced"
    notes: str = ""

    def profile(self) -> ProcessingProfile:
        return PROFILES[self.recommended_profile]


# --------------------------------------------------------------------------
# Presets. Deliberately coarse: these are capture-behaviour classes, not a
# device database. A phone that is not listed falls back to GENERIC, and the
# values are then refined from the file's own metadata by `infer_preset`.
# --------------------------------------------------------------------------

IPHONE_RECENT = DevicePreset(
    key="iphone_recent",
    label="iPhone 12 or newer",
    os="ios",
    common_fps=(30.0, 60.0, 120.0, 240.0),
    typical_resolution=(3840, 2160),
    codec="hevc",
    fov_deg=69.0,
    rolling_shutter="low",
    recommended_profile="balanced",
    notes=(
        "Records HEVC in a .mov container and stores portrait as a rotation "
        "flag. Needs PyAV or a recent OpenCV build to decode. Cinematic and "
        "Action modes crop and stabilise, which moves the apparent camera - "
        "disable both before recording anything you intend to calibrate."
    ),
)

IPHONE_OLDER = DevicePreset(
    key="iphone_older",
    label="iPhone 8 to 11",
    os="ios",
    common_fps=(30.0, 60.0, 240.0),
    typical_resolution=(1920, 1080),
    codec="h264",
    fov_deg=65.0,
    rolling_shutter="moderate",
    recommended_profile="balanced",
)

ANDROID_FLAGSHIP = DevicePreset(
    key="android_flagship",
    label="Android flagship (Pixel 6+, Galaxy S21+)",
    os="android",
    common_fps=(30.0, 60.0, 120.0, 240.0),
    typical_resolution=(3840, 2160),
    codec="h264",
    fov_deg=70.0,
    rolling_shutter="low",
    recommended_profile="balanced",
    notes=(
        "Frame rate is often nominal rather than exact; prefer the timestamps "
        "the decoder reports over fps * frame_index when timing events."
    ),
)

ANDROID_MIDRANGE = DevicePreset(
    key="android_midrange",
    label="Android mid-range / budget",
    os="android",
    common_fps=(30.0,),
    typical_resolution=(1920, 1080),
    codec="h264",
    fov_deg=72.0,
    rolling_shutter="high",
    recommended_profile="fast",
    notes=(
        "Pronounced rolling shutter. Limb positions during a strike or dive "
        "are skewed, so treat peak angular velocities as lower bounds."
    ),
)

ACTION_CAM = DevicePreset(
    key="action_cam",
    label="Action camera (GoPro and similar)",
    os="other",
    common_fps=(30.0, 60.0, 120.0, 240.0),
    typical_resolution=(3840, 2160),
    codec="hevc",
    fov_deg=118.0,
    rolling_shutter="moderate",
    recommended_profile="balanced",
    notes=(
        "Wide-angle barrel distortion is severe enough to break the pitch "
        "homography. Undistort first, or record in the narrowest FOV setting."
    ),
)

GENERIC = DevicePreset(
    key="generic",
    label="Other / unknown phone",
    os="unknown",
    common_fps=(30.0, 60.0),
    typical_resolution=(1920, 1080),
    codec="h264",
    fov_deg=70.0,
    rolling_shutter="moderate",
    recommended_profile="balanced",
)

DEVICE_PRESETS: dict[str, DevicePreset] = {
    preset.key: preset
    for preset in (
        IPHONE_RECENT,
        IPHONE_OLDER,
        ANDROID_FLAGSHIP,
        ANDROID_MIDRANGE,
        ACTION_CAM,
        GENERIC,
    )
}


def get_preset(key: str) -> DevicePreset:
    """Look up a device preset, falling back to :data:`GENERIC`."""
    return DEVICE_PRESETS.get(key, GENERIC)


def get_profile(name: str) -> ProcessingProfile:
    """Look up a processing profile by name."""
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown processing profile {name!r}. Available: {', '.join(sorted(PROFILES))}"
        ) from exc


def infer_preset(video: VideoMetadata) -> DevicePreset:
    """Best guess at which preset matches a clip, from its metadata alone.

    Only the codec and resolution are really diagnostic, so this is a hint for
    the demo app's default selection rather than a detector. The user can
    always override it.
    """
    codec = video.codec.lower()
    long_edge = max(video.width, video.height)

    if "hevc" in codec or "h265" in codec:
        # HEVC at 4K from a 16:9 sensor is the iPhone default; much wider than
        # 16:9 suggests an action cam.
        aspect = long_edge / max(1, min(video.width, video.height))
        if aspect > 1.9:
            return ACTION_CAM
        return IPHONE_RECENT

    if long_edge >= 3000:
        return ANDROID_FLAGSHIP
    if video.fps <= 30.5 and long_edge <= 1920:
        return ANDROID_MIDRANGE
    return GENERIC


def resolve_profile(
    video: VideoMetadata,
    preset: DevicePreset,
    override: str | None = None,
) -> ProcessingProfile:
    """Pick the processing profile for a clip.

    An explicit ``override`` always wins. Otherwise the preset's
    recommendation is adjusted for two cases the preset cannot know about:
    slow-motion clips (many frames, little new information per frame) and
    very long clips.
    """
    if override is not None:
        return get_profile(override)

    profile = preset.profile()

    if video.is_high_speed and profile.frame_stride == 1:
        # 240 fps of a 2-second dive is 480 frames of a mostly static scene.
        # Halving the rate keeps dive-phase timing well inside 10 ms.
        profile = ProcessingProfile(**{**profile.__dict__, "frame_stride": 2})

    if video.duration_s and video.duration_s > 120:
        profile = ProcessingProfile(
            **{**profile.__dict__, "target_long_edge": min(profile.target_long_edge or 1920, 1280)}
        )

    return profile


@dataclass
class CaptureAdvice:
    """Guidance shown in the demo app before a user records."""

    device: DevicePreset
    warnings: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)


def capture_advice(preset: DevicePreset, video: VideoMetadata | None = None) -> CaptureAdvice:
    """Build the per-device guidance the demo app displays.

    Kept here rather than in the UI so the CLI can print the same warnings.
    """
    advice = CaptureAdvice(device=preset)

    if preset.rolling_shutter == "high":
        advice.warnings.append(
            "This phone class has strong rolling shutter. Peak limb speeds will "
            "read low, and the ball may appear bent. Film at 60 fps if offered."
        )
    if preset.fov_deg > 100:
        advice.warnings.append(
            "Wide-angle lens: straight pitch lines will bow. Undistort before "
            "using pitch calibration, or switch the camera to a narrower FOV."
        )
    if preset.os == "ios":
        advice.tips.append(
            "Turn off Action mode and Cinematic mode - both crop and stabilise, "
            "which makes the camera pose move between frames."
        )

    advice.tips.extend(
        [
            "Film in landscape. Portrait crops the run-up out of frame.",
            "Keep the camera still on a tripod or a wall. Panning invalidates a "
            "single pitch calibration for the whole clip.",
            "Keep the whole body in frame, feet included - ankle joints drive "
            "most of the gait and strike metrics.",
            "60 fps or better for shooting and diving; 30 fps is enough for "
            "positioning and distance covered.",
        ]
    )

    if video is not None:
        if video.is_portrait:
            advice.warnings.append(
                "This clip is portrait. It will still process, but expect the "
                "player to leave frame during a run-up or a dive."
            )
        if video.fps < 29:
            advice.warnings.append(
                f"Frame rate is {video.fps:.1f} fps. Below 30 fps, ball-contact "
                "timing is uncertain by more than one frame."
            )

    return advice
