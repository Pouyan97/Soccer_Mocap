"""Video probing and frame iteration, with phone footage in mind.

Two decoders are supported. PyAV is preferred because it exposes the
container-level rotation flag and real presentation timestamps, both of which
phones rely on; OpenCV is the fallback because it is the one that is always
installed. The reader hides the difference behind :class:`VideoReader`.

Rotation is the subtle part. A phone held upright records landscape pixels
plus a "rotate 90" flag. OpenCV ignores the flag entirely and hands back
sideways frames. PyAV's behaviour depends on the version. So we read the flag
ourselves, and apply it ourselves, exactly once.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..exceptions import VideoReadError
from ..types import VideoMetadata


@dataclass
class Frame:
    """One decoded frame with the timing needed to place it in the clip."""

    index: int
    #: Seconds from the start of the clip. From container timestamps when
    #: available, since phone frame rates are often nominal rather than exact.
    timestamp: float
    #: ``(H, W, 3)`` uint8 RGB, already rotated and scaled.
    image: np.ndarray
    #: Factor the frame was downscaled by; multiply coordinates by its inverse
    #: to map a detection back onto the original pixels.
    scale: float = 1.0

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


def _have(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def probe(path: str | Path) -> VideoMetadata:
    """Read a clip's metadata without decoding its pixels."""
    path = Path(path)
    if not path.exists():
        raise VideoReadError(f"Video not found: {path}")

    if _have("av"):
        try:
            return _probe_av(path)
        except Exception:  # noqa: BLE001 - fall through to the other decoder
            pass
    if _have("cv2"):
        return _probe_cv2(path)

    raise VideoReadError(
        "No video decoder available. Install one into the environment:\n"
        "    mamba install -n soccer-mocap -c conda-forge av"
    )


def _probe_av(path: Path) -> VideoMetadata:
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise VideoReadError(f"No video stream in {path}")
        stream = container.streams.video[0]

        fps = float(stream.average_rate) if stream.average_rate else 30.0
        duration = None
        if stream.duration is not None and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000.0

        # Phones write the orientation into a display matrix side-data entry.
        rotation = 0
        raw_rotation = stream.metadata.get("rotate")
        if raw_rotation:
            rotation = int(float(raw_rotation)) % 360
        else:
            side = getattr(stream, "side_data", None)
            if side:
                for entry in side:
                    if "DISPLAYMATRIX" in str(entry).upper():
                        value = getattr(entry, "rotation", None)
                        if value is not None:
                            rotation = int(-float(value)) % 360
                        break

        frame_count = stream.frames or 0
        if frame_count == 0 and duration:
            frame_count = int(round(duration * fps))

        return VideoMetadata(
            path=str(path),
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=fps,
            frame_count=frame_count,
            rotation=rotation,
            codec=stream.codec_context.name or "unknown",
            duration_s=duration,
        )


def _probe_cv2(path: Path) -> VideoMetadata:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoReadError(
            f"Could not open {path}. If this is an iPhone .mov, the codec is "
            "probably HEVC, which needs PyAV:\n"
            "    mamba install -n soccer-mocap -c conda-forge av"
        )
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Present since OpenCV 4.5; 0 on builds that lack it.
        rotation = int(capture.get(getattr(cv2, "CAP_PROP_ORIENTATION_META", -1)) or 0) % 360
        return VideoMetadata(
            path=str(path),
            width=width,
            height=height,
            fps=float(fps),
            frame_count=frame_count,
            rotation=rotation,
            codec="unknown",
            duration_s=frame_count / fps if fps > 0 else None,
        )
    finally:
        capture.release()


def apply_rotation(image: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate a frame to its intended display orientation.

    ``rotation`` is the container flag in degrees clockwise, so we rotate by
    the same amount to undo it.
    """
    if rotation % 360 == 0:
        return image
    turns = (rotation // 90) % 4
    # np.rot90 turns counter-clockwise, the flag is clockwise.
    return np.ascontiguousarray(np.rot90(image, k=-turns))


def resize_long_edge(image: np.ndarray, target: int | None) -> tuple[np.ndarray, float]:
    """Downscale so the longest edge is ``target``. Never upscales.

    Returns the image and the scale factor applied, so keypoints can be mapped
    back to original pixel coordinates.
    """
    if target is None:
        return image, 1.0
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= target:
        return image, 1.0

    scale = target / long_edge
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))

    try:
        import cv2

        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    except ImportError:
        # Nearest-neighbour fallback keeps the package importable without cv2.
        rows = (np.arange(new_size[1]) / scale).astype(int).clip(0, height - 1)
        cols = (np.arange(new_size[0]) / scale).astype(int).clip(0, width - 1)
        resized = image[rows][:, cols]
    return resized, scale


class VideoReader:
    """Iterate a clip as RGB frames, rotated and optionally downscaled.

    Args:
        path: clip to read.
        target_long_edge: downscale so the longest edge is this many pixels.
        stride: yield every Nth frame.
        start_s / end_s: trim window in seconds.
        apply_rotation_flag: honour the container rotation. Almost always yes;
            exposed only for debugging footage whose flag is wrong.

    Example:
        >>> with VideoReader("clip.mov", target_long_edge=1280) as reader:
        ...     for frame in reader:
        ...         process(frame.image)
    """

    def __init__(
        self,
        path: str | Path,
        target_long_edge: int | None = None,
        stride: int = 1,
        start_s: float | None = None,
        end_s: float | None = None,
        apply_rotation_flag: bool = True,
        decoder: str = "auto",
    ) -> None:
        self.path = Path(path)
        self.meta = probe(self.path)
        self.target_long_edge = target_long_edge
        self.stride = max(1, int(stride))
        self.start_s = start_s
        self.end_s = end_s
        self.apply_rotation_flag = apply_rotation_flag
        self.decoder = self._choose_decoder(decoder)
        self._handle = None

    @staticmethod
    def _choose_decoder(requested: str) -> str:
        if requested == "auto":
            if _have("av"):
                return "av"
            if _have("cv2"):
                return "cv2"
            raise VideoReadError(
                "No video decoder available. Install one:\n"
                "    mamba install -n soccer-mocap -c conda-forge av"
            )
        if not _have(requested):
            raise VideoReadError(f"Requested decoder {requested!r} is not installed.")
        return requested

    # -- context manager --------------------------------------------------

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is None:
            return
        if self.decoder == "av":
            self._handle.close()
        else:
            self._handle.release()
        self._handle = None

    # -- iteration --------------------------------------------------------

    def __iter__(self) -> Iterator[Frame]:
        if self.decoder == "av":
            yield from self._iter_av()
        else:
            yield from self._iter_cv2()

    def _postprocess(self, image: np.ndarray, index: int, timestamp: float) -> Frame:
        if self.apply_rotation_flag:
            image = apply_rotation(image, self.meta.rotation)
        image, scale = resize_long_edge(image, self.target_long_edge)
        return Frame(index=index, timestamp=timestamp, image=image, scale=scale)

    def _iter_av(self) -> Iterator[Frame]:
        import av

        container = av.open(str(self.path))
        self._handle = container
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            # Counts frames inside the trim window, so the stride is applied
            # relative to the window rather than to the start of the file.
            considered = 0
            for index, av_frame in enumerate(container.decode(stream)):
                timestamp = (
                    float(av_frame.pts * stream.time_base)
                    if av_frame.pts is not None and stream.time_base
                    else index / self.meta.fps
                )
                if self.start_s is not None and timestamp < self.start_s:
                    continue
                if self.end_s is not None and timestamp >= self.end_s:
                    break
                take = considered % self.stride == 0
                considered += 1
                if not take:
                    continue
                yield self._postprocess(av_frame.to_ndarray(format="rgb24"), index, timestamp)
        finally:
            self.close()

    def _iter_cv2(self) -> Iterator[Frame]:
        import cv2

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise VideoReadError(f"Could not open {self.path}")
        self._handle = capture
        try:
            index = 0
            considered = 0
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                timestamp = index / self.meta.fps if self.meta.fps > 0 else 0.0
                index += 1

                if self.start_s is not None and timestamp < self.start_s:
                    continue
                if self.end_s is not None and timestamp >= self.end_s:
                    break
                take = considered % self.stride == 0
                considered += 1
                if not take:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield self._postprocess(rgb, index - 1, timestamp)
        finally:
            self.close()

    def read_all(self, limit: int | None = None) -> list[Frame]:
        """Decode into memory. Only for short clips - 4K frames are 25 MB each."""
        frames: list[Frame] = []
        for frame in self:
            frames.append(frame)
            if limit is not None and len(frames) >= limit:
                break
        return frames


def write_video(
    path: str | Path,
    frames: list[np.ndarray],
    fps: float,
    codec: str = "mp4v",
) -> Path:
    """Write RGB frames to a file.

    ``mp4v`` is the default because it is the codec every OpenCV build can
    write. It is not the most efficient, but an overlay video that will not
    open is worse than one that is larger than it needs to be.
    """
    try:
        import cv2
    except ImportError as exc:
        raise VideoReadError(
            "Writing video needs OpenCV:\n"
            "    mamba run -n soccer-mocap pip install opencv-contrib-python"
        ) from exc

    if not frames:
        raise VideoReadError("No frames to write.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise VideoReadError(f"Could not open a writer for {path} with codec {codec!r}")
    try:
        for image in frames:
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path
