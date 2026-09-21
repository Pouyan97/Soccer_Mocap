"""Exception types shared across the package."""

from __future__ import annotations


class SoccerMocapError(Exception):
    """Base class for every error raised by this package."""


class BackendUnavailableError(SoccerMocapError):
    """A pose/detection backend was requested but its dependencies are missing.

    The message is expected to carry the exact install command, so a user who
    hits this in the demo app knows what to run without reading the docs.
    """

    def __init__(self, backend: str, reason: str, install_hint: str | None = None) -> None:
        message = f"Backend '{backend}' is unavailable: {reason}"
        if install_hint:
            message += f"\n\nTo enable it:\n    {install_hint}"
        super().__init__(message)
        self.backend = backend
        self.reason = reason
        self.install_hint = install_hint


class BackendNotImplementedError(BackendUnavailableError):
    """A backend is registered as a planned integration but has no inference code yet."""


class VideoReadError(SoccerMocapError):
    """A video file could not be opened or decoded."""


class CalibrationError(SoccerMocapError):
    """Pitch calibration failed or was never performed."""


class ConfigError(SoccerMocapError):
    """A configuration file is malformed or contains an unknown option."""
