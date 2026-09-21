"""Bundled pose backends.

Importing this package registers every backend with the registry. Add a new
one by dropping a module here and importing it below.
"""

from __future__ import annotations

from . import (  # noqa: F401  (imported for the registration side effect)
    mediapipe_backend,
    posepipeline_backend,
    rtmpose_backend,
    sapiens_backend,
)

__all__ = [
    "mediapipe_backend",
    "posepipeline_backend",
    "rtmpose_backend",
    "sapiens_backend",
]
