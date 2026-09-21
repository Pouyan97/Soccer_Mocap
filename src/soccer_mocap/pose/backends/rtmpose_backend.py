"""RTMPose backend - planned integration.

RTMPose (from the MMPose project) is the practical middle ground: clearly
more accurate than MediaPipe, an order of magnitude faster than Sapiens, and
it ships a Halpe-26 variant whose extra foot keypoints (big toe, small toe,
heel) matter for a sport played with the feet. The ``rtmlib`` package runs it
through ONNXRuntime without pulling in the whole mmcv stack, which is the
reason to prefer it over a raw MMPose install.

For match footage this is the backend to reach for: RTMO, its one-stage
sibling, handles crowded multi-person frames at real-time rates, which is
exactly the failure case for MediaPipe.

Implementing this means:

1. ``pip install rtmlib onnxruntime-gpu`` (or the CPU build).
2. Instantiate ``rtmlib.Wholebody`` or ``rtmlib.Body`` with the desired mode;
   models are fetched on first use and cached.
3. Call it per frame - it returns ``(keypoints, scores)`` already batched per
   person, so ``estimate_native`` is close to a passthrough.
4. Register the Halpe-26 layout that :mod:`soccer_mocap.skeleton` already
   defines, and the canonical mapping is handled for you.
"""

from __future__ import annotations

from ...skeleton import HALPE26
from ..base import BackendInfo, StubBackend
from ..registry import register


@register
class RTMPoseBackend(StubBackend):
    """Placeholder for the RTMPose / rtmlib integration."""

    info = BackendInfo(
        key="rtmpose",
        label="RTMPose (MMPose)",
        layout=HALPE26,
        summary=(
            "Fast and accurate multi-person 2D pose with foot keypoints. "
            "The pragmatic choice for match footage."
        ),
        speed="fast",
        multi_person=True,
        needs_gpu=False,
        install_hint=(
            "mamba run -n soccer-mocap pip install rtmlib onnxruntime-gpu\n"
            "    then implement estimate_native() in "
            "src/soccer_mocap/pose/backends/rtmpose_backend.py"
        ),
        homepage="https://github.com/Tau-J/rtmlib",
        implemented=False,
        extras={
            "variants": ["rtmpose-t", "rtmpose-s", "rtmpose-m", "rtmpose-l", "rtmo"],
            "native_keypoints": 26,
            "runtime": "onnxruntime",
        },
    )
