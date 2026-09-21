"""Sapiens backend - planned integration.

Sapiens (Meta, ECCV 2024) is a family of ViT models pretrained on 300M human
images, with heads for 2D pose (up to 308 keypoints), body-part segmentation,
depth, and surface normals. For soccer biomechanics it is the most accurate
option on this list, and the depth head is the part that matters most for us:
it is the only backend here that gives a per-pixel depth prior, which is what
makes plausible 3D from a single camera rather than a guess.

Cost: the 0.3B checkpoint needs roughly 4 GB of VRAM, the 2B closer to 20 GB,
and neither runs at anything like real time. This is an offline-analysis
backend.

Implementing this means:

1. Fetch a checkpoint. The torchscript exports on HuggingFace
   (``facebook/sapiens-pose-*``) avoid a dependency on the training repo and
   are the path of least resistance.
2. Run a person detector first. Sapiens pose is top-down: it expects a
   cropped, padded person box, so it pairs with
   :mod:`soccer_mocap.detection.person`.
3. Preprocess each crop to 1024x768, normalised with the ImageNet statistics
   the checkpoints were trained with.
4. Decode heatmaps to keypoints, then map the 308-point layout onto canonical
   COCO-17 by registering a ``SkeletonLayout`` with the right aliases. Keep
   the native array in ``PersonPose.raw_xy`` - the extra foot and spine
   points are the reason to run this model at all.
"""

from __future__ import annotations

from ...skeleton import COCO17
from ..base import BackendInfo, StubBackend
from ..registry import register


@register
class SapiensBackend(StubBackend):
    """Placeholder for the Sapiens pose integration."""

    info = BackendInfo(
        key="sapiens",
        label="Sapiens (Meta)",
        # Registered as COCO-17 until the 308-point layout is added; the
        # canonical subset is what downstream code consumes either way.
        layout=COCO17,
        summary=(
            "Highest accuracy here, plus depth and surface normals for "
            "single-camera 3D. Needs a GPU and a multi-GB checkpoint."
        ),
        speed="slow",
        multi_person=True,
        needs_gpu=True,
        install_hint=(
            "Download a checkpoint first:\n"
            "        python scripts/download_models.py sapiens\n"
            "    then implement estimate_native() in "
            "src/soccer_mocap/pose/backends/sapiens_backend.py"
        ),
        homepage="https://github.com/facebookresearch/sapiens",
        implemented=False,
        extras={
            "checkpoints": ["0.3b", "0.6b", "1b", "2b"],
            "input_size": (1024, 768),
            "native_keypoints": 308,
            "provides": ["pose", "depth", "normals", "segmentation"],
        },
    )
