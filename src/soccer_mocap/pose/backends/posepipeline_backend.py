"""PosePipeline backend - planned integration.

PosePipeline (Cotton et al., Shirley Ryan AbilityLab) is a DataJoint-backed
markerless pipeline built for clinical and biomechanical work rather than for
demos. Its value here is not a single pose model - it wraps several - but the
provenance: every intermediate result is a database row tied to the exact
algorithm and parameters that produced it, which is what makes a study
reproducible and a result defensible.

It is also the heaviest thing on this list to stand up, because it wants a
MySQL server and a populated DataJoint schema before it will do anything.
Treat it as the destination once the analysis is settled, not the tool you
iterate with.

Implementing this means:

1. Stand up DataJoint (``pip install datajoint``) against a MySQL instance,
   local Docker is fine, and configure ``dj.config`` credentials.
2. Insert the clip into the ``Video`` table and populate ``TrackingBbox``,
   ``PersonBbox``, and then a ``TopDownPerson`` / ``LiftingPerson`` method of
   choice.
3. Read the resulting keypoints back out and yield them frame by frame. Note
   that this inverts the usual control flow - PosePipeline computes the whole
   clip and stores it, so this backend is best implemented as a bulk
   ``estimate_sequence`` that fills a cache which ``estimate_native`` then
   reads per frame.
4. Its top-down methods emit COCO-17 or Halpe-26 depending on the algorithm,
   so read the layout from the populated table rather than hard-coding it.

Because of the inverted control flow, this backend will likely want a small
extension to the :class:`~soccer_mocap.pose.base.PoseBackend` interface - an
optional ``precompute(video_path)`` hook - rather than being forced through
the per-frame API.
"""

from __future__ import annotations

from ...skeleton import COCO17
from ..base import BackendInfo, StubBackend
from ..registry import register


@register
class PosePipelineBackend(StubBackend):
    """Placeholder for the PosePipeline / DataJoint integration."""

    info = BackendInfo(
        key="posepipeline",
        label="PosePipeline (DataJoint)",
        layout=COCO17,
        summary=(
            "Research pipeline with full provenance tracking and several "
            "interchangeable pose algorithms. Needs a MySQL/DataJoint server."
        ),
        speed="slow",
        multi_person=True,
        needs_gpu=True,
        install_hint=(
            "mamba run -n soccer-mocap pip install datajoint\n"
            "    plus a MySQL server, then implement the precompute path in\n"
            "    src/soccer_mocap/pose/backends/posepipeline_backend.py"
        ),
        homepage="https://github.com/peabody124/PosePipeline",
        implemented=False,
        extras={
            "requires": ["datajoint", "mysql"],
            "control_flow": "bulk-precompute",
            "note": "Reproducibility-focused; wraps OpenPose, MMPose, HRNet and others.",
        },
    )
