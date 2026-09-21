"""Exporting results.

Four formats, each for a different consumer:

``json``
    The full result, including metadata, warnings, and the goalkeeper
    report. This is the archival format - it is the only one that records
    *how* the numbers were produced, which is what makes a result
    reproducible six months later.
``csv``
    One row per frame per person, one column per joint coordinate. Wide, but
    it opens in a spreadsheet, which matters more than elegance for a coach.
``parquet``
    The same table, typed and compressed. Use it once clips get long enough
    that CSV parsing is annoying.
``trc``
    Motion-capture marker format, readable by OpenSim and Visual3D. Only
    meaningful once 3D is available; exporting 2D pixels to TRC produces a
    file those tools will happily load and silently misinterpret, so this
    writer refuses rather than allowing it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..skeleton import CANONICAL, SkeletonLayout
from ..types import PoseSequence


class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars, arrays and NaN the way JSON consumers expect."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            value = float(o)
            # NaN is not valid JSON; null is the honest equivalent.
            return None if np.isnan(value) else value
        if isinstance(o, np.bool_):
            return bool(o)
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        return super().default(o)


def _clean_nans(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None, for strict JSON."""
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nans(v) for v in obj]
    return obj


def write_json(
    path: str | Path,
    sequence: PoseSequence,
    goalkeeper_report: Any = None,
    config: Any = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the full result, including provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "video": asdict(sequence.video),
        "backend": sequence.backend,
        "skeleton": {
            "layout": sequence.layout.name,
            "joint_names": list(sequence.layout.joint_names),
            "edges": [list(e) for e in sequence.layout.edges],
        },
        "num_frames": len(sequence.frames),
        "meta": sequence.meta,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "people": [
                    {
                        "track_id": person.track_id,
                        "keypoints": person.xy.tolist(),
                        "confidence": person.confidence.tolist(),
                        "bbox": (
                            person.bbox.as_array().tolist() if person.bbox else None
                        ),
                        "keypoints_3d": (
                            person.xyz.tolist() if person.xyz is not None else None
                        ),
                    }
                    for person in frame.people
                ],
            }
            for frame in sequence.frames
        ],
    }

    if goalkeeper_report is not None:
        payload["goalkeeper"] = (
            goalkeeper_report.to_dict()
            if hasattr(goalkeeper_report, "to_dict")
            else goalkeeper_report
        )
    if config is not None:
        payload["config"] = config.to_dict() if hasattr(config, "to_dict") else config
    if extra:
        payload.update(extra)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(_clean_nans(payload), handle, cls=_NumpyEncoder, indent=2)
    return path


def build_dataframe(sequence: PoseSequence, layout: SkeletonLayout | None = None):
    """Long-to-wide table: one row per (frame, person).

    Returns a pandas DataFrame with ``<joint>_x``, ``<joint>_y`` and
    ``<joint>_conf`` columns.
    """
    import pandas as pd

    layout = layout or sequence.layout
    rows = []
    for frame in sequence.frames:
        for person in frame.people:
            row: dict[str, Any] = {
                "frame": frame.frame_index,
                "time_s": frame.timestamp,
                "track_id": person.track_id,
            }
            for index, name in enumerate(layout.joint_names):
                if index >= person.num_joints:
                    break
                row[f"{name}_x"] = person.xy[index, 0]
                row[f"{name}_y"] = person.xy[index, 1]
                row[f"{name}_conf"] = person.confidence[index]
            rows.append(row)
    return pd.DataFrame(rows)


def write_csv(path: str | Path, sequence: PoseSequence) -> Path:
    """Write the wide per-frame table as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_dataframe(sequence).to_csv(path, index=False)
    return path


def write_parquet(path: str | Path, sequence: PoseSequence) -> Path:
    """Write the wide per-frame table as Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_dataframe(sequence).to_parquet(path, index=False)
    return path


def write_trc(
    path: str | Path,
    sequence: PoseSequence,
    track_id: int | None = None,
    units: str = "m",
) -> Path:
    """Write 3D joint trajectories as an OpenSim/Visual3D TRC file.

    Raises:
        ValueError: when the sequence has no 3D. TRC is a 3D marker format;
            writing pixel coordinates into one produces a file that loads
            without complaint and means nothing, which is worse than an
            error.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tracks = sequence.tracks or sequence.build_tracks()
    if not tracks:
        raise ValueError("No tracks available to export.")

    if track_id is None:
        track = max(tracks.values(), key=len)
    elif track_id in tracks:
        track = tracks[track_id]
    else:
        raise ValueError(f"Track {track_id} not found. Available: {sorted(tracks)}")

    frames = [f for f in track.frame_indices if track.poses[f].xyz is not None]
    if not frames:
        raise ValueError(
            "This track has no 3D data, and TRC is a 3D format. Run a lifting "
            "stage first, or export to CSV/JSON instead."
        )

    layout = CANONICAL
    joint_names = list(layout.joint_names)
    num_joints = len(joint_names)
    fps = sequence.fps

    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
        "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{fps:.2f}\t{fps:.2f}\t{len(frames)}\t{num_joints}\t{units}\t"
        f"{fps:.2f}\t1\t{len(frames)}",
        "Frame#\tTime\t" + "\t\t\t".join(joint_names) + "\t\t",
        "\t\t" + "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(num_joints)),
        "",
    ]

    for row, frame_index in enumerate(frames, start=1):
        pose = track.poses[frame_index]
        xyz = pose.xyz
        values = [f"{row}", f"{frame_index / fps:.5f}"]
        for joint in range(num_joints):
            point = xyz[joint] if joint < len(xyz) else [np.nan] * 3
            values.extend("" if not np.isfinite(v) else f"{v:.6f}" for v in point)
        lines.append("\t".join(values))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_report(path: str | Path, lines: list[str]) -> Path:
    """Write a plain-text report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


#: Format name -> writer. Used by the pipeline to honour `output.formats`.
WRITERS = {
    "json": write_json,
    "csv": write_csv,
    "parquet": write_parquet,
    "trc": write_trc,
}
