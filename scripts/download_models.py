"""Fetch model checkpoints into ``models/``.

MediaPipe 1.0 removed the old ``mp.solutions`` API that bundled its weights
inside the wheel, so a ``.task`` file is now a hard requirement rather than a
convenience. This script gets it, plus the optional detector and the pointers
for the heavier backends.

Usage:
    python scripts/download_models.py                # default set
    python scripts/download_models.py mediapipe      # one group
    python scripts/download_models.py --list
    python scripts/download_models.py mediapipe --variant heavy
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

MEDIAPIPE_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable file."""

    group: str
    variant: str
    filename: str
    url: str
    description: str
    approx_mb: float


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        group="mediapipe",
        variant="lite",
        filename="pose_landmarker_lite.task",
        url=f"{MEDIAPIPE_BASE}/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        description="Fastest pose landmarker. Fine for framing checks.",
        approx_mb=5.5,
    ),
    ModelSpec(
        group="mediapipe",
        variant="full",
        filename="pose_landmarker_full.task",
        url=f"{MEDIAPIPE_BASE}/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        description="Default. Good speed/accuracy balance on CPU.",
        approx_mb=9.0,
    ),
    ModelSpec(
        group="mediapipe",
        variant="heavy",
        filename="pose_landmarker_heavy.task",
        url=f"{MEDIAPIPE_BASE}/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        description="Most accurate MediaPipe variant. Noticeably slower.",
        approx_mb=29.0,
    ),
)

#: Backends whose weights are too large or too license-encumbered to pull
#: automatically. Printed as instructions instead.
MANUAL_NOTES: dict[str, str] = {
    "sapiens": (
        "Sapiens checkpoints live on HuggingFace under facebook/sapiens-pose-*.\n"
        "  The torchscript exports avoid needing the training repo:\n"
        "      huggingface-cli download facebook/sapiens-pose-0.6b-torchscript \\\n"
        "          --local-dir models/sapiens\n"
        "  Expect 1-8 GB depending on the variant, and a GPU to run it."
    ),
    "rtmpose": (
        "rtmlib downloads its ONNX weights on first use and caches them in\n"
        "  ~/.cache/rtmlib, so there is nothing to fetch here. Just install it:\n"
        "      mamba run -n soccer-mocap pip install rtmlib onnxruntime-gpu"
    ),
    "posepipeline": (
        "PosePipeline pulls weights per algorithm through its own DataJoint\n"
        "  tables. Follow the upstream setup at\n"
        "  https://github.com/peabody124/PosePipeline"
    ),
    "yolo": (
        "The person detector downloads itself on first use via ultralytics.\n"
        "  To pre-fetch it:\n"
        "      mamba run -n soccer-mocap python -c \\\n"
        "          \"from ultralytics import YOLO; YOLO('yolo11n.pt')\""
    ),
}

DEFAULT_GROUPS = ("mediapipe",)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def download(spec: ModelSpec, dest_dir: Path, force: bool = False) -> Path:
    """Download one model, skipping it when already present."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / spec.filename

    if target.exists() and not force:
        print(
            f"  [skip]  {spec.filename} already present "
            f"({human_size(target.stat().st_size)})"
        )
        return target

    # Download to a temporary name so an interrupted transfer never leaves a
    # truncated file that later looks valid to the skip check above.
    partial = target.with_suffix(target.suffix + ".part")
    print(f"  [get ]  {spec.filename}  (~{spec.approx_mb:.0f} MB)")

    # A carriage-return progress bar is unreadable once the output is piped to
    # a file or a log, so only draw it on a real terminal.
    interactive = sys.stdout.isatty()

    def report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0 or not interactive:
            return
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        bar = "#" * int(pct // 4)
        sys.stdout.write(f"\r          {bar:<25} {pct:5.1f}%  {human_size(done)}")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(spec.url, partial, reporthook=report)
        if interactive:
            sys.stdout.write("\n")
        partial.replace(target)
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"\nDownload failed for {spec.filename}: {exc}\n"
            f"URL: {spec.url}\n"
            "Check the network connection, or fetch the file manually into models/."
        ) from exc

    digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
    print(f"          saved to {target.relative_to(REPO_ROOT)}  sha256:{digest}...")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "groups",
        nargs="*",
        default=list(DEFAULT_GROUPS),
        help="Model groups to fetch. Default: mediapipe",
    )
    parser.add_argument(
        "--variant",
        default="full",
        choices=sorted({m.variant for m in MODELS}),
        help="Which MediaPipe variant to fetch (default: full)",
    )
    parser.add_argument("--all-variants", action="store_true", help="Fetch every variant")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--list", action="store_true", help="Show what is available and exit")
    parser.add_argument("--dir", type=Path, default=MODELS_DIR, help="Destination directory")
    args = parser.parse_args(argv)

    if args.list:
        print("Downloadable:")
        for spec in MODELS:
            print(
                f"  {spec.group:12s} {spec.variant:6s} "
                f"{spec.approx_mb:6.1f} MB  {spec.description}"
            )
        print("\nManual setup required:")
        for group, note in MANUAL_NOTES.items():
            print(f"  {group}:\n    {note}\n")
        return 0

    requested = [g.lower() for g in args.groups]
    fetched_any = False

    for group in requested:
        if group in MANUAL_NOTES and group not in {m.group for m in MODELS}:
            print(f"\n{group}: manual setup")
            print(f"  {MANUAL_NOTES[group]}")
            continue

        specs = [m for m in MODELS if m.group == group]
        if not specs:
            print(f"\nUnknown group {group!r}. Use --list to see the options.")
            continue
        if not args.all_variants:
            specs = [m for m in specs if m.variant == args.variant] or specs[:1]

        print(f"\n{group}:")
        for spec in specs:
            download(spec, args.dir, force=args.force)
            fetched_any = True

    if fetched_any:
        print(f"\nDone. Models are in {args.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
