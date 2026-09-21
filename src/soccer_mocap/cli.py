"""Command line interface.

    soccer-mocap backends                    # what is installed, what is not
    soccer-mocap devices                     # phone presets and capture advice
    soccer-mocap probe clip.mov              # metadata and warnings, no inference
    soccer-mocap run clip.mov --goalkeeper --goal-posts 410,520 980,528
    soccer-mocap demo                        # launch the phone web app

Run with no arguments to see the same list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_point(text: str) -> list[float]:
    """Parse ``"410,520"`` into ``[410.0, 520.0]``."""
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected a point as x,y (for example 410,520), got {text!r}"
        )
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Point coordinates must be numbers: {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soccer-mocap",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="Print the version and exit")
    subparsers = parser.add_subparsers(dest="command")

    # -- run --
    run = subparsers.add_parser("run", help="Process a video")
    run.add_argument("video", type=Path, help="Video file to analyse")
    run.add_argument("-c", "--config", type=Path, help="YAML config file")
    run.add_argument("-o", "--output", type=Path, help="Output directory")
    run.add_argument("--backend", help="Pose backend (default: from config, or auto)")
    run.add_argument(
        "--device",
        dest="device_preset",
        help="Phone preset key, or 'auto' to infer it from the file",
    )
    run.add_argument("--profile", choices=["fast", "balanced", "accurate"])
    run.add_argument("--start", type=float, help="Trim start, seconds")
    run.add_argument("--end", type=float, help="Trim end, seconds")
    run.add_argument(
        "--formats",
        nargs="+",
        choices=["json", "csv", "parquet", "trc"],
        help="Export formats (default: json)",
    )
    run.add_argument("--no-overlay", action="store_true", help="Skip the overlay video")

    goalkeeper = run.add_argument_group("goalkeeper")
    goalkeeper.add_argument(
        "--goalkeeper", action="store_true", help="Run the goalkeeper analysis"
    )
    goalkeeper.add_argument(
        "--goal-posts",
        nargs=2,
        type=_parse_point,
        metavar="X,Y",
        help="Foot of each post, e.g. --goal-posts 410,520 980,528",
    )
    goalkeeper.add_argument(
        "--gk-track", type=int, help="Force a track id instead of identifying one"
    )
    goalkeeper.add_argument(
        "--dive-threshold",
        type=float,
        help="Dive detection speed threshold, in body heights per second",
    )

    # -- other commands --
    subparsers.add_parser("backends", help="List pose backends and their availability")
    subparsers.add_parser("devices", help="List phone presets and capture advice")

    probe_cmd = subparsers.add_parser("probe", help="Show a clip's metadata and warnings")
    probe_cmd.add_argument("video", type=Path)

    demo = subparsers.add_parser("demo", help="Launch the phone-friendly web app")
    demo.add_argument("--port", type=int, default=7860)
    demo.add_argument("--host", default="0.0.0.0")
    demo.add_argument(
        "--share", action="store_true", help="Create a temporary public link"
    )

    return parser


# -- commands -------------------------------------------------------------


def cmd_backends() -> int:
    from .pose.registry import availability, describe

    checks = availability()
    print(f"{'BACKEND':<14} {'STATUS':<12} {'SPEED':<9} {'GPU':<5} DESCRIPTION")
    print("-" * 100)
    for info in sorted(describe(), key=lambda i: (not i.implemented, i.key)):
        usable, reason = checks[info.key]
        status = "ready" if usable else ("planned" if not info.implemented else "missing deps")
        print(
            f"{info.key:<14} {status:<12} {info.speed:<9} "
            f"{'yes' if info.needs_gpu else 'no':<5} {info.summary}"
        )
        if not usable:
            print(f"{'':<14} -> {reason}")
            if info.install_hint:
                for line in info.install_hint.splitlines():
                    print(f"{'':<17} {line.strip()}")
    return 0


def cmd_devices() -> int:
    from .devices.presets import DEVICE_PRESETS, PROFILES, capture_advice

    print("PHONE PRESETS")
    print("-" * 100)
    for preset in DEVICE_PRESETS.values():
        rates = "/".join(f"{f:g}" for f in preset.common_fps)
        print(f"  {preset.key:<18} {preset.label}")
        print(
            f"  {'':<18} {preset.typical_resolution[0]}x{preset.typical_resolution[1]}, "
            f"{rates} fps, {preset.codec}, {preset.fov_deg:.0f} deg FOV, "
            f"rolling shutter {preset.rolling_shutter}"
        )
        if preset.notes:
            print(f"  {'':<18} {preset.notes}")
        print()

    print("PROCESSING PROFILES")
    print("-" * 100)
    for profile in PROFILES.values():
        edge = profile.target_long_edge or "native"
        print(
            f"  {profile.name:<10} long edge {edge}, stride {profile.frame_stride}, "
            f"complexity {profile.model_complexity} - {profile.description}"
        )

    print()
    print("GENERAL CAPTURE ADVICE")
    print("-" * 100)
    for tip in capture_advice(DEVICE_PRESETS["generic"]).tips:
        print(f"  - {tip}")
    return 0


def cmd_probe(video: Path) -> int:
    from .devices.presets import capture_advice, infer_preset, resolve_profile
    from .io.video import probe

    meta = probe(video)
    preset = infer_preset(meta)
    profile = resolve_profile(meta, preset)

    print(f"File        : {meta.path}")
    print(f"Stored size : {meta.width}x{meta.height}")
    print(f"Display size: {meta.display_size[0]}x{meta.display_size[1]}"
          f" ({'portrait' if meta.is_portrait else 'landscape'})")
    print(f"Rotation    : {meta.rotation} deg")
    print(f"Frame rate  : {meta.fps:.3f} fps{'  (slow motion)' if meta.is_high_speed else ''}")
    print(f"Frames      : {meta.frame_count}")
    print(f"Duration    : {meta.duration_s:.2f} s" if meta.duration_s else "Duration    : unknown")
    print(f"Codec       : {meta.codec}")
    print()
    print(f"Inferred phone  : {preset.label} ({preset.key})")
    print(f"Chosen profile  : {profile.name} "
          f"(long edge {profile.target_long_edge or 'native'}, stride {profile.frame_stride})")

    advice = capture_advice(preset, meta)
    if advice.warnings:
        print()
        print("Warnings")
        for warning in advice.warnings:
            print(f"  ! {warning}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .config import PipelineConfig, load_config
    from .pipeline import Pipeline

    config = load_config(args.config) if args.config else PipelineConfig()

    if args.backend:
        config.pose.backend = args.backend
    if args.device_preset:
        config.video.device_preset = args.device_preset
    if args.profile:
        config.video.profile = args.profile
    if args.start is not None:
        config.video.start_s = args.start
    if args.end is not None:
        config.video.end_s = args.end
    if args.output:
        config.output.directory = str(args.output)
    if args.formats:
        config.output.formats = list(args.formats)
    if args.no_overlay:
        config.output.overlay_video = False

    if args.goalkeeper or args.goal_posts or args.gk_track is not None:
        config.goalkeeper.enabled = True
    if args.goal_posts:
        config.goalkeeper.goal_posts = list(args.goal_posts)
    if args.gk_track is not None:
        config.goalkeeper.track_id = args.gk_track
        config.goalkeeper.identification = "manual"
    if args.dive_threshold is not None:
        config.goalkeeper.dive_speed_threshold = args.dive_threshold

    last_message = ""

    def progress(fraction: float, message: str) -> None:
        nonlocal last_message
        if message != last_message:
            print(f"[{fraction * 100:5.1f}%] {message}")
            last_message = message

    result = Pipeline(config).run(args.video, progress=progress)

    print()
    print(result)

    if result.outputs:
        print()
        print("Outputs")
        for name, path in result.outputs.items():
            print(f"  {name:<8} {path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    try:
        from apps.demo.app import launch
    except ImportError:
        # Running from an installed package rather than the repo checkout.
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root))
        try:
            from apps.demo.app import launch
        except ImportError as exc:
            print(
                f"Could not import the demo app: {exc}\n"
                "Run it directly from the repository instead:\n"
                "    python apps/demo/app.py",
                file=sys.stderr,
            )
            return 1

    launch(host=args.host, port=args.port, share=args.share)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"soccer-mocap {__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    from .exceptions import SoccerMocapError

    try:
        if args.command == "backends":
            return cmd_backends()
        if args.command == "devices":
            return cmd_devices()
        if args.command == "probe":
            return cmd_probe(args.video)
        if args.command == "run":
            return cmd_run(args)
        if args.command == "demo":
            return cmd_demo(args)
    except SoccerMocapError as exc:
        # Our own errors carry actionable messages; a traceback would only
        # bury them.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
