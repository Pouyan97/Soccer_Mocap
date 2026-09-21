"""Phone-friendly demo app.

A Gradio web app, so it runs on any phone without installing anything: start
it on a laptop, open the URL in Safari or Chrome, and record or upload a clip
from the phone's own camera. That is the only cross-platform,
Python-only way to reach both iOS and Android, and it is why Gradio is here
rather than a native app.

Four tabs:

``Analyse``
    Upload or record a clip, choose the phone it came from, run the pipeline,
    and get an overlay video plus a report.
``Goalkeeper``
    Mark the two goal posts by tapping them. That is what turns pixel
    measurements into metres, and two taps beats a goal detector that is
    confidently wrong.
``Capture guide``
    Device-specific advice, shown before recording rather than after.
``Backends``
    What pose systems are installed, what is planned, and how to enable each.

Run it with::

    python apps/demo/app.py
    python apps/demo/app.py --share      # temporary public link for a phone
    soccer-mocap demo
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

# Allow `python apps/demo/app.py` from a source checkout without installing.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from soccer_mocap.config import PipelineConfig  # noqa: E402
from soccer_mocap.devices.presets import (  # noqa: E402
    DEVICE_PRESETS,
    PROFILES,
    capture_advice,
    get_preset,
    infer_preset,
)
from soccer_mocap.io.video import VideoReader, probe  # noqa: E402
from soccer_mocap.pipeline import Pipeline  # noqa: E402
from soccer_mocap.pose.registry import availability, describe  # noqa: E402

# Phone screens are narrow, so everything is one column with large targets.
MOBILE_CSS = """
.gradio-container { max-width: 900px !important; margin: 0 auto !important; }
#run-button { min-height: 56px; font-size: 1.05rem; font-weight: 600; }
#post-image img { touch-action: manipulation; }
.report-box textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                       font-size: 0.82rem; line-height: 1.45; }
@media (max-width: 640px) {
  .gradio-container { padding: 0 8px !important; }
  #run-button { min-height: 64px; }
}
"""

#: Longest edge of the goal-marking preview, in pixels.
PREVIEW_LONG_EDGE = 1280

DEVICE_CHOICES = [(preset.label, key) for key, preset in DEVICE_PRESETS.items()]
PROFILE_CHOICES = [(f"{p.name} - {p.description}", p.name) for p in PROFILES.values()]


# -- helpers ---------------------------------------------------------------


def _first_frame(video_path: str) -> tuple[np.ndarray | None, float]:
    """Grab one frame for goal marking, plus the factor it was scaled by.

    The preview is downscaled so a 4K frame is not shipped to a phone, which
    means a tap on it is in preview pixels while every keypoint the pipeline
    produces is in original-video pixels. The scale is returned so the two
    can be reconciled; without it a marked goal is silently wrong by the
    downscale factor on any clip above 1280 px.
    """
    try:
        with VideoReader(video_path, target_long_edge=PREVIEW_LONG_EDGE) as reader:
            for frame in reader:
                return frame.image, frame.scale
    except Exception:  # noqa: BLE001 - the UI reports it, no need to crash
        return None, 1.0
    return None, 1.0


def _describe_clip(video_path: str) -> tuple[str, str]:
    """Return ``(metadata_markdown, advice_markdown)`` for an uploaded clip."""
    meta = probe(video_path)
    preset = infer_preset(meta)
    advice = capture_advice(preset, meta)

    width, height = meta.display_size
    lines = [
        f"**{width} x {height}**"
        f" ({'portrait' if meta.is_portrait else 'landscape'})"
        f" &nbsp;|&nbsp; **{meta.fps:.1f} fps**"
        f"{' (slow motion)' if meta.is_high_speed else ''}"
        f" &nbsp;|&nbsp; **{meta.duration_s:.1f} s**" if meta.duration_s else "",
        f"Codec `{meta.codec}`, rotation flag {meta.rotation} deg.",
        f"Looks like: **{preset.label}**",
    ]

    advice_lines = []
    if advice.warnings:
        advice_lines.append("**Before you rely on these numbers**")
        advice_lines.extend(f"- {w}" for w in advice.warnings)
    return "\n\n".join(x for x in lines if x), "\n".join(advice_lines)


def _draw_markers(image: np.ndarray, points: list[list[float]]) -> np.ndarray:
    """Draw the tapped post positions onto the preview frame."""
    if image is None:
        return image
    marked = image.copy()
    try:
        import cv2
    except ImportError:
        return marked

    for index, (x, y) in enumerate(points):
        centre = (int(x), int(y))
        cv2.circle(marked, centre, 10, (255, 60, 60), -1, lineType=cv2.LINE_AA)
        cv2.circle(marked, centre, 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            marked, f"{index + 1}", (centre[0] + 14, centre[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
    if len(points) == 2:
        cv2.line(
            marked,
            (int(points[0][0]), int(points[0][1])),
            (int(points[1][0]), int(points[1][1])),
            (255, 60, 60), 3, lineType=cv2.LINE_AA,
        )
    return marked


# -- event handlers --------------------------------------------------------


def on_video_change(video_path: str | None):
    """Populate the metadata, advice, and the goal-marking frame."""
    if not video_path:
        return (
            gr.update(value=""),
            gr.update(value="", visible=False),
            gr.update(value=None),
            [],
            gr.update(value="No goal marked. Distances will be in body heights."),
            1.0,
        )

    try:
        info, advice = _describe_clip(video_path)
    except Exception as exc:  # noqa: BLE001
        return (
            gr.update(value=f"Could not read this video: {exc}"),
            gr.update(value="", visible=False),
            gr.update(value=None),
            [],
            gr.update(value="No goal marked."),
            1.0,
        )

    frame, preview_scale = _first_frame(video_path)
    return (
        gr.update(value=info),
        gr.update(value=advice, visible=bool(advice)),
        gr.update(value=frame),
        [],
        gr.update(value="No goal marked. Tap the foot of each post."),
        preview_scale,
    )


def on_frame_click(points: list, image: np.ndarray, evt: gr.SelectData):
    """Record a tapped goal post. Two taps define the goal mouth."""
    points = list(points or [])
    x, y = evt.index[0], evt.index[1]

    # A third tap starts over, which is the least surprising way to correct
    # a mistake on a touchscreen.
    if len(points) >= 2:
        points = []
    points.append([float(x), float(y)])

    if len(points) == 1:
        status = f"First post at ({x}, {y}). Now tap the other post."
    else:
        span = abs(points[1][0] - points[0][0])
        status = (
            f"Goal marked: {span:.0f} px across = 7.32 m, "
            f"so {span / 7.32:.1f} px/m. Tap again to redo."
        )
    return points, _draw_markers(image, points), status


def clear_points(image: np.ndarray):
    return [], image, "Cleared. Tap the foot of each post."


def run_analysis(
    video_path: str | None,
    device_key: str,
    profile_name: str,
    goalkeeper_enabled: bool,
    goal_points: list,
    preview_scale: float,
    dive_threshold: float,
    make_overlay: bool,
    progress=gr.Progress(),  # noqa: B008 - this is Gradio's documented idiom
):
    """Run the pipeline and format everything the UI shows."""
    if not video_path:
        return None, "Upload or record a clip first.", None, gr.update(visible=False)

    config = PipelineConfig()
    config.video.device_preset = device_key
    config.video.profile = profile_name
    config.output.overlay_video = make_overlay
    config.output.report = False
    config.output.formats = ["json"]

    if goalkeeper_enabled:
        config.goalkeeper.enabled = True
        config.goalkeeper.dive_speed_threshold = float(dive_threshold)
        if goal_points and len(goal_points) == 2:
            # Taps are in preview pixels; the pipeline works in original
            # video pixels, so undo the preview downscale.
            scale = preview_scale or 1.0
            config.goalkeeper.goal_posts = [
                [p[0] / scale, p[1] / scale] for p in goal_points
            ]

    def report_progress(fraction: float, message: str) -> None:
        progress(min(max(fraction, 0.0), 1.0), desc=message)

    try:
        result = Pipeline(config).run(video_path, progress=report_progress)
    except Exception as exc:  # noqa: BLE001 - shown in the UI, not the console
        detail = traceback.format_exc(limit=3)
        return (
            None,
            f"Analysis failed: {exc}\n\n{detail}",
            None,
            gr.update(visible=False),
        )

    overlay_path = result.outputs.get("overlay")
    report_text = "\n".join(result.summary_lines())

    dive_rows = None
    if result.goalkeeper and result.goalkeeper.dives:
        dive_rows = [
            [
                row["#"], row["start_s"], row["duration_s"], row["direction"],
                row["height"], row["peak_speed_bh_s"],
                row["lateral_m"] if row["lateral_m"] is not None else "-",
                row["confidence"],
            ]
            for row in result.goalkeeper.dive_table()
        ]

    return (
        str(overlay_path) if overlay_path else None,
        report_text,
        dive_rows,
        gr.update(visible=dive_rows is not None),
    )


def render_capture_guide(device_key: str) -> str:
    """Markdown for the capture-guide tab."""
    preset = get_preset(device_key)
    advice = capture_advice(preset)
    rates = ", ".join(f"{f:g}" for f in preset.common_fps)

    lines = [
        f"### {preset.label}",
        "",
        f"- Typical recording: **{preset.typical_resolution[0]}x"
        f"{preset.typical_resolution[1]}**, {rates} fps, `{preset.codec}`",
        f"- Field of view: about {preset.fov_deg:.0f} degrees",
        f"- Rolling shutter: **{preset.rolling_shutter}**",
        f"- Suggested profile: **{preset.recommended_profile}**",
    ]
    if preset.notes:
        lines += ["", f"> {preset.notes}"]
    if advice.warnings:
        lines += ["", "#### Watch out for", *[f"- {w}" for w in advice.warnings]]
    lines += ["", "#### How to film", *[f"- {t}" for t in advice.tips]]
    return "\n".join(lines)


def render_backends() -> str:
    """Markdown table of backend availability."""
    checks = availability()
    rows = [
        "| Backend | Status | Speed | GPU | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for info in sorted(describe(), key=lambda i: (not i.implemented, i.key)):
        usable, reason = checks[info.key]
        status = "**ready**" if usable else ("planned" if not info.implemented else "missing deps")
        note = info.summary if usable else f"{reason}"
        rows.append(
            f"| `{info.key}` | {status} | {info.speed} | "
            f"{'yes' if info.needs_gpu else 'no'} | {note} |"
        )

    hints = ["", "#### Enabling a backend", ""]
    for info in describe():
        if not checks[info.key][0] and info.install_hint:
            hints.append(f"**{info.label}**")
            hints.append("```")
            hints.extend(line.strip() for line in info.install_hint.splitlines())
            hints.append("```")
    return "\n".join(rows + hints)


# -- interface -------------------------------------------------------------


def build_interface() -> gr.Blocks:
    """Assemble the Gradio app."""
    with gr.Blocks(title="Soccer Mocap", fill_width=False) as demo:
        gr.Markdown(
            "# Soccer Mocap\n"
            "Single-camera markerless analysis. Record on any phone, open this "
            "page in its browser, and upload the clip."
        )

        goal_points = gr.State([])
        preview_scale = gr.State(1.0)

        with gr.Tabs():
            # ---------------------------------------------------------- analyse
            with gr.Tab("Analyse"):
                video_input = gr.Video(
                    label="Clip",
                    sources=["upload", "webcam"],
                    height=320,
                    include_audio=False,
                )
                clip_info = gr.Markdown("")
                clip_warnings = gr.Markdown("", visible=False)

                with gr.Accordion("Settings", open=False):
                    device_dropdown = gr.Dropdown(
                        choices=DEVICE_CHOICES,
                        value="generic",
                        label="Phone",
                        info="Sets the expected frame rate, codec and lens behaviour.",
                    )
                    profile_dropdown = gr.Dropdown(
                        choices=PROFILE_CHOICES,
                        value="balanced",
                        label="Quality",
                        info="Lower is faster. 'fast' is enough to check framing.",
                    )
                    overlay_checkbox = gr.Checkbox(
                        value=True,
                        label="Render an overlay video",
                        info="Slower, but it is the quickest way to spot a bad result.",
                    )

                goalkeeper_checkbox = gr.Checkbox(
                    value=True, label="Goalkeeper analysis"
                )
                with gr.Accordion("Goalkeeper settings", open=False):
                    dive_slider = gr.Slider(
                        minimum=0.4, maximum=3.0, value=1.2, step=0.1,
                        label="Dive sensitivity (body heights / second)",
                        info="Lower detects more, including quick sidesteps.",
                    )
                    goal_status_analyse = gr.Markdown(
                        "No goal marked. Distances will be in body heights."
                    )

                run_button = gr.Button(
                    "Analyse", variant="primary", elem_id="run-button"
                )

                overlay_output = gr.Video(
                    label="Overlay", height=320, autoplay=True, loop=True
                )
                report_output = gr.Textbox(
                    label="Report", lines=22, max_lines=40,
                    elem_classes=["report-box"],
                )
                dive_table = gr.Dataframe(
                    headers=["#", "start s", "dur s", "dir", "height",
                             "peak bh/s", "lateral m", "conf"],
                    label="Dives",
                    wrap=True,
                    visible=False,
                )

            # ------------------------------------------------------ goalkeeper
            with gr.Tab("Goalkeeper"):
                gr.Markdown(
                    "### Mark the goal\n"
                    "Tap the **foot of each post** where it meets the grass. "
                    "A goal is 7.32 m wide by law, so those two taps are what "
                    "turn pixels into metres - distances off the line, dive "
                    "length, and where the keeper stood across the mouth.\n\n"
                    "Without them everything still works, but in *body heights* "
                    "instead of metres."
                )
                post_image = gr.Image(
                    label="Tap the two posts",
                    type="numpy",
                    interactive=False,
                    height=380,
                    elem_id="post-image",
                )
                goal_status = gr.Markdown("Upload a clip on the Analyse tab first.")
                clear_button = gr.Button("Clear marks")

            # ---------------------------------------------------- capture guide
            with gr.Tab("Capture guide"):
                guide_dropdown = gr.Dropdown(
                    choices=DEVICE_CHOICES, value="iphone_recent", label="Phone"
                )
                guide_markdown = gr.Markdown(render_capture_guide("iphone_recent"))

            # --------------------------------------------------------- backends
            with gr.Tab("Backends"):
                gr.Markdown(
                    "### Pose backends\n"
                    "The pose stage is swappable. MediaPipe runs out of the box; "
                    "the others are registered with their integration notes."
                )
                gr.Markdown(render_backends())

        # -- wiring --
        video_input.change(
            on_video_change,
            inputs=[video_input],
            outputs=[
                clip_info, clip_warnings, post_image, goal_points,
                goal_status, preview_scale,
            ],
        )
        post_image.select(
            on_frame_click,
            inputs=[goal_points, post_image],
            outputs=[goal_points, post_image, goal_status],
        )
        # Keep the Analyse tab's copy of the status in step with the marking tab.
        goal_status.change(
            lambda text: gr.update(value=text),
            inputs=[goal_status],
            outputs=[goal_status_analyse],
        )
        clear_button.click(
            clear_points,
            inputs=[post_image],
            outputs=[goal_points, post_image, goal_status],
        )
        guide_dropdown.change(
            render_capture_guide, inputs=[guide_dropdown], outputs=[guide_markdown]
        )
        run_button.click(
            run_analysis,
            inputs=[
                video_input,
                device_dropdown,
                profile_dropdown,
                goalkeeper_checkbox,
                goal_points,
                preview_scale,
                dive_slider,
                overlay_checkbox,
            ],
            outputs=[overlay_output, report_output, dive_table, dive_table],
        )

    return demo


def launch(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = False,
) -> None:
    """Start the app.

    ``host="0.0.0.0"`` is the default on purpose: a phone on the same Wi-Fi
    reaches the laptop by its LAN address, which is the usual way this gets
    used. ``share=True`` gives a public tunnel for phones on mobile data.
    """
    demo = build_interface()
    launch_kwargs: dict[str, Any] = {
        "server_name": host,
        "server_port": port,
        "share": share,
        "inbrowser": inbrowser,
        "show_error": True,
    }

    import inspect

    # Gradio 6 moved theme and css from the Blocks constructor to launch().
    # Both spellings are probed so the app is not pinned to one release.
    launch_params = inspect.signature(demo.launch).parameters
    if "theme" in launch_params:
        launch_kwargs["theme"] = gr.themes.Soft()
    if "css" in launch_params:
        launch_kwargs["css"] = MOBILE_CSS
    # Installable-web-app mode, so a phone can add it to the home screen.
    if "pwa" in launch_params:
        launch_kwargs["pwa"] = True

    demo.launch(**launch_kwargs)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Soccer Mocap demo app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share", action="store_true", help="Create a temporary public link"
    )
    parser.add_argument("--open", action="store_true", help="Open a local browser")
    args = parser.parse_args(argv)

    launch(host=args.host, port=args.port, share=args.share, inbrowser=args.open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
