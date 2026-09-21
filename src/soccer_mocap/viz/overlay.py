"""Drawing skeletons and annotations onto frames.

The overlay is the main debugging tool for this pipeline. Almost every bad
result - a swapped track, a mis-marked goal, a keeper who is actually a
defender - is obvious in one glance at an overlaid frame and invisible in
the numbers. It is worth rendering one even when you think you do not need
it.

Colours are chosen to stay legible against grass, which rules out most
greens, and to stay distinguishable in the compressed video the demo app
hands back to a phone.
"""

from __future__ import annotations

import numpy as np

from ..skeleton import CANONICAL, J, SkeletonLayout
from ..types import FramePoses, PersonPose

#: Per-track colours, RGB. Picked for contrast against a green pitch and for
#: distinguishability by the most common forms of colour vision deficiency.
TRACK_COLOURS: tuple[tuple[int, int, int], ...] = (
    (255, 87, 51),    # orange-red
    (51, 153, 255),   # blue
    (255, 215, 0),    # gold
    (204, 51, 255),   # magenta
    (0, 230, 195),    # teal
    (255, 255, 255),  # white
)

GOAL_COLOUR = (255, 60, 60)
TEXT_COLOUR = (255, 255, 255)
TEXT_BACKGROUND = (0, 0, 0)

#: Limbs drawn in a warmer shade, so left and right are separable at a glance.
LEFT_JOINTS = frozenset(
    {J.LEFT_SHOULDER, J.LEFT_ELBOW, J.LEFT_WRIST, J.LEFT_HIP, J.LEFT_KNEE, J.LEFT_ANKLE}
)


def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Drawing overlays needs OpenCV:\n"
            "    mamba run -n soccer-mocap pip install opencv-contrib-python"
        ) from exc


def track_colour(track_id: int | None) -> tuple[int, int, int]:
    """Stable colour for a track id."""
    if track_id is None:
        return (180, 180, 180)
    return TRACK_COLOURS[track_id % len(TRACK_COLOURS)]


def draw_pose(
    image: np.ndarray,
    pose: PersonPose,
    colour: tuple[int, int, int] | None = None,
    layout: SkeletonLayout = CANONICAL,
    min_confidence: float = 0.3,
    thickness: int = 2,
    radius: int = 3,
    draw_bbox: bool = False,
    label: str | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Draw one skeleton. Modifies ``image`` in place and returns it.

    ``scale`` converts keypoints into this image's pixel space. Keypoints are
    stored in original-video coordinates, so an overlay rendered on a
    downscaled inference frame must pass the same factor the frame was
    downscaled by, or the skeleton lands somewhere off to the side.
    """
    cv2 = _require_cv2()
    colour = colour or track_colour(pose.track_id)
    visible = pose.visible(min_confidence)
    xy = pose.xy * scale if scale != 1.0 else pose.xy

    for a, b in layout.edges:
        if a >= len(visible) or b >= len(visible) or not (visible[a] and visible[b]):
            continue
        # Tint limbs on the subject's left so the two sides are separable.
        edge_colour = colour if not (a in LEFT_JOINTS and b in LEFT_JOINTS) else tuple(
            min(255, int(c * 0.6 + 100)) for c in colour
        )
        cv2.line(
            image,
            tuple(np.round(xy[a]).astype(int)),
            tuple(np.round(xy[b]).astype(int)),
            edge_colour,
            thickness,
            lineType=cv2.LINE_AA,
        )

    for index in np.flatnonzero(visible):
        cv2.circle(
            image,
            tuple(np.round(xy[index]).astype(int)),
            radius,
            colour,
            -1,
            lineType=cv2.LINE_AA,
        )

    if draw_bbox and pose.bbox is not None:
        box = pose.bbox
        cv2.rectangle(
            image,
            (int(box.x1 * scale), int(box.y1 * scale)),
            (int(box.x2 * scale), int(box.y2 * scale)),
            colour,
            1,
        )

    if label and pose.bbox is not None:
        draw_label(
            image,
            label,
            (int(pose.bbox.x1 * scale), int(pose.bbox.y1 * scale) - 6),
            colour,
        )

    return image


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int] = TEXT_COLOUR,
    scale: float = 0.5,
) -> np.ndarray:
    """Draw text with a filled backing box, so it stays readable over grass."""
    cv2 = _require_cv2()
    thickness = 1
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = origin
    y = max(y, height + 4)

    cv2.rectangle(
        image,
        (x - 2, y - height - 3),
        (x + width + 2, y + baseline - 1),
        TEXT_BACKGROUND,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        lineType=cv2.LINE_AA,
    )
    return image


def draw_frame(
    image: np.ndarray,
    frame: FramePoses,
    layout: SkeletonLayout = CANONICAL,
    min_confidence: float = 0.3,
    highlight_track: int | None = None,
    show_ids: bool = True,
    scale: float = 1.0,
) -> np.ndarray:
    """Draw every person in a frame.

    ``highlight_track`` dims everyone else, which is how the goalkeeper is
    picked out of a crowded box.
    """
    output = image.copy()
    for pose in frame.people:
        is_highlight = highlight_track is None or pose.track_id == highlight_track
        colour = track_colour(pose.track_id)
        if not is_highlight:
            colour = tuple(int(c * 0.35) for c in colour)

        label = None
        if show_ids and pose.track_id is not None:
            label = f"#{pose.track_id}"
            if pose.track_id == highlight_track:
                label += " GK"

        draw_pose(
            output,
            pose,
            colour=colour,
            layout=layout,
            min_confidence=min_confidence,
            thickness=2 if is_highlight else 1,
            label=label,
            scale=scale,
        )
    return output


def draw_goal(image: np.ndarray, goal, show_scale: bool = True, scale: float = 1.0) -> np.ndarray:
    """Draw the marked goal mouth and its derived scale."""
    cv2 = _require_cv2()
    left = tuple(np.round(goal.left_post * scale).astype(int))
    right = tuple(np.round(goal.right_post * scale).astype(int))

    cv2.line(image, left, right, GOAL_COLOUR, 2, lineType=cv2.LINE_AA)
    for post in (left, right):
        cv2.circle(image, post, 6, GOAL_COLOUR, -1, lineType=cv2.LINE_AA)

    if goal.crossbar_y is not None:
        bar_y = int(round(goal.crossbar_y * scale))
        cv2.line(image, (left[0], bar_y), (right[0], bar_y), GOAL_COLOUR, 1)
        cv2.line(image, left, (left[0], bar_y), GOAL_COLOUR, 1)
        cv2.line(image, right, (right[0], bar_y), GOAL_COLOUR, 1)

    if show_scale:
        centre = np.round(goal.center * scale).astype(int)
        draw_label(
            image,
            f"goal 7.32 m = {goal.width_px:.0f} px ({goal.px_per_m:.1f} px/m)",
            (int(centre[0]) - 90, int(centre[1]) + 22),
            GOAL_COLOUR,
        )
    return image


def draw_dive_banner(
    image: np.ndarray,
    dive,
    frame_index: int,
) -> np.ndarray:
    """Mark frames that fall inside a detected dive."""
    if not (dive.start_frame <= frame_index <= dive.end_frame):
        return image

    cv2 = _require_cv2()
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (255, 60, 60), 3)

    phase = _phase_at(dive, frame_index)
    draw_label(
        image,
        f"DIVE {dive.direction.value.upper()} / {dive.height.value} - {phase}",
        (12, 26),
        (255, 220, 220),
        scale=0.6,
    )
    return image


def _phase_at(dive, frame_index: int) -> str:
    """Name the dive phase a frame falls in."""
    phases = dive.phases
    ordered = [
        ("landing", phases.landing),
        ("flight", phases.flight_start),
        ("takeoff", phases.takeoff),
        ("preparatory", phases.preparatory_start),
        ("set", phases.set_frame),
    ]
    for name, boundary in ordered:
        if boundary is not None and frame_index >= boundary:
            return name
    return "dive"


def render_overlay_frames(
    frames: list[np.ndarray],
    poses: list[FramePoses],
    goal=None,
    dives: list | None = None,
    highlight_track: int | None = None,
    layout: SkeletonLayout = CANONICAL,
    min_confidence: float = 0.3,
    scale: float = 1.0,
) -> list[np.ndarray]:
    """Render a whole clip's overlay.

    ``frames`` and ``poses`` are matched by position, so they must come from
    the same strided pass over the video. ``scale`` is the factor those
    frames were downscaled by, since keypoints are stored at original size.
    """
    rendered: list[np.ndarray] = []
    for image, frame_poses in zip(frames, poses, strict=False):
        output = draw_frame(
            image,
            frame_poses,
            layout=layout,
            min_confidence=min_confidence,
            highlight_track=highlight_track,
            scale=scale,
        )
        if goal is not None:
            draw_goal(output, goal, scale=scale)
        for dive in dives or []:
            draw_dive_banner(output, dive, frame_poses.frame_index)
        rendered.append(output)
    return rendered
