# Architecture

## The shape of it

```
video ─► pose ─► tracking ─► lifting ─► calibration ─► analysis ─► export
```

Each stage is optional and independently testable. `Pipeline` holds no state
beyond its config, so two clips run concurrently by constructing two
pipelines.

Frames are **streamed, never batched into memory**. A two-minute 4K clip is
about 45 GB of decoded RGB; the only frames retained are the ones the overlay
renderer asked for, and those are kept at inference resolution.

## The one important decision: a canonical skeleton

Every pose backend emits its own layout — MediaPipe 33 landmarks, COCO 17,
Halpe 26, Sapiens up to 308. If that leaked downstream, every analysis
function would need a branch per backend and swapping backends would be a
rewrite.

Instead `skeleton.py` defines **COCO-17 as canonical**, and each backend
converts on the way out. Analysis code indexes joints through `J.LEFT_KNEE`
and never learns which model produced them. Backend-native keypoints survive
in `PersonPose.raw_xy` for anything that needs the extra detail.

COCO-17 is canonical because every candidate backend can produce it natively
or by subsetting, which keeps the conversion lossless in the direction that
matters.

## Adding a pose backend

Write one class and register it:

```python
@register
class MyBackend(PoseBackend):
    info = BackendInfo(key="mine", layout=COCO17, ...)

    @classmethod
    def available(cls) -> tuple[bool, str]:
        # Must NOT import the heavy dependency — use importlib.util.find_spec,
        # so that listing backends stays cheap.
        ...

    def setup(self) -> None: ...          # load models once
    def estimate_native(self, image): ... # -> [(xy, confidence), ...]
```

`PoseBackend.estimate()` handles canonical conversion and undoes the
inference downscale. Nothing else changes: the CLI, configs and the demo app
all refer to backends by name only.

Registration is lazy — the registry stores classes, not instances — so
`soccer-mocap backends` never imports torch.

## Coordinate spaces

Three, and mixing them is the easiest bug to write here.

| Space | Units | Where |
| --- | --- | --- |
| **Original video** | pixels, origin top-left, y down | everything in `PersonPose.xy` |
| **Inference** | pixels of the downscaled frame | inside a backend only |
| **Pitch** | metres, origin at the centre spot | `PersonPose.xy_pitch` |

Keypoints are always stored in **original video pixels**, even when
inference ran on a downscaled frame — `PoseBackend.estimate()` divides by the
scale factor. Anything drawing on a downscaled frame must therefore scale
back *down*; `draw_pose(..., scale=)` exists for exactly that, and the demo
app's goal-marking preview does the same conversion in reverse.

## Phone footage

`devices/presets.py` encodes what differs between phones in ways that change
the result, not just the file size:

- **Rotation.** Phones record landscape pixels and set a container flag.
  Decoders that ignore it hand you a sideways person, which is the one thing
  pose models reliably fail on. The flag is read and applied exactly once, in
  `VideoReader`.
- **Frame rate.** 30, 60, 240. Every velocity divides by it.
- **Rolling shutter.** Skews fast limbs on budget sensors, biasing exactly
  the moments of interest.
- **Field of view.** Ultra-wide lenses bend the straight lines a homography
  assumes.

A `ProcessingProfile` decides what to do about it: inference resolution,
frame stride, model complexity. `resolve_profile()` additionally adapts to
slow-motion and long clips, which the preset cannot know about.

## Robustness fixes worth knowing about

Three bugs found by running real inference on a real file, each fixed in a
way that generalises:

1. **Stature from segment lengths, not bounding-box height.** A horizontal
   diving keeper collapsed the vertical extent to 9 px and produced speeds of
   ~6000 body-heights/s. Summed segment lengths are orientation- *and*
   posture-invariant.

2. **One reference stature per track.** Height does not change during a clip,
   so per-frame normalisation only injects pose jitter into the denominator.

3. **Duplicate suppression by containment, not just IoU.** Pose models emit a
   small partial skeleton nested inside the real one during fast motion. Such
   a pair can have IoU as low as 0.2 while being unambiguously the same
   person, so an IoU-only filter lets it through — and it then starts its own
   track and splits a dive in two.

The tracker also falls back to **centre distance** when IoU fails, because a
body going from upright to horizontal turns a tall narrow box into a short
wide one that barely overlaps its predecessor.

## Honesty as a design constraint

Single-camera capture has hard limits, and the code is built to surface them
rather than paper over them:

- `PipelineResult.warnings` and `GoalkeeperReport.warnings` carry the caveats
  that apply to *that specific run* — uncalibrated distances, low frame rate,
  sparse detections, an uncertain keeper pick.
- Unknown config keys **raise**. A silently-dropped typo is how you come to
  believe you ran an ablation you never ran.
- `write_trc()` refuses to write 2D pixels into a 3D marker format, because
  OpenSim would load it without complaint and mean nothing by it.
- Unimplemented backends are **registered stubs** that name what is missing
  and how to enable it, rather than being absent or pretending to work.
- Implausible derived values (a 500 steps/min cadence) are withheld with an
  explanation instead of printed.

## Testing

98 tests, no video files. `tests/synthetic.py` builds COCO-17 skeletons from
high-level parameters with exact contracts:

```
measured knee angle      == 180 - knee_flexion_deg
measured stance width    == stance_width
```

That lets tests assert against physics rather than against whatever the code
currently returns — a velocity test checks a known constant speed, and the
dive tests check that left and right dives are symmetric, which is how a
scale bug announces itself.

## Where to extend

| Want | Go to |
| --- | --- |
| A better pose model | `pose/backends/`, implement a stub |
| Crowded multi-person tracking | replace `tracking/iou.py` with ByteTrack |
| Real 3D | `lifting/base.py`, `MonocularLifter` |
| Ball tracking | new module; unlocks reaction time and save outcomes |
| Automatic goal detection | `goalkeeper/goal.py`, `detect_goal_posts()` |
| Team assignment | `goalkeeper/identify.py` already samples kit colour |
