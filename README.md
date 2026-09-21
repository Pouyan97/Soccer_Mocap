# Soccer_Mocap

Single-camera markerless motion capture and analysis for soccer, built on
open-source pose estimation, with a goalkeeper-specific analysis track and a
demo app that runs in any phone's browser.

Everything is Python, and the environment is managed with mamba.

---

## What this is

Point one phone at a player, get back joint trajectories, kinematics, and a
coaching report. No suits, no markers, no second camera.

The design is a pipeline of swappable stages:

```
video → pose → tracking → lifting → calibration → analysis → export
```

The **pose** stage is the one that matters and the one most likely to change,
so it sits behind a registry. MediaPipe runs today; Sapiens, RTMPose and
PosePipeline are registered as stubs carrying their integration notes. Every
backend converts to one canonical COCO-17 skeleton on the way out, so nothing
downstream knows or cares which one produced a pose.

A **goalkeeper** package sits alongside the general analysis, because a keeper
is a different measurement problem: a known region of the pitch, a goal of
exactly known size to calibrate against, and short explosive actions rather
than continuous running.

## Quick start

```bash
mamba env create -f environment.yml
mamba activate soccer-mocap
pip install -e .

python scripts/download_models.py mediapipe   # required: see note below
```

MediaPipe 1.0 removed the bundled weights, so that download is a setup step,
not an optional extra. Full details and the Windows/CUDA gotchas are in
[docs/setup.md](docs/setup.md).

Then, with your own footage — any video of someone moving, soccer or not,
`.mp4`/`.mov`/anything OpenCV or PyAV can decode:

```bash
soccer-mocap probe data/raw/your_clip.mov          # sanity-check first
soccer-mocap run data/raw/your_clip.mov            # general analysis
soccer-mocap run data/raw/your_clip.mov --goalkeeper --goal-posts 300,300 1000,300
```

## Using it

```bash
soccer-mocap backends          # what is installed, what is planned
soccer-mocap devices           # phone presets and capture advice
soccer-mocap probe clip.mov    # metadata and warnings, no inference
soccer-mocap run clip.mov -c configs/goalkeeper.yaml
soccer-mocap demo              # the phone app
```

From Python:

```python
from soccer_mocap import Pipeline, PipelineConfig

config = PipelineConfig()
config.goalkeeper.enabled = True
config.goalkeeper.goal_posts = [[410, 520], [980, 528]]

result = Pipeline(config).run("keeper.mov")
print(result.goalkeeper)
```

## The demo app

```bash
soccer-mocap demo --share
```

A Gradio web app, which is the only Python-only way to reach both iOS and
Android without writing a native app. Start it on a laptop, open the URL on a
phone, record or upload a clip. `--share` gives a temporary public link for
phones that are not on the same Wi-Fi.

Four tabs: **Analyse** (upload, run, get an overlay and a report),
**Goalkeeper** (tap the two goal posts to get metres), **Capture guide**
(device-specific filming advice), and **Backends**.

It handles the things that actually differ between phones: container rotation
flags, fractional and slow-motion frame rates, HEVC from iPhones, and lenses
wide enough to bend the pitch lines.

## Goalkeeper analysis

The part with the most domain logic. See
[docs/goalkeeper.md](docs/goalkeeper.md) for the methods and their limits.

- **Goal geometry.** A goal is 7.32 m wide by law, which makes it the one
  object in frame with a known dimension. Two tapped posts give a metric
  scale and the reference line every positioning metric is measured against.
- **Identification.** Picks the keeper out of the tracks by proximity to the
  goal, containment within the mouth, and stillness — or by kit colour, or
  manually. Always reports *why*, so a wrong pick is visible.
- **Set position.** Stance width, knee and hip angles, trunk lean, hand
  height, crouch depth. All as ratios, so clips shot at different distances
  are comparable without any calibration.
- **Dives.** Detected from centre-of-mass speed combined with trunk lean —
  the combination is what separates a dive from a fast sidestep. Segmented
  into set / preparatory / takeoff / flight / landing, and classified by
  direction and height.

Speeds are reported in **body-heights per second**, a dimensionless unit that
survives changes in camera distance. Multiply by the player's real stature for
m/s.

## Layout

```
src/soccer_mocap/
  skeleton.py       canonical COCO-17 layout and the cross-backend remapping
  types.py          the data contract every stage speaks
  config.py         nested dataclass config, YAML-backed, typo-rejecting
  pipeline.py       stage orchestration
  cli.py            command line entry point
  io/               video decode (rotation, HEVC, timestamps) and exporters
  devices/          phone capture profiles and processing presets
  pose/             the backend interface, registry, and implementations
  tracking/         IoU + centre-distance tracking, duplicate suppression
  detection/        person detection, for top-down pose backends
  lifting/          2D → 3D, and an honest account of what that can mean
  calibration/      pitch homography
  analysis/         kinematics, filtering, kick and contact events
  goalkeeper/       goal geometry, identification, stance, dives, report
  viz/              skeleton and event overlays
apps/demo/          the Gradio phone app
configs/            ready-made pipeline configs
scripts/            model download utility
tests/              98 tests, no video files needed
docs/               setup, architecture, goalkeeper methods
```

## Status

Working end to end: decode → MediaPipe pose → tracking → 3D passthrough →
goalkeeper report → JSON/CSV/overlay export, driven from the CLI or the phone
app.

Registered but not implemented, each with integration notes in its module:
Sapiens, RTMPose, PosePipeline, learned 2D→3D lifting, automatic goal
detection, ball tracking.

`soccer-mocap backends` always shows the current state.

## Caveats worth reading before trusting a number

- **A single camera cannot measure absolute depth.** 3D output is a model
  prior about human bodies, not a measurement. Relative joint angles are
  defensible; absolute positions in space are not.
- **A homography maps the ground plane only.** Project feet, never heads.
- **Distances off the goal line** use a scale measured *at* the line, so they
  grow optimistic as the keeper advances. Good to roughly 10% on the line.
- **Film at 60 fps or better** for dives and strikes. At 30 fps a dive is
  about eight frames and the phase timings are coarse.
- **Use a tripod.** Any camera motion invalidates a single calibration.
- The coaching reference ranges in the stance analysis are convention, not a
  normative dataset. Replace them with values from your own cohort before
  drawing conclusions.

The pipeline attaches these caveats to its own output rather than relying on
you having read this section.

## Development

```bash
pytest                 # unit tests for kinematics/tracking math (no footage needed)
ruff check .
```

`tests/synthetic.py` builds hand-built skeletons with known ground truth
purely to unit-test the *math* (angle formulas, velocity, height estimation) —
it never stands in for real footage. Every actual pipeline run should be
against a real video; put yours in `data/raw/` (gitignored).
