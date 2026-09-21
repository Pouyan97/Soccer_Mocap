# Goalkeeper analysis

How the goalkeeper track works, what each number means, and where it stops
being trustworthy.

## Why it is separate from the general analysis

A goalkeeper is a different measurement problem from an outfield player:

- They occupy a known, fixed region of the pitch.
- The goal beside them has **exactly known dimensions** — 7.32 m by 2.44 m,
  by law, everywhere. In a scene where nothing else has a reliable
  dimension, that is the most valuable calibration target available.
- What matters is short, explosive, and rare, rather than continuous.

That is enough of a difference to justify its own package rather than a flag
on the general path.

## The goal as a ruler

Two tapped points — the foot of each post where it meets the grass — give:

- a **metric scale** (`px_per_m = width_px / 7.32`),
- the **goal line**, which every positioning metric is measured against,
- the **goal centre**, for the coverage metric.

Marking the crossbar as well is optional and gives a second, independent
scale estimate. When the two disagree by more than ~20%, the report says so:
either the points are wrong or the lens distortion is severe.

**Why not detect the goal automatically?** A Hough transform over the white
pixels would find the posts most of the time. The failure mode is the
problem: a confidently wrong goal silently corrupts every metric downstream,
with no visible symptom. Two taps in the demo app is a better trade.
`detect_goal_posts()` exists as the seam if you disagree.

## Identifying the keeper

Three strategies, in descending order of trust:

1. **Manual** — you say which track id. One tap in the app.
2. **Geometric** — scores every track on median distance off the goal line
   (weighted 0.55), fraction of frames within the mouth's width (0.25), and
   stillness (0.20). Proximity dominates because it is by far the most
   diagnostic; the others break ties between a keeper and a centre-back
   camped on the line for a corner.
3. **Kit colour** — keepers must wear a colour distinct from both teams, so
   the keeper is the colour outlier. Fails on training footage where
   everyone wears the same bib.

Every strategy returns a **score and its reasoning**, not just an id, so a
wrong pick is visible rather than mysterious. The report carries the ranked
candidates.

## Scale-free units

Every measurement is either a **ratio** or in **body-heights per second**.
Neither depends on how far away the camera was or how tall the keeper is, so
numbers from different sessions are comparable with no calibration at all.
Multiply by the player's real stature to get SI units.

Stature itself is estimated from the **summed length of the body's segments**
(nose → shoulder → hip → knee → ankle), not from the bounding box height.
Two reasons, both of which bite hard here:

- **Orientation.** A diving keeper is horizontal. Vertical extent collapses
  toward zero exactly when the interesting things are happening — this
  produced velocities of ~6000 body-heights/s before it was fixed.
- **Posture.** A set keeper is crouched, understating their vertical extent
  by 10–20%.

One robust stature (the median) is used for the whole track, because a
person's height does not change during a clip and normalising each frame by
its own noisy estimate puts the pose model's jitter in the denominator.

## Set position

Found as the stillest frame **before the fastest movement in the track** —
the restriction matters, because a keeper lying on the ground after a dive
is the stillest they will be all clip.

Measured there:

| Metric | Meaning | Typical |
| --- | --- | --- |
| `stance_width_ratio` | ankle separation ÷ shoulder separation | 0.9–1.7 |
| `mean_knee_angle` | hip–knee–ankle, 180° = straight | 130–165° |
| `trunk_flexion_deg` | trunk away from vertical | 5–30° |
| `hand_height_ratio` | wrist height above hips, in body heights | > −0.02 |
| `com_height_ratio` | COM above ankles — crouch depth | — |

Values outside those bands produce a plain-language coaching flag ("narrow
stance — limited lateral base").

> The reference ranges are **coaching convention, not a normative dataset.**
> They exist to make the output legible. Replace them with values from your
> own cohort before anyone draws a conclusion from them.

## Dive detection

A dive is a burst of centre-of-mass speed **combined with** the trunk going
off vertical. The combination is the whole trick:

- fast but upright → a sidestep, rejected
- leaning but slow → a crouch, rejected
- fast and leaning → a dive

Defaults: 1.2 body-heights/s sustained for at least 4 frames, with at least
25° of trunk lean somewhere in the burst. Lower the speed threshold for
youth footage, where body-heights per second run higher for the same
absolute speed.

No ball tracking is involved, so a dive at nothing is indistinguishable from
a save.

### Phases

| Phase | How it is found |
| --- | --- |
| `set` | stillest frame before movement |
| `preparatory` | walking back to where speed lifted off its baseline |
| `takeoff` | speed crosses the threshold |
| `flight` | **ankles rise clear of their pre-dive level** |
| `peak` | maximum COM speed |
| `landing` | speed drops back below the threshold |

Flight is detected from the feet leaving the ground rather than from COM
height, because a diving body's centre of mass *falls* while it is airborne.

Phase boundaries are good to about one frame. At 30 fps a dive is only
around eight frames long, which is the main argument for filming keepers at
60 fps or higher; the report warns when the clip is slower than that.

### Classification

- **Direction** — from signed lateral COM displacement, with a deadband so a
  small shuffle reads as `centre`.
- **Height** — primarily from how far the COM dropped, because the hands are
  the joints most often lost to motion blur at the moment of the save.

### Reaction time

Reported **only** when you supply shot times. Reaction time is the interval
between the ball being struck and the keeper first moving, and nothing in
this project detects a ball. A reaction under 100 ms is flagged, because it
means either anticipation or a misaligned shot time — not superhuman
reflexes.

## Positioning

| Metric | Note |
| --- | --- |
| `median_depth_m` | distance off the goal line |
| `depth_range_m` | 5th–95th percentile spread |
| `median_coverage` | 0 = left post, 0.5 = centre, 1 = right post |
| `time_in_goal_area` | fraction of frames inside the six-yard box |
| `distance_covered_m` | total ground covered |
| `peak_speed_bh_s` | always available, calibrated or not |

**The main caveat:** distances off the line use a scale measured *at* the
line, so they become progressively optimistic as the keeper advances. Good
to roughly 10% for a keeper on their line. For anything further out,
calibrate a full pitch homography and use that instead.

The report attaches this caveat to its own output rather than relying on you
remembering it.

## Worked example

```bash
soccer-mocap run data/raw/your_clip.mov \
    --goalkeeper --goal-posts 300,300 1000,300
```

```
Goalkeeper: track 0
Clip: 2.5 s at 60.0 fps

Positioning
  Median distance off line : 1.87 m
  Median mouth position    : 0.67 (toward the right post)
  Distance covered         : 3.5 m
  Peak speed               : 4.54 body heights/s

Dives detected: 1
  1. right low dive, 0.33 s, peak 4.4 bh/s, travelled 2.13 m
```

Numbers above are illustrative; run it on your own footage to see real ones.

## From Python

```python
from soccer_mocap.goalkeeper import GoalGeometry, build_report, identify_goalkeeper

goal = GoalGeometry(left_post=(300, 300), right_post=(1000, 300))
track_id, candidates = identify_goalkeeper(tracks, goal=goal)

report = build_report(tracks[track_id], fps=60.0, goal=goal, candidates=candidates)
print(report)                 # formatted text
report.to_dict()              # JSON-ready
report.dive_table()           # rows for a DataFrame
```

## Not implemented

- Ball tracking, and therefore true reaction time and save/concede outcomes
- Automatic goal detection
- Handling a camera that pans
- Distinguishing a save from a dive at nothing
