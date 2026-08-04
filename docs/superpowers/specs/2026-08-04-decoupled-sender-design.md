# Decoupled sender, rotation prediction, and a measurement harness

Date: 2026-08-04
Status: approved, not yet implemented

## Problem

Two defects in the output path, plus the fact that neither can currently be
settled the way every other lever in this project was settled — by measurement.

**1. Output is locked to camera rate.** `cmd_run` sends inside the capture loop,
paced by `period = 1.0 / SEND_HZ` with `SEND_HZ = 30`. `TrackerPredictor`'s own
docstring advertises the fix and the loop never uses it:

> Prediction also lets output run faster than the camera. VRChat applies tracker
> data per rendered frame, so 30Hz updates on a 72-90Hz headset are held for two
> or three frames at a time; predicting at each send smooths that.

On a 72 Hz Quest, a 30 Hz send is held for 2.4 rendered frames, adding an average
~16.5 ms of staleness on top of the ~50 ms pipeline latency already measured.
No amount of filtering addresses this, because the filter is not the bottleneck —
the update rate is.

**2. Positions are predicted, rotations are not.** `bodytracker.py:243` takes
positions from `latch.at(t, lead)`, extrapolated ~50 ms forward, while the line
above takes rotations from `rotation_latch.all_current()`, unpredicted. Hip and
chest rotations derive from the same joints whose positions were just
extrapolated, so every frame ships an internally inconsistent pose: a hip
predicted 50 ms ahead carrying a hip rotation 50 ms stale. VRChat's IK resolves
against both.

**3. The headline number is not reproducible.** The lead sweep in commit
`4a88d40` ("0ms → 9.0mm, 50ms → 3.8mm") was produced by a script that never
landed in the repo; that commit touched only README.md, bodytracker.py and
transform.py. `measure` reports jitter/lag/bone-spread but not display-time
error. There is currently no way to re-derive the number the README leads with,
and therefore no way to check whether a change to the output path helps.

## Non-goals

- Async MediaPipe inference (`LIVE_STREAM` mode). A real latency win, but it
  restructures the loop and would invalidate the existing lead-time sweep.
  Separate project.
- Per-joint depth Kalman filtering. Speculative against a codebase that measured
  four depth reducers and found 7% between them.
- Adaptive-cutoff rotation smoothing. `RotationSmoother`'s reject/accept
  hysteresis already covers the foot-flip case it was built for, and hips/chest
  measure 0.3–0.7 deg/frame. Revisit only if the sweep says rotations dominate.

## Methodology: why the recording must be dense

The pipeline runs at ~30 fps. Scoring a 90 Hz sender requires knowing where the
body was *between* camera frames, and a 30 fps recording does not contain that.
Interpolating between recorded frames to fill the gap would bias the yardstick
toward the thing under test: linear interpolation between two samples is exactly
the model linear extrapolation assumes, so prediction would score well because
the ground truth was built on its own premise. That is the same circularity
`4a88d40` already had to fix once ("it compared the predicted pose against the
same frame, so lead=0 scored a perfect 0.0mm").

The fix is to record faster than the pipeline runs and subsample:

```
record at R fps ─┬─ every k-th frame (k = R/30) → what the pipeline sees
                 └─ all frames                  → ground truth
```

Ground truth is then composed of real measurements the pipeline never consumed.
Send rates at or below R are measured; above R the sweep reports them as
unmeasured rather than interpolating.

Inference costs ~24 ms, so frames cannot be solved at R fps live. The recorder
buffers raw colour+depth to disk, then solves every frame offline and keeps only
the joint track.

## Component 1: `bodytracker record`

```
bodytracker record --save session.npz [--seconds 20] [--fps 0]
```

- `--fps 0` (default) probes the device for the highest colour framerate
  available at the configured resolution and uses it; an explicit value overrides.
  `DepthCamera` already accepts `fps`.
- Estimates output size before starting (colour 848×480×3 + depth 848×480×2 ≈
  **2.03 MB/frame**; 20 s at 60 fps ≈ **2.4 GB**), checks free space, and aborts
  with a clear message rather than filling the disk.
- Buffers raw frames to `np.lib.format.open_memmap` arrays in a temp directory,
  so RAM use stays flat regardless of duration.
- After capture, solves every buffered frame with the production `PoseSolver`,
  then deletes the temp files.
- Saves `times`, `joints`, and `rate` (the true recording rate) to the `.npz`.
  `*.npz` is already gitignored.
- Records `visual_preset_applied` and `sensor_warnings` into the archive, so a
  recording made on a degraded camera stream is identifiable after the fact
  rather than silently poisoning every measurement taken from it.

## Component 2: `bodytracker sweep`

```
bodytracker sweep --load session.npz [--display-hz 72]
```

Replays the dense track through the real production objects — `SkeletonStabilizer`,
`build_trackers`, `TrackerPredictor`, `RotationPredictor` — never a reimplementation.

Simulation per (send_hz, lead) cell:

1. Pipeline input: every k-th dense frame, k = round(rate / 30).
2. For each such frame: stabilise, build trackers, `predictor.update(...)`.
3. Sender: at each send instant on a fixed clock, `predictor.at(s, lead)`.
4. Display: at each instant on a `display_hz` clock, the pose VRChat holds is the
   most recent send at or before it. This is what models the staircase.
5. Ground truth at that instant: `build_trackers` on the nearest dense frame,
   unstabilised and unpredicted.

Reported: median display-time error, median step between consecutive display
instants (jitter, which captures both extrapolation noise and the hold
staircase), and worst-case error.

Two honesty requirements, both to be stated in the printed output:

- **Ground truth is unstabilised**, so it carries measurement noise. Absolute
  error values are therefore inflated; only *differences between rows* are
  meaningful. Stabilised truth was rejected because it would build the
  stabiliser's own lag into the yardstick.
- **Today's behaviour is a distinct row.** The current sender is camera-triggered
  (sends immediately after each frame, zero added phase); a fixed 30 Hz clock adds
  up to 33 ms of phase. Modelling today as a fixed clock would understate it. The
  table therefore reports `30 (camera-triggered)` — the status quo — separately
  from fixed-clock rows.

## Component 3: decoupled sender

New `PoseSender` in `osc_out.py`: a daemon thread owning a `TrackerSender`.

- `update(trackers, rotations, t)` — called from the capture loop, takes a lock,
  stores a snapshot.
- Thread loop: sleep to the next tick, snapshot under the lock, evaluate
  `predictor.at(now, lead)` and the rotation predictor, send one bundle.
- `stop()` sets an `Event`; joined on context-manager exit.
- Prediction is evaluated **on the sender thread**, not the capture thread — that
  is the entire point. The capture thread contributes measurements; the sender
  extrapolates them to its own clock.

CLI: `--send-hz`, default `0`, meaning "send once per camera frame — current
behaviour". Any positive value starts the thread at that rate. Default 0 keeps
shipped behaviour bit-identical until the sweep justifies otherwise, which is the
agreed rollout.

## Component 4: rotation prediction

New `RotationPredictor` in `transform.py`, mirroring `TrackerPredictor` in SO(3).

Euler angles are never extrapolated — that is wrong at the wrap, for the same
reason `RotationSmoother` already refuses to average them. Work in matrices:

- Angular velocity from consecutive matrices: `dR = R_cur @ R_prev.T`, converted
  to a rotation vector via Rodrigues, divided by `dt`.
- EMA-smoothed, like linear velocity and for the same reason: it is a difference
  of two noisy orientations.
- Clamped in angular speed, and the prediction horizon clamped to 0.12 s, matching
  `TrackerPredictor`'s existing guards.
- `at(t, lead)` returns euler only at the very end.

Needs Rodrigues helpers (`rotvec_to_matrix`, `matrix_to_rotvec`) in `transform.py`.

## Testing

Unit tests, no camera required:

- `RotationPredictor`: constant angular velocity is extrapolated correctly;
  zero velocity is a no-op; a rotation crossing the ±180° wrap does not produce a
  spurious flip; angular-speed and horizon clamps hold.
- Rodrigues round-trip: `matrix_to_rotvec(rotvec_to_matrix(v)) == v` over random
  vectors, including near-zero and near-π.
- `PoseSender`: against a fake sender, achieved rate is within tolerance of the
  requested rate; `stop()` joins without hanging; concurrent `update()` while the
  thread sends never yields a torn pose (position from one frame, rotation from
  another).
- `sweep` replay is deterministic: same archive and parameters produce identical
  numbers.

Existing 35 tests must continue to pass, and `--send-hz 0` must leave the sent
byte stream unchanged from today.

## Rollout

1. Implement and test all four components with `--send-hz 0` default.
2. User records one session and runs the sweep.
3. Defaults set from the resulting table; README updated with the numbers.
4. If 90 Hz does not beat camera-triggered 30 Hz on display-time error, the
   default stays as it is and the negative result is documented. That is a valid
   outcome, not a failure.
