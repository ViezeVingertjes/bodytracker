# What is actually installed on this machine

Executed 2026-08-04 on Ubuntu 26.04 (kernel 7.0.0-28-generic). This is the record of the
real setup; `RESEARCH.md` §5 is the plan it was based on.

## Hardware — verified

```
Intel RealSense D415
  USB id        8086:0ad3
  serial        932122060573
  firmware      5.17.0.9      (Nuitrack needs >= 5.8.15 -- no update required)
  USB link      5000M (USB 3.0 SuperSpeed) on bus 004
```

Confirmed streaming: `rs-distance` returned stable live depth readings (~1.17 m off a
real surface), so the depth pipeline works end to end, not just enumeration.

## librealsense

Built from source — there is **no apt package for Ubuntu 26.04**.

- Source: `vendor/librealsense` (upstream **`realsenseai/librealsense`**, shallow clone,
  commit `7c3ee3f`, 2026-07-22). Note the org: RealSense spun out of Intel in 2025;
  `IntelRealSense/librealsense` is legacy and `intelrealsense.com` no longer resolves.
- Version built: **2.58.3**
- Binaries: `vendor/librealsense/build/Release/` (not installed system-wide)

Configure flags used:

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=true \
  -DBUILD_EXAMPLES=true \
  -DBUILD_GRAPHICAL_EXAMPLES=true \
  -DCHECK_FOR_UPDATES=false \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build -j 14
```

Because nothing was installed system-wide, tools need the library path:

```bash
cd vendor/librealsense/build/Release
LD_LIBRARY_PATH=$PWD ./realsense-viewer
LD_LIBRARY_PATH=$PWD ./rs-enumerate-devices
```

## Three places reality differed from the plan

**1. `-DCHECK_FOR_UPDATES=false` is required.** Without it the build fails at the very
end, linking `realsense-viewer` and `rs-depth-quality`:

```
libcurl.a(idn.c.o): undefined reference to `idn2_lookup_ul'
```

librealsense vendors a static libcurl solely for its update-check feature, configured
with IDN support but without linking `libidn2`. Disabling the feature drops the
dependency. Keep this flag on any rebuild.

**2. Do not run `scripts/setup_udev_rules.sh`.** It depends on `v4l-utils` (not installed
here) and interactively demands you unplug the camera partway through. Only the libusb
rules matter for the RSUSB backend, so this is sufficient:

```bash
sudo cp vendor/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

The rules file covers `8086:0ad3` with `MODE:="0666", GROUP:="plugdev"`, and the user is
already in `plugdev`. Before this, `rs-enumerate-devices` failed with
`RS2_USB_STATUS_ACCESS` — the device node was `crw-rw-r-- root root`. After: `crw-rw-rw-
root plugdev`, and the camera enumerates. If it ever regresses, replug the camera.

**3. Python is 3.12 in a venv, not system 3.14.** The plan assumed a
`--system-site-packages` venv over system Python. That does not work here:

- system Python is **3.14**, but librealsense pins **pybind11 v2.13.6**, which predates
  Python 3.14 support
- MediaPipe has no 3.14 wheels either, which would block Option B outright

So `.venv` is a **uv-managed CPython 3.12.13**, and `pyrealsense2` is built against that
interpreter. This also sidesteps PEP 668 entirely — nothing is installed into system
Python.

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install python-osc numpy
```

Note `uv venv` does not provide `pip` inside the venv — use `uv pip install` (with
`VIRTUAL_ENV` set), not `.venv/bin/python -m pip`.

## Python capture — verified

`pyrealsense2` **2.58.3** built against the venv's Python 3.12 and imports without any
`LD_LIBRARY_PATH` (RUNPATH is baked into the `.so`). Wired in via a `.pth` file in
site-packages pointing at `vendor/librealsense/build/Release`.

Measured on the real camera through `capture.py`:

```
848x480 @ 29.8 fps sustained, colour + depth aligned
depth_scale     0.001 m/unit
intrinsics      fx=612.2 fy=611.2 ppx=429.0 ppy=236.7 (inverse_brown_conrady)
deprojection    centre pixel -> (-0.010, 0.006, 1.185) m  [sane]
USB             3.2
depth coverage  84.8% of frame valid, 0.39-10.79 m
                ^ scene-dependent smoke test (camera pointed at a cluttered desk
                  at ~1.2 m). NOT a spec figure -- do not read this as expected
                  coverage against a person at 2.3 m.
```

**Axis signs verified empirically** rather than assumed: +X right, **+Y down**, +Z
forward (right-handed). Above centre gives Y=−0.237, below gives Y=+0.231. `transform.py`
will be written against this, and it is the classic mirrored-avatar bug if wrong.

**FOV: unsettled, see `RESEARCH.md` §3.** Device intrinsics report ~70° H × 43.1° V vs
the datasheet's 65°×40°. Do not treat the intrinsics as four confirmations — `fy/height`
is exactly 1.26746 in all four depth modes, i.e. one calibration scaled by resolution.
Needs a tape-measure test to resolve; plan against the conservative 2.5 m until then.

Do avoid **640×480** regardless: it crops horizontal FOV to 55.5° for nothing. Use
848×480 (also Nuitrack's recommended size) or 1280×720.

Depth-valid area spans 98% of width and 99% of height of the colour-aligned frame, so
aligning depth→colour costs essentially no usable area here.

## Camera is mounted upside down

The D415 is physically inverted, so `DepthCamera(rotate_180=True)` is the normal mode
here (the app defaults to it; pass `--upright` if the camera is ever remounted).

This rotates the frames **and** the intrinsics together. Rotating only the image is the
trap: MediaPipe would detect the pose happily, the preview would look perfect, and
deprojection would quietly return mirrored 3D coordinates. A 180° rotation maps pixel
(x, y) → (W−1−x, H−1−y), so the principal point moves the same way and focal lengths are
unchanged.

Verified numerically — two arbitrary points deprojected through the rotated path returned
exactly (−X, −Y, Z) versus the unrotated path, and ppx 429.00→418.00, ppy 236.68→242.32
as predicted.

Only safe because **every distortion coefficient on this camera reads 0.0**. With
tangential distortion, p1/p2 would need sign flips; `_rotated_intrinsics` raises rather
than silently mis-correcting if it ever sees non-zero coeffs.

## OSC output — verified

`osc_out.py` + `bodytracker.py fake` were tested against a loopback OSC receiver. All
eight expected addresses arrive with correct float payloads:

```
/tracking/trackers/1/position         (0.0, 1.0, 0.0)
/tracking/trackers/1/rotation         (0.0, 0.0, 0.0)
/tracking/trackers/3/position         (-0.12, 0.05, 0.0)
/tracking/trackers/4/position         (0.12, 0.05, 0.0)
/tracking/trackers/head/position      (0.0, 1.7, 0.0)
...
```

## Validated against real VRChat — 2026-08-04

VRChat installed via **snap** Steam, running under Proton Experimental with `--no-vr`.

**Snap confinement is not a problem.** `VRChat.exe` shares our network namespace
(`net:[4026531833]`, identical to ours), so `127.0.0.1` reaches it. This was worth
checking — a private loopback would have swallowed everything silently.

`VRChat.exe` binds `0.0.0.0:9000` **only after OSC is enabled in-game** (desktop mode:
hold `R` → Options → OSC → Enabled). Nothing on 9000 means OSC is off, not that VRChat is
broken.

Results:

- ✅ **All 8 tracker indices are accepted and rendered.** Probed 1–4 then 5–8; 4 of 4
  appeared in OSC Debug both times. VRChat does not filter by index.
- ✅ Address format, port, float payloads, and metre scaling all accepted by VRChat itself
- ❌ **Avatar does not move** — expected. OSC trackers feed the *calibrated* FBT IK; without
  calibration VRChat knows a tracker exists but not which body part it drives. True even
  with a headset.
- ✅ **Index→role mapping resolved** — by reading SlimeVR's source rather than testing.
  hip=1, feet=2/3, knees=4/5, chest=6, elbows=7/8. VRChat's prose order (hip, chest,
  feet, knees, elbows) is NOT the index order, and taking it literally gave chest=2 and
  feet=3/4, which is wrong. VRChat itself infers roles from position at calibration
  (its Auto-center finds "the two lowest trackers" and calls them feet), so the
  numbering matters for other OSC tooling, not for VRChat's IK.

**How to probe indices without misreading the result:** 4 at a time, chest height
(y≈1.10), within a ±0.4 m row, alternating bob phase so neighbours move in opposition.
`bodytracker.py fake --indices 1,2,3,4`. A vertical stack of 8 spanning y=0.2–1.6 put
index 8 within 4 cm of the head anchor at y=1.70; they merged and the count read as 7.
Spreading markers too far instead pushes them off screen, which reads as absent.

## Tools in this repo

| File | What it does |
|---|---|
| `capture.py` | `DepthCamera` — aligned colour+depth, `deproject()` pixel→metres, `depth_at()` hole-tolerant depth sampling |
| `osc_out.py` | `TrackerSender` — VRChat OSC tracker output |
| `bodytracker.py fake` | Synthetic trackers at 30 Hz, no camera. `--indices 1,2,3,4` probes specific indices as a countable row. |
| `bodytracker.py listen` | Listens on 9001 for VRChat's *outgoing* OSC — proves OSC is enabled |

## Testing against VRChat on this machine

VRChat has no native Linux build; it runs under Proton. Steam setup:

1. `sudo add-apt-repository -y multiverse && sudo apt-get install -y steam-installer`
2. Steam → Settings → Compatibility → **Enable Steam Play for all other titles**
3. Install **Proton EasyAntiCheat Runtime** from the Library (a Tool — enable the Tools
   filter to see it). VRChat uses EAC.
4. Proton-GE is worth preferring over Proton Experimental (bundles the media codecs
   VRChat's video players want).
5. In VRChat: **Options → OSC → Enabled**, then **OSC Debug**.

**Known limitation — desktop VRChat cannot do FBT.** All body-tracking and FBT
calibration settings are hidden when no HMD is connected (there is an open VRChat feature
request asking for exactly this, for OSC tracker debugging). So on this machine you get:

- ✅ OSC Debug lists received trackers and their values — real validation of address
  format, port, float types, height scaling
- ❌ no calibration, no avatar movement, and **no answer to the index→role question**
  (`RESEARCH.md` §2). That requires the Quest.

## Tracking quality — measured 2026-08-04

Everything below is measured with `bodytracker.py measure`, which records once
and replays the **same** recording through every configuration. Re-running the camera
per configuration compares different movements, not different settings.

### Depth post-processing (capture.py)

Paired comparison on identical frames — raw vs the RealSense filter chain
(disparity → spatial → temporal → hole-fill):

```
RAW       coverage 90.4%    per-pixel temporal noise 13.37 mm
FILTERED  coverage 96.2%    per-pixel temporal noise  6.52 mm
          +5.8 pts                                    -51%
```

The disparity round-trip matters: stereo error is uniform in *disparity* and grows
quadratically in depth, so smoothing raw metres over-smooths near geometry and
under-smooths far — at 2.5 m that is exactly where a body is.

`visual_preset = HIGH_DENSITY` needs a **retry loop**. The option goes over a USB
extension unit that is busy during stream start-up, and the first attempts fail with
`get_xu(ctrl=1) failed! Resource temporarily unavailable`. Without retrying it
silently never applies.

### Joint stabilisation (stabilize.py)

416-frame recording of real movement:

| config | jitter | lag | bone spread | worst jump |
|---|---|---|---|---|
| raw | 13.7 mm | 0 | 65.5 mm | 594 mm |
| gate only | 3.6 mm | — | 43.8 mm | 143 mm |
| bones only | 3.8 mm | — | 27.4 mm | 163 mm |
| **full** | **2.8 mm** | 43.9 mm | **26.7 mm** | **148 mm** |

Jitter −80%, bone spread −59%, worst glitch −75%.

**Read jitter and lag together.** Jitter alone is a metric you win by smoothing
until the tracker stops following you. `min_cutoff` 0.3 → 2.0 trades 3.7 mm of
jitter against 10.6 mm of lag; **0.5 is the chosen default**.

Two measurement traps hit along the way, both of which produced confidently wrong
numbers before being caught:

- An **unpaired** filter comparison said filtering made jitter 4× *worse*. It was
  comparing two different moments of a moving scene. Always compare on identical frames.
- The aggregate bone-spread metric said the constraint did nothing, because it
  averaged over torso bones that `apply()` never corrected. Per-bone breakdown showed
  it working (L elbow–wrist 129 → 23 mm). Torso bones are now corrected too.

### The actual bottleneck: framing

From the same recording:

```
nose, shoulders, hips  100%
knees                  87% / 95%
ANKLES                 0.7% / 0.2%      <-- 3 and 1 frames out of 416
```

**Feet are two of the three default trackers, and they are essentially never seen.**
No amount of stabilisation helps a joint that is not in frame. Getting the whole body
into frame — the §3 constraint — dominates every other quality lever here.

## Pipeline quality — final measurements

Recording of a real person, 446 frames, independently verified as human (8/8 body
segments within adult range: shoulders 0.35 m, thigh 0.43 m, torso 0.55 m — worth
checking, because MediaPipe will happily fit a pose to furniture).

| stage | jitter | bone spread | worst glitch |
|---|---|---|---|
| raw | 10.2 mm | 84.9 mm | 865 mm |
| **full stabilisation** | **4.4 mm** | **56.8 mm** | **218 mm** |

- **Occlusion filling** takes ankle availability from 71%/87% to **100%/100%**, with
  5 mm median reappearance jump. Missing joints are held in the *body* frame, not
  world space, so an unseen foot travels with the hips instead of being stranded.
- **Rotation smoothing** removes all 14 foot-yaw flips (p95 15.4° → 3.4°/frame) while
  leaving already-steady hip/chest rotation alone. Smoothing happens on rotation
  *matrices* — averaging euler angles would send a foot oscillating around ±180° to
  0°, i.e. pointing backwards.
- **Gate limits 5.0/2.0 m/s** were found by sweep and are a genuine minimum:
  tightening to 3.0/1.5 makes the worst glitch *worse* (287 mm vs 218 mm), because an
  over-eager gate holds a joint then releases it further away.

### Performance

```
446 frames in 15 s = 29.7 fps -- camera-limited, not CPU-limited
capture   19.55 ms  <- wait_for_frames IDLING for the next 30 Hz frame, not compute
solve     12.90 ms  <- MediaPipe inference, the only real cost
stabilise  0.11 ms
transform  0.02 ms
osc        0.13 ms
```

Actual CPU work is 13.2 ms/frame against a 33 ms budget — roughly 20 ms of headroom.

### OSC transport

One **bundle** per frame, not loose messages. Eight trackers plus head is 18 messages;
sent individually that is 540 datagrams/second, enough for a busy WiFi link to reorder
or drop some and tear a pose across frames (a hip from frame N with a foot from N−1).
A bundle is one datagram: it arrives whole or not at all.

`TrackerSender` exposes a single `send_frame()`. An earlier queue+`flush()` design let
a caller silently transmit **nothing** by forgetting to flush — which is exactly what
happened to `fake_tracker.py` during development, and was only caught because a test
asserted on received addresses rather than on the code looking correct.

## Code review findings (2026-08-04)

Two real bugs found by reviewing behaviour rather than reading code. Both would have
shipped and both are the kind that look fine in the source:

**1. `RotationSmoother` locked out genuine rotation, permanently.** It rejected any
frame differing >45° from the stored matrix — so a real 90° foot turn differed by 90°
from the stale stored value *every subsequent frame*, was rejected forever, and the
foot never turned again. Caught by feeding a sustained 90° change and watching the
output sit at 0.00 indefinitely. Fixed with a consecutive-rejection counter: three
rejections (100 ms) and the new orientation is accepted as real. Verified across all
three cases — sustained turn converges, single-frame glitch suppressed, normal motion
smooth.

**2. The OSC queue+`flush()` API silently sent nothing.** `send_tracker()` appended to
a buffer that a separate `flush()` transmitted; any caller that forgot to flush
transmitted **zero**, with no error. That is exactly what happened to
`fake_tracker.py`. Replaced with a single `send_frame()` that takes the whole frame,
making the mistake unrepresentable rather than merely fixed.

**Plausibility bounds were initially far too tight — twice.** First a 0.25 m minimum
shoulder width rejected 100% of live frames. Loosening that revealed a 0.30 m torso
floor rejecting 36.5%. Measuring the actual live distribution showed why no fixed
threshold can work: the *same person* measures torso 0.55 m / shoulders 0.35 m standing
and framed, but torso 0.31 m / shoulders 0.24 m seated and closer. An absolute floor
cannot tell "not a person" from "person in a different posture", and every wrongly
rejected frame is a frozen limb in VRChat.

Replaced with a **self-calibrating** check: absolute bounds only catch gross nonsense,
and real discrimination is deviation (>40%) from a rolling median of *this* person's
torso length, learned over 40 accepted frames. Result: **100% tracked over 1050
consecutive live frames**, from 63%.

*(superseded)* **Plausibility bounds were initially too tight.** A 0.25 m minimum shoulder width
rejected 100% of live frames (`no pose -- shoulder width 0.20 m`). Turning sideways
occludes the far shoulder, its depth collapses onto the near body, and measured width
shrinks well below any frontal value — so a tight bound drops tracking exactly when
someone turns. Loosened to 0.12 m and the discriminating work moved to torso length,
which is vertical and barely affected by yaw. Regression-tested: accepts a person
turned hard (0.15 m), still rejects the original false positive (0.105 m).

Dead code removed (`capture.depth_at`, superseded by `depth_samples` + body-depth
gating). `stale_keys` and `last_rejection` were unused but worth keeping, so they are
now surfaced in the tracker's status output — a stale tracker is still being *sent*, so
without reporting it looks correct in VRChat while being wrong.

`ruff check --select E,F,W,B,SIM,UP,C4,RET,ARG`: clean.

## Camera settings — measured, not assumed (2026-08-04)

Run with `benchmark.py`. Depth-only benchmarks need a static scene; `body` and
`models` need a person in frame.

### Visual preset: MEDIUM_DENSITY

On a real body, 5 s per preset:

| preset | detected | joints/frame | jitter | bone spread | ankles |
|---|---|---|---|---|---|
| HIGH_DENSITY | 97% | 12.4 | 7.6 mm | 55.3 mm | 58% |
| **MEDIUM_DENSITY** | **100%** | **14.3** | **5.1 mm** | **41.4 mm** | **96%** |
| HIGH_ACCURACY | 65% | 11.4 | 20.3 mm | 116.2 mm | 21% |

MEDIUM_DENSITY wins on every metric; ankle availability 58% → 96% matters most,
since ankles are two of the three trackers we send.

**This is the opposite of what a static scene says.** On a textured desk,
HIGH_ACCURACY looks best (3.61 mm noise vs 6.53 mm) and was very nearly adopted on
that basis. A desk is nothing like a body: clothing and skin are low-texture, which
is exactly where HIGH_ACCURACY loses the coverage a pose model needs. **Benchmark on
a person, not on furniture.**

### Resolution / framerate: 848×480 @ 30 — higher is worse

| mode | coverage | noise | achieved fps |
|---|---|---|---|
| 1280×720@30 | 96.0% | 8.18 mm | 29.4 |
| **848×480@30** | **96.2%** | **5.48 mm** | **30.1** |
| 848×480@60 | 96.2% | 7.57 mm | 59.9 |
| 640×360@30 | 96.3% | 5.72 mm | 30.1 |

1280×720 carries 50% more depth noise at identical coverage — stereo matching uses
smaller correlation windows per pixel. 60 fps halves exposure and costs 38% more
noise. So the default is a genuine optimum, not a compromise.

### GPU: unavailable, and not needed

```
CPU: OK  median 11.61 ms
GPU: UNAVAILABLE -- ImageCloneCalculator: GPU processing is disabled in build flags
```

The machine has an AMD Barcelo iGPU (no NVIDIA). MediaPipe exposes a GPU delegate
enum but the pip wheel is built with GPU processing disabled; using it would require
building MediaPipe from source. OpenCV reports no OpenCL, and librealsense's
acceleration is CUDA-only, so NVIDIA-only regardless.

It does **not** degrade gracefully — requesting the GPU delegate raises — so the code
correctly never asks for it. Nor is it needed: the pipeline is camera-limited at
30 fps with ~20 ms/frame spare, so moving inference to a GPU would buy idle time,
not framerate.

### Pose model: still `full`, unmeasured on a body

`lite` and `heavy` are downloaded (`models/`) and `benchmark.py models` compares
them, but that benchmark needs a person in frame and has not been run. `full` costs
11.6 ms against a 33 ms budget, so `heavy` is probably affordable and may track
better. Switch with `PoseSolver(model_path=...)` if you ever want to test it.

### Preset application was silently failing

`visual_preset` goes over a USB extension unit that is busy during stream start-up.
Retrying only inside `start()` was not enough — observed failing after a full second
of retries on a freshly replugged camera, leaving the camera on default depth
settings with no indication. It now retries from `read()` against live frames until
it takes (5/5 runs, first attempt after the fix), and `visual_preset_applied` records
whether it ever did.

### Known issue: benchmark preview window hangs

`benchmark.py --preview` reliably hangs partway through a multi-pass run (third
preset), apparently when the camera is reopened between passes with a cv2 window
live. Preview is therefore **off by default** in benchmarks. Use
`bodytracker.py preview` to check framing first, then run the benchmark blind.
The live tracker's own `run --preview` opens the camera once and is unaffected.

## Camera and model settings — measured, not assumed (2026-08-04)

Re-run any of these with `benchmark.py {presets,body,modes,filters,models}`. Depth
benchmarks need only a static scene; `body` and `models` need a person in frame.

### Visual preset: MEDIUM_DENSITY

Measured on a real body, 5 s per preset:

| preset | detected | joints/frame | jitter | bone spread | ankles |
|---|---|---|---|---|---|
| HIGH_DENSITY | 97% | 12.4 | 7.6 mm | 55.3 mm | 58% |
| **MEDIUM_DENSITY** | **100%** | **14.3** | **5.1 mm** | **41.4 mm** | **96%** |
| HIGH_ACCURACY | 65% | 11.4 | 20.3 mm | 116.2 mm | 21% |

MEDIUM_DENSITY wins on every metric. Ankle availability 58% → 96% matters most,
since ankles are two of the three trackers we send.

**This is the opposite of what a static scene says.** On a textured desk,
HIGH_ACCURACY looks clearly best (3.61 mm noise vs 6.53 mm) and was nearly adopted on
that basis. A desk is nothing like a body: clothing and skin are low-texture, exactly
where HIGH_ACCURACY loses the coverage a pose model needs. **Benchmark on a person,
not on furniture.**

### Resolution / framerate: 848×480 @ 30 — higher is worse

| mode | coverage | noise | achieved fps |
|---|---|---|---|
| 1280×720@30 | 96.0% | 8.18 mm | 29.4 |
| **848×480@30** | **96.2%** | **5.48 mm** | **30.1** |
| 848×480@60 | 96.2% | 7.57 mm | 59.9 |
| 640×360@30 | 96.3% | 5.72 mm | 30.1 |

1280×720 carries 50% more depth noise at identical coverage — stereo matching gets
smaller correlation windows per pixel. 60 fps halves exposure and costs 38% more
noise. The default is a genuine optimum, not a compromise.

### Pose model: `full` — the most accurate one that stays real time

Measured with a person in frame:

| model | solve | p95 | achieved fps | holds 30 fps? |
|---|---|---|---|---|
| **full** | **16.69 ms** | 19.45 ms | **29.7** | **yes** |
| heavy | 51.15 ms | 56.77 ms | 15.0 | no |

`heavy` was briefly the default, on the strength of a benchmark that timed all three
models at ~11 ms. **That benchmark ran on blank frames**, where the detector finds
nothing and short-circuits before the landmark stage — so it measured the give-up
path, not tracking, and every model looked identical. With a body present, `heavy`
costs 3× more and halves the framerate.

Halving the update rate is worse for VRChat than any per-frame accuracy gain:
trackers arrive at 15 Hz and every stabilisation stage gets half the samples. So the
default is the most accurate model *subject to staying real time*. `--model heavy` is
there if you ever run this on hardware that can afford it.

**The general lesson, twice over now:** benchmark on a real body. The same mistake
picked the wrong visual preset (a static desk favours HIGH_ACCURACY, which is the
worst preset on an actual person) and the wrong model.

### GPU: unavailable, and not needed

```
CPU: OK  median 11.61 ms
GPU: UNAVAILABLE -- ImageCloneCalculator: GPU processing is disabled in build flags
```

AMD Barcelo iGPU, no NVIDIA. MediaPipe exposes a GPU delegate enum but the pip wheel
is built with GPU processing disabled; using it would mean building MediaPipe from
source. OpenCV reports no OpenCL, and librealsense acceleration is CUDA-only.

It does **not** degrade gracefully — requesting the GPU delegate raises — so the code
correctly never asks for it. Nor is it needed: camera-limited at 30 fps means a GPU
would buy idle time, not framerate.

### Preset application was silently failing

`visual_preset` goes over a USB extension unit busy during stream start-up. Retrying
only inside `start()` was not enough — observed failing after a full second of retries
on a freshly replugged camera, leaving the camera on default depth settings with no
indication. It now retries from `read()` against live frames until it takes (5/5 runs,
first attempt after the fix); `visual_preset_applied` records whether it ever did.

### Known issue: benchmark preview window hangs

`benchmark.py --preview` reliably hangs partway through a multi-pass run, apparently
when the camera is reopened between passes with a cv2 window live. Preview is
therefore **off by default** in benchmarks — check framing with
`bodytracker.py preview` first, then benchmark. The live tracker's `run --preview`
opens the camera once and is unaffected.

## RealSense feature audit (2026-08-04)

Every option the device exposes was enumerated and assessed, rather than assuming
the obvious ones were the only ones.

### Adopted

**Threshold filter, first in the chain** (0.3–4.0 m). Everything beyond that is wall
or another room. Letting it into the chain actively hurts: the spatial filter drags
far values across limb edges and hole filling pulls them into gaps on the body.
Cutting first means later stages only see plausible subject depth.

**Hole filling mode 2 (`nearest_from_around`), not the default 1
(`farthest_from_around`).** The default is actively wrong for this application: it
fills a hole with the *furthest* neighbour, so a gap on a torso or leg is filled with
the wall behind the person. The solver then either rejects that joint as background
(losing it) or believes it. We track a foreground subject, so the nearest neighbour is
the correct guess.

Measured paired on identical frames:

| variant | in-volume coverage | noise |
|---|---|---|
| farthest fill, no clipping (old) | 90.1% | 5.38 mm |
| nearest fill | 90.9% | 5.30 mm |
| **nearest + threshold** | **94.4%** | 5.34 mm |

+4.3 points of usable coverage at identical noise.

### Already correct, confirmed

- **Laser power 360 (max)** and emitter on — the single biggest coverage lever indoors.
- **Auto Exposure Priority = 0** on the RGB sensor. Had this been 1, auto-exposure
  would be permitted to *drop the framerate* in dim light. It defaults to 0 here.
- **Global Time Enabled = 1** — hardware timestamps in a common time domain.
- **Filter order** matches Intel's documented recommendation exactly
  (threshold → depth2disparity → spatial → temporal → disparity2depth → hole filling).

### Considered and rejected, with reasons

- **Frames Queue Size** (default 16). Measured frame age at queue sizes 16 / 4 / 1:
  35.4 / 33.7 / 35.8 ms — no difference, because the pipeline consumes every frame in
  real time and the queue never backs up. Left at default.
- **Depth Units** (0.001 m, settable to 1e-6). Finer units would only help if the
  hardware resolved finer than 1 mm. It does not: at 2.5 m with a 55 mm baseline and
  fx≈608, one subpixel disparity step is ≈5.8 mm. **That also explains the measured
  ~5–6.5 mm noise floor — it is disparity quantisation, not filtering.** No gain
  available here.
- **Decimation filter.** Changes resolution, which would break the pixel
  correspondence between the colour frame MediaPipe runs on and the depth frame.
- **HDR (`Hdr Enabled`, sequence options).** Alternates two exposures and merges them,
  which would improve depth in mixed lighting but halves the effective temporal
  resolution. We already established that halving the update rate costs more than
  per-frame quality gains.
- **Advanced Mode** is available and enabled on this device, exposing the full
  DepthControl parameter set. Not pursued: the visual presets *are* advanced-mode
  configurations, and MEDIUM_DENSITY was already chosen by measurement on a body.
  This is the obvious place to go for further depth tuning.
- **Manual RGB exposure.** Auto-exposure can reach 10 ms, which risks motion blur and
  therefore landmark error, but pinning it lower darkens the image and costs landmark
  confidence. Left on auto; revisit only if blur is observed.
- **`Emitter Enabled = 2`** (alternating on/off) exists for capturing clean IR without
  the projected pattern. No use here.
- **IMU.** The D415 has none (that is the D435i).

## Still to do

- [x] ~~Confirm VRChat accepts our OSC trackers~~ — all 8 indices, local desktop VRChat
- [x] ~~Depth post-processing, stabilisation, rotations, occlusion handling~~
- [ ] **Stand at 2.4–2.6 m** — the last recording was at 3.12 m, past the camera's 3 m
      ideal range. Closer should measurably reduce noise (untested: proving it needs two
      recordings of the *same* movement at two distances, not two different sessions)
- [ ] Point `bodytracker.py run <quest-ip>` at the Quest and confirm trackers in OSC Debug
- [ ] **Resolve index→role during a real FBT calibration** (only the Quest can)
- [ ] Confirm VRChat interprets our rotations as intended (`--no-rotations` if not)
