# bodytracker

Full-body tracking for **VRChat on Quest standalone** from an **Intel RealSense D415**.
No PC VR, no SteamVR, no Windows — the Linux box is a tracking server that sends
VRChat's OSC tracker messages over your LAN.

```
D415 ──► MediaPipe pose ──► depth lift ──► stabilise ──► OSC bundle ──► Quest:9000
        (2D landmarks)     (real metres)   (gate/bones/    (1 datagram      VRChat
                                            smooth/fill)    per frame)      FBT IK
```

## Install

**Use Python 3.12 on both platforms.** It is the highest version with a MediaPipe
wheel, and MediaPipe is the binding constraint.

### Windows

Everything is a pip wheel — no source build needed.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install pyrealsense2 mediapipe opencv-contrib-python python-osc numpy
.venv\Scripts\python bodytracker.py fetch
```

Windows resolves MediaPipe to **0.10.21** (1.0.0 ships no Windows wheel). That is
fine — this code uses the Tasks API, which is present and identical there; verified
against 0.10.21 explicitly.

### Linux

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install mediapipe opencv-contrib-python python-osc numpy
.venv/bin/python bodytracker.py fetch
```

`pyrealsense2` is the one exception: on Ubuntu 26.04 there is no apt package and no
wheel for this Python, so it must be built from source. `SETUP.md` documents the exact
build, including two flags without which it fails, and the udev rule the camera needs
in order to be usable without root.

## Quick start

One application, five subcommands:

```bash
# check framing first -- camera + overlay, sends nothing
.venv/bin/python bodytracker.py preview

# track and send to the Quest, WITH the live overlay window
.venv/bin/python bodytracker.py run 192.168.1.42 --preview

# same, headless
.venv/bin/python bodytracker.py run 192.168.1.42
```

| subcommand | what it does |
|---|---|
| `run [host]` | track and send OSC. `--preview` adds the overlay window in the same process |
| `preview` | camera + overlay only, sends nothing — for framing and diagnosis |
| `fake [host]` | synthetic trackers, no camera — isolates network problems from tracking ones |
| `listen` | watch VRChat's *outgoing* OSC on 9001, proving OSC is actually enabled |
| `measure` | record a session and replay it through each stabilisation config; `--save`/`--load` |

The preview draws the **stabilised skeleton re-projected through the camera
intrinsics** — i.e. exactly the data being sent over OSC, not the raw landmarks.
Joints held or reconstructed through an occlusion are drawn in **cyan and labelled
`held`**, so a limb being inferred never looks the same as one being measured.

## Using it with the Quest

1. **Give the Quest a static DHCP lease.** If its IP changes, the tracker sends into
   the void with no error — UDP has nothing to report.
2. **Enable OSC in VRChat**: Action Menu → Options → OSC → Enabled. VRChat does not
   open port 9000 until you do; nothing bound on 9000 means OSC is off, not broken.
3. **Set your real height** in VRChat settings. OSC tracker scaling depends on it.
4. **Stand back far enough that the camera sees you head to feet.**
   Verify with `bodytracker.py preview` — both `HEAD in frame` and `FEET in frame` green.
5. **Run** `bodytracker.py run <quest-ip> --preview`, then check Options → OSC →
   **OSC Debug** to confirm trackers are arriving.
6. **Calibrate**: Quick Menu → Launch Pad → Calibrate FBT, line the markers up with
   your body in a mirror, pull both triggers.

### Where to stand

**Use `bodytracker.py preview` and back up until both `HEAD in frame` and
`FEET in frame` are green.** That is the measurement; everything below is provisional reasoning that
has not been tested end to end.

Provisional, and flagged as such deliberately:

- The camera's own intrinsics give a 43.1° vertical FOV, implying ~2.28 m to frame a
  1.8 m person. The datasheet says 40°, implying 2.47 m. **Unresolved** — `fy/height`
  is identical across all four depth modes, so those are one calibration scaled by
  resolution, not four independent confirmations. `SETUP.md` describes a tape-measure
  test that settles it in five minutes.
- The D415's ideal range is 0.5–3 m, and stereo depth error grows with distance, so
  standing closer *should* be less noisy. **This has not been measured here.** The
  obvious-looking evidence (a 3.12 m recording being noisier than a 2.4 m one) came
  from two sessions differing in posture, movement and joint visibility as well as
  distance — the same unpaired-comparison mistake that once made depth filtering look
  4× worse than it is. Proving it needs the same movement recorded at two distances.

So: stand where the preview says your whole body is visible, prefer closer within
that, and treat the specific numbers as untested until someone runs the test.

The camera is mounted **upside down**, which the code assumes by default. Pass
`--upright` if you remount it.

## Options

```
bodytracker.py run [host] [--port 9000]
    --preview                            live overlay window, same process
    --roles hip,left_foot,right_foot     which trackers to send
    --upright                            camera mounted the right way up
    --no-head                            omit the space-alignment anchor
    --min-cutoff 0.5 --beta 0.35         smoothing (lower cutoff = smoother, laggier)
    --no-bones --no-gate --no-fill       disable a stabilisation stage (A/B testing)
    --no-rotations --no-filter           disable rotations / depth post-processing
```

Every `--no-*` switch exists so a stage can be measured against its own absence with
`bodytracker.py measure`, not as a workaround.

## Which trackers to send

Default is **hip + chest + both feet**. VRChat's docs warn that fewer trackers often
track better — but that is a warning about *reliability*, not a rule about count: its
IK compensates well for a **missing** point and badly for a **wrong** one. So the
roles were measured rather than guessed, over 446 frames of real movement:

| role | idx | measured | jitter | |
|---|---|---|---|---|
| hip | 1 | 100% | 4.2 mm | core |
| left_foot | 2 | 99% | 7.9 mm | standard FBT |
| right_foot | 3 | 100% | 4.8 mm | standard FBT |
| left_knee | 4 | 100% | 6.7 mm | optional |
| right_knee | 5 | 100% | 5.6 mm | optional |
| chest | 6 | 100% | **3.1 mm** | core — the steadiest point we produce |
| left_elbow | 7 | 100% | 5.5 mm | not recommended |
| right_elbow | 8 | 94% | 6.7 mm | not recommended |

Indices follow **SlimeVR's mapping**, which is the de-facto standard
(`VRCOSCHandler.kt`, carrying the comment *"Don't change as third party
applications may rely on this for mapping trackers to body parts"*).

- **Chest** is in by default on the numbers — it is our most stable tracker, and gives
  VRChat torso lean and twist that hip alone cannot.
- **Knees** track well and are one flag away
  (`--roles hip,chest,left_foot,right_foot,left_knee,right_knee`). They are not default
  because VRChat's IK already infers knee position from hip and foot, so they add
  little unless you kneel or sit often.
- **Elbows are not recommended.** Your hands come from the Quest controllers, so an
  elbow tracker constrains an arm whose endpoint is *already* authoritatively tracked.
  A slightly-wrong elbow fights the controller instead of helping. Available via
  `--roles` if you want to judge for yourself.

## What is verified, and what is not

**Verified on hardware:** camera setup, depth quality (coverage 90→96%, per-pixel
noise −51%), axis conversion and rotation signs, Euler maths (20 000 random rotations
round-trip to 2.6e-15), stabilisation (jitter −80%, worst glitch −75%, ankle
availability 71%→100%), 100% tracked over 1050 consecutive live frames, and that
VRChat accepts all 8 tracker indices along with our address format, port and payloads.

**A tracker, once seen, is sent forever.** This is deliberate, not a leak. OSC has no
"tracker removed" message, so staying silent does not retract a tracker — VRChat keeps
whatever it last received. Since a limb cannot be withdrawn, the only choice is what
value to keep sending. For up to 3 s a missing joint is reconstructed in the body's
frame so it moves with you; after that its last value persists and it is reported as
`[stale: left_foot]` in the status line and in the preview window. A stale foot looks correct in VRChat
while being wrong, so that report is the only way to notice.

**Not verified, and only the Quest can settle it:**

- **Index → role mapping — now resolved by convention, not by guessing.** VRChat's
  own docs never state one; they only list which roles exist. We follow SlimeVR's
  mapping (see above). VRChat itself probably ignores the index entirely: its docs say
  the system "should function similarly to our existing implementation for SteamVR
  trackers", and *Auto-center OSC Trackers* works by finding "the two lowest trackers
  on the y axis" and assuming they are feet — i.e. roles are inferred from **position
  at calibration**, not from the address number. Matching the convention costs nothing
  and is what other OSC tools expect.
- **Rotations.** The maths is proven correct in isolation and the values are stable
  on real recordings, but nothing has confirmed VRChat interprets them as intended.
  `--no-rotations` falls back to identity if they look wrong.

## Camera settings

Measured, not assumed — see `SETUP.md` for the tables:

- **MEDIUM_DENSITY** visual preset (ankle availability 96% vs 58% for HIGH_DENSITY)
- **848×480 @ 30 fps** — 1280×720 has 50% more depth noise at the same coverage
- **CPU inference**; no usable GPU path exists on this machine and none is needed
  (camera-limited at 30 fps with ~20 ms/frame spare)

`benchmark.py` re-runs any of these: `presets`, `body`, `modes`, `filters`, `models`.
The `body` and `models` benchmarks need a person in frame.

## Camera and model settings

Measured rather than assumed — full tables in `SETUP.md`:

- **MEDIUM_DENSITY** preset — ankle availability 96% vs 58% for HIGH_DENSITY, and it
  beats every alternative on every metric. Note a *static-scene* benchmark says the
  opposite; benchmark on a person.
- **848×480 @ 30 fps** — 1280×720 carries 50% more depth noise at identical coverage.
- **`full` pose model** — the most accurate that holds 30 fps. `heavy` is 3× slower
  with a body in frame (51 ms/frame → 15 fps), and halving the update rate costs more
  than its per-frame accuracy gains. `--model heavy` if your hardware can afford it.
- **CPU inference.** No usable GPU path exists here and none is needed — the pipeline
  is camera-limited at 30 fps with ~20 ms/frame spare.

- **Depth chain**: threshold clip (0.3–4 m) → disparity → spatial → temporal →
  disparity⁻¹ → hole fill (**nearest**, not the default farthest, which would fill
  gaps on your body with the wall behind you). +4.3 pts usable coverage.

`benchmark.py {presets,body,modes,filters,models}` re-runs any of it. `SETUP.md` also
records the RealSense options audited and deliberately *not* used, with reasons.

## Lighting: this needs a lit room

**It does not work in the dark**, despite being a depth camera. Landmarks are
found by MediaPipe in the **RGB** image; depth only supplies their 3D positions
afterwards. Kill the lights and detection finds nothing to place.

Depth itself is unaffected (the IR projector runs at full power), so the failure
is asymmetric and easy to misread: the depth stream looks perfectly healthy while
tracking has stopped completely. `bodytracker.py preview` shows `NO POSE
DETECTED` with a good depth image in `d` view — that combination means "too dark",
not "camera broken".

A depth-native tracker (Nuitrack, evaluated on a branch and dropped) would work in
the dark. It tracked noticeably worse in the light, which is why it was dropped —
but low-light operation is the one thing it clearly buys.

## Hands

Your controllers own the hands. VRChat's OSC tracker API has **no hand or wrist slot**
— the 8 are hip, chest, 2 feet, 2 knees, 2 elbows — so nothing here can conflict with
Quest controller or hand tracking. Wrists are tracked internally only to derive elbow
orientation and are never transmitted, and the default roles (hip + 2 feet) send
nothing that touches your arms at all.

See `RESEARCH.md` for the feasibility study and `SETUP.md` for the build record and
every measurement behind the numbers above.
