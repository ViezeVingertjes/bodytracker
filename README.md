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

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install python-osc numpy mediapipe
./fetch-models.sh                     # pose models, 44 MB, not in git
```

`pyrealsense2` must be built from source on Ubuntu 26.04 — there is no apt package
and no wheel for this Python. `SETUP.md` documents the exact build, including two
flags without which it fails.

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

Default is **hip + 2 feet** — the standard 3-point FBT set. VRChat's own docs note
that fewer trackers often track better, because its IK compensates well for a missing
point and badly for a wrong one, and a single camera's knee/elbow estimates are its
least reliable. `--roles` can send up to all 8.

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

- **Index → role mapping.** VRChat's spec lists *which roles exist* but never says
  index 1 is the hip. Desktop VRChat accepts all 8 indices under either rule, so the
  test cannot discriminate. If calibration puts trackers on the wrong body parts,
  the mapping in `transform.TRACKER_ROLES` is what to change.
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
- **`heavy` pose model** by default (most accurate). `--model full` / `--model lite`
  step down on slower machines.
- **CPU inference.** No usable GPU path exists here and none is needed — the pipeline
  is camera-limited at 30 fps with ~20 ms/frame spare.

`benchmark.py {presets,body,modes,filters,models}` re-runs any of it.

## Hands

Your controllers own the hands. VRChat's OSC tracker API has **no hand or wrist slot**
— the 8 are hip, chest, 2 feet, 2 knees, 2 elbows — so nothing here can conflict with
Quest controller or hand tracking. Wrists are tracked internally only to derive elbow
orientation and are never transmitted, and the default roles (hip + 2 feet) send
nothing that touches your arms at all.

See `RESEARCH.md` for the feasibility study and `SETUP.md` for the build record and
every measurement behind the numbers above.
