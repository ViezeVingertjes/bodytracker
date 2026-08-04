# Intel RealSense D415 → VRChat full-body tracking (Quest standalone)

Feasibility study + setup plan. Researched 2026-08-04.

**Target setup (confirmed with user):**
- Camera: Intel RealSense D415 (already owned)
- Tracking host: this machine — Ubuntu 26.04, kernel 7.0
- VRChat: **Quest standalone**, no PC in the loop, no SteamVR

---

## 1. Verdict

**Yes, this works — but nothing off-the-shelf does it. You have to write the bridge.**

The chain is:

```
D415 (depth+RGB)  →  skeleton solver  →  OSC sender  →  Quest:9000  →  VRChat OSC trackers
   USB 3.0            Ubuntu box         UDP over LAN                   FBT IK
```

Every link exists and is documented. The middle two are yours to build; there is no
published RealSense→VRChat project (searched: GitHub topics `full-body-tracking`,
VRChat OSC community tools — everything camera-based is MediaPipe/RGB or SteamVR-driver
based, and the SteamVR ones are useless to you on Quest standalone).

Two pieces of good news for this specific configuration:

- **Quest standalone is the *easiest* VRChat target, not the hardest.** VRChat's OSC
  tracker API is network-transparent. You send UDP to the headset's IP. No SteamVR
  driver, no Windows, no Proton, no `Driver4VR` (Windows-only — irrelevant here).
  The Ubuntu box just becomes a tracking server on the LAN.
- **The D415 is genuinely the right RealSense for skeletons.** See §3.

The bad news is in §7 — read it before committing a weekend.

---

## 2. What VRChat expects (the contract you're coding against)

From VRChat's OSC tracker spec:

| Item | Value |
|---|---|
| Address | `/tracking/trackers/{1..8}/position`, `/tracking/trackers/{1..8}/rotation` |
| Head anchor | `/tracking/trackers/head/position`, `/tracking/trackers/head/rotation` |
| Payload | 3 floats (Vector3) — world-space metres, or euler degrees |
| Coord system | Unity: **left-handed, +Y up**, `1.0 = 1 metre` |
| Euler order | applied Z, X, Y |
| Port | **9000** on the Quest (VRChat receives on 9000, sends on 9001) |
| Roles | hip, chest, 2x feet, 2x knees, 2x elbows (upper arms) — 8 max |

**Partly verified. All 8 indices work; the role mapping is still open.**

*Tested 2026-08-04 against desktop VRChat on this machine (Proton, `--no-vr`).* Sent
indices 1–4, then 5–8, as a compact chest-height row. **All four appeared in OSC Debug in
both halves — so VRChat accepts and renders all 8 indices, and does not filter by index.**

But that does **not** answer the role question. VRChat's spec says verbatim only
*"Currently up to 8 trackers are supported: hip, chest, 2x feet, 2x knees, 2x elbows
(upper arms)"* — it never states that index 1 is the hip. Two readings remain live: a
fixed index→role mapping, or roles assigned during FBT calibration from where you place
the balls (the way SteamVR trackers work). **VRChat would accept and draw all 8 indices
under either rule**, so the acceptance test cannot discriminate between them.

Role assignment only manifests during calibration, and desktop mode cannot calibrate
(see §6). **This needs the Quest**, at step 4 of §8. Until then, do not assume a numbering.

Testing note for whoever repeats this: probe **4 indices at a time, not 8**, and keep them
at chest height in a ~±0.4 m row with alternating bob phase. A first attempt used a
vertical stack of 8 spanning y=0.2–1.6, which put index 8 within 4 cm of the head anchor
at y=1.70; the two markers merged and the count read as 7. Markers that are spread too
widely also fall outside the visible area in desktop mode and read as absent.

Two behaviours worth internalising:

- **The head tracker is the alignment mechanism, not a body tracker.** You publish where
  *your* system thinks the head is; VRChat shifts your entire tracking space each frame so
  that point lands on the avatar's head bone. Yaw is lerped over ~10s. This means your
  camera space does **not** need to agree with the Quest's room-scale space — the head
  anchor reconciles them. It also means: **if your head estimate is noisy, your whole body
  jitters.** Prioritise a stable head joint.
- **Fewer trackers beat more.** VRChat's own docs say so. Send **hip + 2 feet** (indices
  1, 3, 4) and let VRChat's IK solve knees and elbows. A single-viewpoint depth camera's
  knee/elbow estimates are the least reliable joints you have; feeding them in will look
  worse than omitting them.

---

## 3. Is the D415 the right camera? Mostly yes — with one hard constraint

**The good.** Counter-intuitively, the D415 is the *better* 400-series camera for skeleton
tracking. Nuitrack's own community guidance says the D435 "is NOT a good sensor for doing
skeletal tracking" — its wide-FOV stereo has high RMS depth error past ~1 m, which
compounds into unstable joints. The D415's narrow FOV gives it the highest depth quality
per degree in the series. You own the right one.

**The constraint: field of view sets your minimum room depth.**

*Updated 2026-08-04. Two numbers disagree and the disagreement is not yet settled —
plan against the conservative one.*

| Source | Depth V-FOV | Distance for a 1.8 m person |
|---|---|---|
| Datasheet (nominal, ±1°) | 40° | **2.47 m** ← plan against this |
| This unit's calibrated intrinsics | 43.1° | 2.28 m |

The camera's own intrinsics report ~70° H × 43.1° V. Tempting to treat that as
authoritative over the datasheet, but it is **one** measurement, not four: `fy/height`
is exactly 1.26746 in every depth mode (1280×720, 848×480, 640×480, 640×360), so all
modes are one calibration scaled by resolution. Intrinsics also describe the imager's
optical FOV, whereas the datasheet's 65°×40° plausibly describes the *usable* depth FOV
after stereo overlap and disparity search discard the margins — in which case both
numbers are right and the datasheet is the one that matters to us.

**Settle it with a tape measure, not another query.** Tape a metre rule vertically, stand
it at exactly 2.00 m from the lens, and look at the depth stream in `realsense-viewer`:

```
vertical coverage at 2.0 m:   1.46 m if 40° is right      1.58 m if 43.1° is right
```

That 12 cm difference is easy to read off a metre rule, and it decides a 20 cm difference
in the room requirement.

Meanwhile: **camera at ~2.5 m, mounted at ~1.0 m height** (waist/chest, so the body is
vertically centred). Ideal range remains 0.5–3 m, so 2.5 m still sits inside it.

- You need **~2.5 m of clear floor depth** between camera and user, plus space behind the
  camera. Verify in the actual play space — the most likely dealbreaker.
- **Use 848×480 or 1280×720, not 640×480.** The 4:3 mode throws away 14° of horizontal
  FOV for nothing. 848×480 is also Nuitrack's recommended size.
- Raising your arms overhead will clip out of frame. Fine for FBT (you don't track hands
  from the camera — the Quest controllers do that), but it means the head joint can drop
  out if you crouch or jump.
- Tilting the camera up slightly buys a little headroom at the cost of floor visibility.
  Prefer keeping it level and centred on the torso.

**Rolling shutter.** The D415 uses rolling shutter (the D435 is global). Fast lateral
motion — a quick dodge or a dance move — will skew the depth frame. It won't break
tracking, but it caps how well this handles fast movement. Slow-to-moderate motion is
where this rig lives.

---

## 4. Choosing the skeleton solver

This is the one real design decision. Two viable paths:

### Option A — Nuitrack (paid, proven, fastest to working)

3DiVi's Nuitrack is the de-facto skeletal middleware for RealSense. It explicitly lists
the D415 as a supported sensor, runs on Linux x64, and is actively maintained (v0.38.5,
July 2025). It gives you a 19-joint skeleton with per-joint orientation — which matters,
because VRChat wants *rotation* as well as position, and deriving stable joint rotations
yourself from point clouds is the hard part of this project.

- **Cost:** $49.99/yr per sensor (Nuitrack AI subscription). Perpetual licences are bound
  to the sensor serial number.
- **Trial:** free, but **stops after 3 minutes of running** and must be restarted — and it
  is bound to the *PC*, not the sensor. Three minutes is enough to see a skeleton and enough
  to get trackers into VRChat's OSC Debug, but FBT calibration will very likely time out
  mid-flow. Expect to restart the tracker a few times while calibrating; that's friction,
  not a blocker.
- **Python:** official `py_nuitrack` wheel in `3DiVi/nuitrack-sdk` (PythonNuitrack-beta).
  Good enough to prototype the whole bridge in Python.
- **Risk:** a paid, closed dependency in the middle of your pipeline; licence is tied to
  that specific D415.

### Option B — MediaPipe Pose + D415 depth (free, more code)

Run BlazePose on the D415's RGB stream, then use the aligned depth frame to replace
MediaPipe's guessed Z with **measured** Z per landmark. This is strictly better than the
existing RGB-only community tools (`ju1ce/Mediapipe-VR-Fullbody-Tracking`,
`Alpyg/vrc_osc_tracker`), whose known weakness is exactly the depth axis — and it's the
reason owning a depth camera is worth anything here.

- **Cost:** free. No licence, no serial binding.
- **Work:** you must derive joint rotations yourself (hip yaw from the shoulder/hip vector,
  foot direction from ankle→toe), and do your own temporal filtering (One-Euro filter is
  the usual choice for this).
- **Specific risk — will it see you *wearing the Quest*?** BlazePose's person detector uses
  the face as its ROI prior, and you'll be wearing a large headset with most of your face
  occluded. Nuitrack works off depth silhouette and is far less exposed to this. Test this
  before committing to Option B (see §8 step 1) — it is the failure mode that would kill
  the free path outright.
- **Upside:** it's a real, interesting project and you own all of it — which is presumably
  why this directory exists.

**Recommendation: prototype against Option A's trial to prove the end-to-end chain works
(camera → OSC → Quest → avatar moving), then decide whether to pay the $50/yr or invest
the time in Option B.** Proving the chain first means that if §7's occlusion problem
sinks the idea, you've spent an evening and no money.

---

## 5. Setup — Ubuntu 26.04 side

> **This section is the plan as researched. It has since been executed — see `SETUP.md`
> for what is actually installed on this machine and the three places reality differed
> from the plan below.**

### 5.1 librealsense on kernel 7.0

Do **not** bother with the kernel-patching path (`patch-realsense-ubuntu-lts-hwe.sh`).
The upstream script tracks specific LTS kernels (5.15/6.5/6.8/6.11/6.14) and there are
open issues for it failing on kernels well below yours; kernel 7.0 is far outside what it
knows. DKMS builds are failing for users on much older kernels than this.

Use the **userspace RSUSB backend** instead — it bypasses the kernel UVC driver entirely
via libuvc and is fully supported for the D400 series:

```bash
sudo apt install -y git cmake build-essential libssl-dev libusb-1.0-0-dev \
    pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev

git clone https://github.com/realsenseai/librealsense.git
cd librealsense
sudo ./scripts/setup_udev_rules.sh
mkdir build && cd build
cmake .. -DFORCE_RSUSB_BACKEND=true -DBUILD_EXAMPLES=true \
         -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=true
make -j$(nproc)
sudo make install
```

**Python gotcha (this will bite):** `sudo make install` puts `pyrealsense2` into system
dist-packages, but Ubuntu 26.04's system Python is externally-managed (PEP 668), so you'll
be working in a venv for `python-osc` / `mediapipe` — and `pyrealsense2` won't be visible
from inside it. Create the venv as:

```bash
python3 -m venv --system-site-packages .venv
```

(or export `PYTHONPATH` to the install dir).

Note the repo: **`realsenseai/librealsense`**. RealSense spun out of Intel in 2025;
`IntelRealSense/librealsense` and `intelrealsense.com` are legacy (the latter no longer
resolves). Current docs live at `dev.realsenseai.com`.

### 5.2 Verify the camera before writing any code

```bash
realsense-viewer
```

Confirm: device detected, **USB 3.0** (not 2.1 — a USB 2 link silently caps you to
unusable framerates; use the cable that came with it and a blue/SS port), depth stream
live at 1280×720. Update firmware from the viewer if it's old — Nuitrack requires D415
firmware **5.8.15 or newer**.

Stand at 2.5 m and check in the viewer that your head and feet are both in frame. This
is the §3 constraint, tested in five minutes, before anything else.

### 5.3 Nuitrack (if Option A)

Install the Linux x64 SDK from `3DiVi/nuitrack-sdk`, run `nuitrack_license_tool` to
activate the trial against the D415's serial, then verify with the bundled
`nuitrack_sample` that you get a tracked skeleton. Then `pip install py_nuitrack.whl`
plus `opencv-python`.

### 5.4 The bridge (what you actually write)

```
bodytracker/
  capture.py     # librealsense pipeline: aligned depth+color, 640x480 or 848x480 @30
  solver.py      # Nuitrack skeleton  OR  MediaPipe+depth landmark lift
  transform.py   # camera space → Unity space, joint orientations, One-Euro filtering
  osc_out.py     # python-osc UDP client → QUEST_IP:9000
  bodytracker.py <- single CLI: run / preview / fake / listen / measure
```

`pip install python-osc`. The transform step is where the bugs will be — camera space is
right-handed with +Z forward from the lens; Unity is left-handed with +Y up. Get one axis
sign wrong and the avatar mirrors or inverts, which is the classic symptom.

Send at **30 Hz** (matching the camera) — VRChat does not need or want faster, and the
head anchor applies per-frame with no smoothing, so unsmoothed data reads as jitter.

---

## 6. Setup — Quest / VRChat side

1. **Find the Quest's LAN IP:** Quick Settings → Wi-Fi → network details. Give it a DHCP
   reservation on your router; if it changes, your bridge silently sends into the void.
2. **Enable OSC in VRChat:** hold **Y** (left controller) → Options → **OSC** → Enabled.
3. **Verify data is arriving:** same menu → **OSC Debug**. Your trackers should appear
   here before you attempt calibration. If they don't, it's a network/IP/port problem, not
   a tracking problem — check the Quest and the Ubuntu box are on the same subnet and that
   AP/client isolation is off on the router (this bites people on mesh and guest networks).
4. **Set your real height** in VRChat's settings — OSC tracker scaling depends on it.
5. **Calibrate FBT:** go to a world with a full-length mirror, Quick Menu → Launch Pad →
   **Calibrate FBT**. Avatar goes T-pose with white balls for trackers. Line the balls up
   with the corresponding body parts, then pull **both triggers**.
6. If trackers are offset from the start, use IK settings → **Auto-Center OSC Trackers**
   (it centres the two lowest trackers' midpoint under your head; clicking repeatedly
   alternates the guessed forward direction).

---

## 7. Honest risks — read before investing

**Occlusion is the fundamental limitation, and it is not fixable in software.** One camera
sees one side of you. Turn 90° and your far leg is hidden behind your near leg; turn 180°
and the solver is guessing at everything. IMU-based systems (SlimeVR) don't care which way
you face — that's precisely why they won the FBT market. If you dance, turn frequently, or
socialise in a circle, a single-camera rig will frustrate you. If you mostly face one
direction (sitting, standing, casual conversation, streaming to a fixed camera), it's fine.

Secondary risks, in rough order of likelihood:

- **Room depth** — 3 m of clear floor is a real requirement (§3). Check first.
- **WiFi latency and jitter** — OSC is UDP over your LAN to a headset. Wired-backhaul AP
  and 5 GHz help; congested 2.4 GHz will show up as visible tracker lag.
- **Rolling shutter** caps fast motion quality (§3).
- **Nuitrack licence binding** to the sensor serial, if you go Option A and later replace
  the camera.

**The comparison you deserve:** a SlimeVR 5-point set is ~$200–300, has no occlusion
problem, no room-size requirement, and works out of the box with Quest standalone over
this exact same OSC path. This project is worth doing because you already own the D415 and
because building it is interesting — not because it will beat SlimeVR on tracking quality.
It won't.

---

## 8. Suggested order of work

1. `realsense-viewer` — confirm USB 3.0, update firmware, **stand at 2.5 m and check
   framing**. (Kill criterion: if the room can't do it, stop here.) While you're stood
   there, do the Option B de-risk in the same session: run MediaPipe Pose on the RGB stream
   **while wearing the Quest**, and see whether it still finds you.
2. Hard-code a fake tracker at a fixed position, send it to `QUEST_IP:9000`, and see it in
   VRChat's OSC Debug. Proves the network path with zero tracking code.
3. Nuitrack trial → skeleton on screen (remember: 3-minute sessions).
4. Wire skeleton → OSC, hip + 2 feet only, **positions with identity rotations** — you
   don't need correct orientations to answer "does the avatar move with me," and rotation
   derivation is the fiddly part. Calibrate, look in a mirror, and use this run to settle
   the index→role question from §2.
5. Then, and only then, worry about rotations, filtering, and whether to go Option B.

---

## Sources

- [VRChat — OSC Trackers](https://docs.vrchat.com/docs/osc-trackers)
- [VRChat — OSC Overview](https://docs.vrchat.com/docs/osc-overview)
- [vrchat-community/vrc-oscquery-lib — osc-trackers.md](https://github.com/vrchat-community/vrc-oscquery-lib/blob/main/osc-trackers.md)
- [VRChat — Full-Body Tracking](https://docs.vrchat.com/docs/full-body-tracking)
- [SlimeVR Docs — OSC / VRChat on Quest](https://docs.slimevr.dev/server/osc-information.html)
- [Nuitrack SDK](https://nuitrack.com/) · [3DiVi/nuitrack-sdk](https://github.com/3DiVi/nuitrack-sdk) · [PyNuitrack](https://github.com/3DiVi/nuitrack-sdk/blob/master/PythonNuitrack-beta/README.MD)
- [Nuitrack licensing doc](https://github.com/3DiVi/nuitrack-sdk/blob/master/doc/Licensing.md) · [Nuitrack FAQ (trial limits)](https://nuitrack.com/faq)
- [Nuitrack community — D415 vs D435 skeleton stability](https://community.nuitrack.com/t/nuitracks-skeleton-tracking-stability-on-d415-and-d435-which-one-is-better/823)
- [RealSense D415 product page](https://www.realsenseai.com/products/stereo-depth-camera-d415/) · [D400 series datasheet](https://www.mouser.com/pdfdocs/Intel_D400_Series_Datasheet.pdf)
- [realsenseai/librealsense](https://github.com/realsenseai/librealsense) · [Linux install from source](https://dev.realsenseai.com/installation/linux-ubuntu-installation-from-source/)
- [ju1ce/Mediapipe-VR-Fullbody-Tracking](https://github.com/ju1ce/Mediapipe-VR-Fullbody-Tracking) · [Alpyg/vrc_osc_tracker](https://github.com/Alpyg/vrc_osc_tracker)
- [Driver4VR](https://www.driver4vr.com/) (Windows/SteamVR only — not applicable to Quest standalone)
