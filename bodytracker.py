#!/usr/bin/env python3
"""bodytracker -- RealSense D415 full-body tracking for VRChat over OSC.

One entry point, five subcommands:

    bodytracker.py run [host]      track and send to VRChat  (add --preview)
    bodytracker.py preview         camera + overlay only, sends nothing
    bodytracker.py measure         record/replay a session, compare stabilisation
    bodytracker.py fake [host]     synthetic trackers, no camera -- tests the network
    bodytracker.py listen          watch VRChat's outgoing OSC, proves OSC is enabled
    bodytracker.py fetch           download the pose models (run this first)
    bodytracker.py diagnose        why joints get dropped -- drives what to fix
    bodytracker.py depth-estimators  which depth patch reducer is best

Before anything appears in VRChat you must, in VRChat:
  1. enable OSC          (desktop: hold R -> Options -> OSC -> Enabled)
  2. set your real height in settings -- OSC tracker scaling depends on it
  3. calibrate FBT       (Quick Menu -> Launch Pad -> Calibrate FBT)

Step 3 is impossible in desktop mode: VRChat hides all FBT settings without an
HMD. On a PC with no headset you will see trackers in OSC Debug and the avatar
will not move. That is expected, not a fault.
"""

import argparse
import collections
import math
import pathlib
import shutil
import sys
import time

import numpy as np

import overlay
from osc_out import VRCHAT_OSC_PORT, PoseSender, TrackerSender
from solver import (DEFAULT_MODEL, MODEL_DIR, MODELS, NEEDED, PoseSolver,
                    Skeleton, model_path)
from stabilize import ALL_BONES, SkeletonStabilizer
from transform import (
    DEFAULT_ROLES,
    TRACKER_ROLES,
    RotationPredictor,
    RotationSmoother,
    TrackerPredictor,
    build_trackers,
)

SEND_HZ = 30
# Seconds of continuous loss before reporting it. Short dropouts are normal even
# at high tracking rates, and announcing each one buries the real problems.
LOSS_REPORT_S = 0.75


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------

def add_camera_args(parser):
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--flip", dest="rotate_180", action="store_true",
                        help="camera is mounted upside down (rotates frames AND "
                             "intrinsics together)")
    parser.add_argument("--no-filter", dest="filtering", action="store_false",
                        help="disable RealSense depth post-processing")
    parser.add_argument("--ir-gain", type=int, default=None,
                        help="sensor gain in IR mode (16-248; default 128). "
                             "Higher = brighter IR, slightly noisier depth")
    parser.add_argument("--ir-mode", choices=("global", "masked", "clahe",
                                              "masked_clahe"),
                        default="masked",
                        help="IR contrast handling (default: masked). clahe "
                             "amplifies the projector pattern -- only use it "
                             "with --ir-emitter off")
    parser.add_argument("--ir-emitter", choices=("on", "off", "alternate"),
                        default="on",
                        help="projector during IR capture. 'off' gives a clean "
                             "image but needs another IR light source and costs "
                             "depth coverage")
    parser.add_argument("--source", choices=("color", "ir"), default="color",
                        help="image the pose model sees. 'ir' works in the dark "
                             "and needs no depth alignment")
    parser.set_defaults(rotate_180=False, filtering=True)


def add_model_args(parser):
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL,
                        help=f"pose model (default: {DEFAULT_MODEL}, most accurate)")


def add_stabilise_args(parser):
    parser.add_argument("--min-cutoff", type=float, default=0.5,
                        help="One Euro min cutoff: lower = smoother but laggier")
    parser.add_argument("--beta", type=float, default=0.35,
                        help="One Euro speed coefficient: higher = less lag when moving")
    parser.add_argument("--no-bones", action="store_true",
                        help="disable bone-length constraints")
    parser.add_argument("--no-gate", action="store_true",
                        help="disable outlier speed gating")
    parser.add_argument("--no-fill", action="store_true",
                        help="disable body-frame occlusion filling")


def make_stabilizer(args):
    return SkeletonStabilizer(
        args.min_cutoff, args.beta,
        enable_bones=not args.no_bones,
        enable_gate=not args.no_gate,
        enable_fill=not args.no_fill,
    )


def open_camera(args):
    # Imported here, not at module scope: capture.py raises a deliberate
    # ImportError when pyrealsense2 is missing, and pyrealsense2 is an optional
    # extra. At module scope that error fires on `import bodytracker`, taking
    # down `fake`, `listen` and `fetch` -- the subcommands that exist precisely
    # so you can make progress before the camera stack builds.
    from capture import DepthCamera

    return DepthCamera(args.width, args.height,
                       rotate_180=args.rotate_180, filtering=args.filtering,
                       source=getattr(args, "source", "color"),
                       ir_mode=getattr(args, "ir_mode", "masked"),
                       ir_emitter=getattr(args, "ir_emitter", "on"),
                       ir_gain=getattr(args, "ir_gain", None) or 128)


def parse_roles(parser, spec):
    roles = [r.strip() for r in spec.split(",") if r.strip()]
    unknown = [r for r in roles if r not in TRACKER_ROLES]
    if unknown:
        parser.error(f"unknown role(s): {', '.join(unknown)}. "
                     f"Valid: {', '.join(TRACKER_ROLES)}")
    return roles


# --------------------------------------------------------------------------
# run -- the actual tracker
# --------------------------------------------------------------------------

def cmd_run(args, parser, *, sending=True):
    roles = parse_roles(parser, args.roles)
    show = args.preview or not sending

    sender = TrackerSender(args.host, args.port) if sending else None
    stabilizer = make_stabilizer(args)
    rotation_smoother = RotationSmoother()
    send_hz = getattr(args, "send_hz", 0.0)
    # The sender's clock has to share the loop's timebase (seconds since t0),
    # or the horizon it extrapolates over is meaningless. t0 is not known until
    # the camera is open, so it is patched in below.
    clock_origin = [None]
    pose_sender = PoseSender(
        sender, TrackerPredictor(), RotationPredictor(),
        lead=args.lead_ms / 1000.0, hz=send_hz if sending else 0.0,
        clock=lambda: (time.monotonic() - clock_origin[0]
                       if clock_origin[0] is not None else 0.0),
        include_head=not args.no_head,
    )

    if sending:
        how = f"{send_hz:g} Hz" if send_hz > 0 else f"{SEND_HZ} Hz (per camera frame)"
        print(f"sending to {args.host}:{args.port} at {how}")
        for role in roles:
            print(f"  {role:<12} -> tracker index {TRACKER_ROLES[role]}")
        print("  (indices follow SlimeVR's convention -- see RESEARCH.md section 2)")
    else:
        print("preview only -- nothing is being sent")
    print("ctrl-c to stop" + ("   |   window: q quit, d depth view" if show else ""),
          flush=True)

    cv2 = overlay.require_cv2() if show else None
    period = 1.0 / SEND_HZ
    frames = sent = 0
    lost_since = None
    announced_loss = False
    show_depth = False
    fps, last_tick = 0.0, time.monotonic()

    # pose_sender last, so its thread is stopped FIRST on the way out -- before
    # the camera and solver it shares a loop with are torn down, and including
    # on ctrl-c. Between entering here and the first update() its predictors are
    # empty, so poses_at() yields nothing and TrackerSender sends no datagram.
    with open_camera(args) as cam, PoseSolver(model_path(args.model)) as solver, \
            pose_sender:
        for _ in range(10):
            cam.read()
        if cam.sensor_warnings:
            print(f"camera warnings: {cam.sensor_warnings}", flush=True)
        t0 = time.monotonic()
        clock_origin[0] = t0
        # Tracked separately from `fps`, which is the end-to-end pipeline rate
        # and so falls with a heavier model or a slower CPU -- feeding that to
        # health_warning() reports a hardware fault for what is a model choice.
        # `cam_fps` is how fast the camera itself delivers: when the pipeline is
        # the bottleneck, frames are already queued and read() returns at once,
        # so this stays high. Seeded at nominal so it cannot fire before it has
        # converged, and an EMA rather than a mean since t0 so that degradation
        # part-way into a session actually moves it.
        cam_fps = float(cam.fps)

        while True:
            loop_start = time.monotonic()
            frame = cam.read()
            # read() blocks only when the camera has nothing queued, so its
            # duration measures delivery, not our own throughput. Updated before
            # the None check: a dropped frame still cost us the wait.
            cam_fps = 0.9 * cam_fps + 0.1 / max(time.monotonic() - loop_start, 1e-6)
            if frame is None:
                continue
            color, depth_raw, _ = frame
            frames += 1
            drawable = None

            now = time.monotonic()
            t = now - t0
            skeleton = solver.solve(cam, color, depth_raw, t * 1000)

            if skeleton is None:
                if lost_since is None:
                    lost_since = now
                elif not announced_loss and now - lost_since > LOSS_REPORT_S:
                    announced_loss = True
                    # Say WHY when we know. "no pose" and "pose rejected as
                    # implausible" need completely different fixes.
                    why = solver.last_rejection
                    print(f"no pose -- {why}" if why
                          else "no pose -- are you in frame?", flush=True)
            else:
                if announced_loss:
                    print(f"pose reacquired after {now - lost_since:.1f}s", flush=True)
                lost_since = None
                announced_loss = False

                # Stabilise in CAMERA space, before the axis conversion -- bone
                # lengths and speeds are physical quantities, and constraining
                # them after the transform would mean reasoning in a mirrored
                # frame for no benefit.
                stable = Skeleton(
                    stabilizer(skeleton.joints, t), skeleton.visibility,
                    skeleton.pixels, skeleton.rejected, skeleton.body_depth,
                )
                drawable = stable
                trackers, rotations, head = build_trackers(
                    stable, roles, with_rotations=not args.no_rotations)
                pose_sender.update(trackers, rotation_smoother(rotations), head, t)

                # With a sender thread running, sending here as well would just
                # emit an extra frame at camera rate on top of its clock.
                if sending and send_hz <= 0:
                    pose_sender.send_once(t)
                sent += 1

            fps = 0.9 * fps + 0.1 / max(now - last_tick, 1e-6)
            last_tick = now

            if show:
                canvas = overlay.depth_colormap(depth_raw) if show_depth else color.copy()
                # Draw what is actually being SENT -- the stabilised skeleton,
                # re-projected through the camera intrinsics -- not the raw
                # landmarks. A preview of the raw solver output would show jitter
                # we already removed and hide joints we are inferring.
                overlay.draw_skeleton(canvas, drawable, camera=cam,
                                      inferred=stabilizer.inferred)
                extra = []
                margin = cam.light_margin()
                if margin:
                    exposure, exp_max, gain, gain_max, stops = margin
                    colour = overlay.GREEN if stops > 2 else overlay.AMBER
                    extra.append((f"light headroom {stops:.1f} stops "
                                  f"(exp {exposure:.0f}/{exp_max:.0f}us "
                                  f"gain {gain:.0f}/{gain_max:.0f})", colour))
                if sending:
                    stale = pose_sender.stale_keys(t)
                    names = {v: k for k, v in TRACKER_ROLES.items()}
                    live = list(pose_sender.poses_at(t)[0])
                    achieved = pose_sender.frames_sent / max(t, 1e-6)
                    extra.append((f"sending {len(live)} -> {args.host}:{args.port} "
                                  f"@ {achieved:.0f} Hz", overlay.CYAN))
                    if stale:
                        labels = sorted({names.get(k, str(k)) for k in stale})
                        extra.append((f"STALE: {', '.join(labels)}", overlay.AMBER))
                overlay.draw_lines(canvas, overlay.status_lines(
                    skeleton, fps, extra, frame_height=canvas.shape[0]))
                cv2.imshow("bodytracker", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    show_depth = not show_depth

            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

            if frames % 150 == 0:
                warning = cam.health_warning(cam_fps)
                if warning and not getattr(cmd_run, "_warned", False):
                    cmd_run._warned = True
                    print(f"WARNING: {warning}", flush=True)

            if frames % 150 == 0 and not show:
                rate = 100.0 * sent / max(frames, 1)
                # Stale trackers are still being SENT, so they are invisible
                # unless reported -- a foot not measured for seconds looks fine
                # in VRChat while being wrong.
                stale = pose_sender.stale_keys(t)
                names = {v: k for k, v in TRACKER_ROLES.items()}
                note = ""
                if stale:
                    note = f"  [stale: {', '.join(sorted(names.get(k, str(k)) for k in stale))}]"
                # ACHIEVED send rate, not the requested one. With a sender
                # thread these differ: it competes for the GIL with inference,
                # so --send-hz 90 can deliver appreciably less. Without this
                # number a by-feel comparison of send rates is measuring
                # something it cannot see.
                achieved = pose_sender.frames_sent / max(t, 1e-6)
                if pose_sender.send_errors:
                    note += (f"  [{pose_sender.send_errors} send errors, "
                             f"last: {pose_sender.last_error}]")
                print(f"{frames} frames, {sent} tracked ({rate:.0f}%), "
                      f"{achieved:.0f} Hz sent{note}", flush=True)

    if show:
        overlay.require_cv2().destroyAllWindows()


def cmd_preview(args, parser):
    args.host, args.port = "127.0.0.1", VRCHAT_OSC_PORT
    args.roles = ",".join(DEFAULT_ROLES)
    args.no_head = False
    args.no_rotations = False
    args.lead_ms = 0.0      # nothing is sent, so prediction would only confuse
    args.preview = True
    cmd_run(args, parser, sending=False)


# --------------------------------------------------------------------------
# fake -- synthetic trackers, no camera
# --------------------------------------------------------------------------

# All 8 trackers VRChat supports, placed where each role sits on a body. This is
# the COMPLETE set: there is no hand, wrist or head body tracker. Hands come from
# the controllers, and the head address is a space anchor, not a tracker.
#
# Indices are looked up in TRACKER_ROLES rather than written out again. They were
# written out once, in VRChat's prose ordering (chest=2, feet=3/4) -- the mapping
# transform.py explains is wrong relative to the whole ecosystem -- so `fake` and
# `run` placed DIFFERENT bodies. That defeats the entire point of this command:
# it exists to be the trustworthy reference when `run` looks wrong.
_BODY = [
    ("hip",         0.00, 1.00, 0.5, 0.05),
    ("chest",       0.00, 1.35, 0.7, 0.05),
    ("left_foot",  -0.15, 0.08, 0.9, 0.06),
    ("right_foot",  0.15, 0.08, 1.1, 0.06),
    ("left_knee",  -0.16, 0.50, 1.3, 0.06),
    ("right_knee",  0.16, 0.50, 1.5, 0.06),
    ("left_elbow", -0.35, 1.15, 1.7, 0.06),
    ("right_elbow", 0.35, 1.15, 1.9, 0.06),
]
ANATOMICAL = [(TRACKER_ROLES[role], role.replace("_", " "), x, y, freq, amp)
              for role, x, y, freq, amp in _BODY]


def probe_layout(indices):
    """Chosen indices as a compact, countable row.

    Design constraints learned the hard way: keep everything near chest height
    and within arm's reach in x, or markers fall outside the visible area in
    desktop mode and get miscounted as absent; stay clear of the head anchor at
    y=1.70; probe 4 at a time, not 8; and alternate the bob phase so neighbours
    move in opposition and read as two markers even when close together.
    """
    n = max(len(indices), 1)
    span = 0.25 * (n - 1)
    return [(idx, f"tracker {idx}", -span / 2 + 0.25 * k, 1.10,
             0.6 + 0.35 * k, 0.12, (k % 2) * math.pi)
            for k, idx in enumerate(indices)]


def cmd_fake(args, parser):
    if args.indices:
        try:
            indices = [int(s) for s in args.indices.split(",") if s.strip()]
        except ValueError:
            parser.error("--indices takes a comma-separated list, e.g. 1,2,3,4")
        if not indices or any(not 1 <= i <= 8 for i in indices):
            parser.error("indices must be in 1..8; VRChat supports at most 8 trackers")
        layout = probe_layout(indices)
    else:
        layout = [(*row, 0.0) for row in ANATOMICAL]

    sender = TrackerSender(args.host, args.port)
    print(f"sending to {args.host}:{args.port} at {SEND_HZ} Hz -- ctrl-c to stop")
    if not args.no_head:
        print("  head    anchor at y=1.70 (static, not a body tracker)")
    for index, label, x, y, freq, amp, phase in layout:
        print(f"  index {index}  {label:<12} x={x:+.2f} y={y:.2f} "
              f"bobbing {freq:.1f} Hz +/-{amp:.2f} m, "
              f"{'down-first' if phase else 'up-first'}")
    print(flush=True)

    period = 1.0 / SEND_HZ
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        poses = {index: (x, y + amp * math.sin(t * freq * 2 * math.pi + phase), 0.0)
                 for index, _label, x, y, freq, amp, phase in layout}
        sender.send_frame(poses, None if args.no_head else (0.0, 1.70, 0.0))
        time.sleep(period)


# --------------------------------------------------------------------------
# depth-estimators -- which patch reducer is best ON A BODY?
# --------------------------------------------------------------------------

def cmd_depth_estimators(args, _parser):
    """Compare depth patch reducers on a real body, in ONE recording.

    A static surface says the mean beats the median (1.00 -> 0.82 mm jitter,
    because a median returns one of the raw 1 mm-quantised values and so cannot
    resolve below the quantum). But a body is not a static surface: depth jitter
    on a body is ~4 mm, dominated by the landmark sliding across a curved limb
    rather than by quantisation. So the estimators are re-run here against the
    same frames, with a person in them, because that is the case we care about.

    All estimators see IDENTICAL frames -- re-recording per estimator would
    compare different movements, which is the mistake that made depth filtering
    look 4x worse than it is earlier in this project.
    """
    from solver import DEFAULT_DEPTH_ESTIMATOR, DEPTH_ESTIMATORS, reduce_depth

    cv2 = overlay.require_cv2() if args.preview else None
    print(f"Recording {args.seconds:.0f}s -- stand in frame, whole body "
          "visible, move naturally.", flush=True)

    # (joint -> list of per-frame sample arrays), captured once, replayed for all
    captured = collections.defaultdict(list)
    with open_camera(args) as cam, PoseSolver(model_path(args.model)) as solver:
        for _ in range(10):
            cam.read()
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            frame = cam.read()
            if frame is None:
                continue
            color, depth_raw, _ = frame
            sk = solver.solve(cam, color, depth_raw, (time.monotonic() - t0) * 1000)
            if cv2 is not None:
                canvas = color.copy()
                overlay.draw_skeleton(canvas, sk, camera=cam)
                overlay.draw_lines(canvas, [
                    ("DEPTH ESTIMATOR COMPARISON", overlay.CYAN),
                    (f"{args.seconds - (time.monotonic() - t0):4.1f}s left",
                     overlay.WHITE),
                    *overlay.status_lines(sk, 0.0,
                                          frame_height=canvas.shape[0])[1:],
                ])
                cv2.imshow("bodytracker depth estimators", canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            if sk is None:
                continue
            for idx, (px, py) in sk.pixels.items():
                if idx not in sk.joints:
                    continue
                samples = cam.depth_samples(depth_raw, px, py)
                if samples.size >= 8:
                    captured[idx].append(samples)

    if cv2 is not None:
        cv2.destroyAllWindows()

    usable = {k: v for k, v in captured.items() if len(v) >= 40}
    if not usable:
        print("Not enough data -- were you in frame?")
        return 1
    print(f"\n{sum(len(v) for v in usable.values())} joint-samples over "
          f"{len(usable)} joints\n")
    print(f"{'estimator':<12}{'depth jitter (median over joints)':>36}")
    print("-" * 50)
    results = {}
    for estimator in DEPTH_ESTIMATORS:
        per_joint = []
        for series in usable.values():
            values = np.array([reduce_depth(s, estimator) for s in series])
            per_joint.append(np.median(np.abs(np.diff(values))) * 1000)
        results[estimator] = float(np.median(per_joint))
        print(f"{estimator:<12}{results[estimator]:>32.2f}mm")
    best = min(results, key=results.get)
    # Read the default rather than restating it. Hardcoding "trimmed" here
    # meant this tool advised switching to the estimator already in force.
    current = DEFAULT_DEPTH_ESTIMATOR
    print(f"\nbest: {best} ({results[best]:.2f}mm)   current default: {current} "
          f"({results[current]:.2f}mm)")
    if best != current:
        print(f"-> change DEFAULT_DEPTH_ESTIMATOR in solver.py to {best!r}")
    else:
        print("-> current default is already the best of these")
    return 0


# --------------------------------------------------------------------------
# diagnose -- why are joints being dropped?
# --------------------------------------------------------------------------

def cmd_diagnose(args, _parser):
    """Count, per landmark, WHY it was dropped. Needs a person in frame.

    The solver already records a reason for every landmark it discards
    (`background` / `no depth` / `low visibility` / `off frame`), but nothing has
    ever aggregated them -- so which failure actually dominates has been a guess.
    The three reasons need three different, mutually exclusive fixes:

        background      the landmark pixel sits off the body, reading the wall.
                        Fixable with depth: snap it back onto the silhouette.
        no depth        a stereo hole ON the subject. A filter/preset problem,
                        not a landmark problem.
        low visibility  MediaPipe is not confident. Depth cannot help at all.

    Shows a preview so you can see that you are actually framed while it counts.
    """
    counts = collections.defaultdict(collections.Counter)
    present = collections.Counter()
    tracked = frames = 0
    # Error decomposition. Frame-to-frame deltas, not variance over the run:
    # real motion between two frames at 30 Hz is small next to measurement
    # noise, whereas run variance is dominated by you actually moving.
    prev_sample = {}
    depth_jitter = collections.defaultdict(list)
    lateral_jitter = collections.defaultdict(list)

    cv2 = overlay.require_cv2() if args.preview else None
    print(f"Recording {args.seconds:.0f}s -- stand in frame, whole body visible, "
          "and move naturally.", flush=True)

    with open_camera(args) as cam, PoseSolver(model_path(args.model)) as solver:
        for _ in range(10):
            cam.read()
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            frame = cam.read()
            if frame is None:
                continue
            color, depth_raw, _ = frame
            frames += 1
            sk = solver.solve(cam, color, depth_raw, (time.monotonic() - t0) * 1000)

            if cv2 is not None:
                canvas = color.copy()
                overlay.draw_skeleton(canvas, sk, camera=cam)
                remaining = args.seconds - (time.monotonic() - t0)
                overlay.draw_lines(canvas, [
                    ("DIAGNOSE -- why joints drop", overlay.CYAN),
                    (f"{remaining:4.1f}s left", overlay.WHITE),
                    *overlay.status_lines(sk, 0.0,
                                          frame_height=canvas.shape[0])[1:],
                ])
                cv2.imshow("bodytracker diagnose", canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if sk is None:
                continue
            tracked += 1
            intrinsics = cam.intrinsics
            for idx in NEEDED:
                if idx in sk.joints:
                    present[idx] += 1
                else:
                    counts[idx][sk.rejected.get(idx, "not found")] += 1

            for idx, point in sk.joints.items():
                pixel = sk.pixels.get(idx)
                if pixel is None or intrinsics is None:
                    continue
                if idx in prev_sample:
                    pz, ppx, ppy = prev_sample[idx]
                    depth_jitter[idx].append(abs(point[2] - pz))
                    # pixel delta -> metres at this joint's distance
                    lateral_jitter[idx].append(float(np.hypot(
                        (pixel[0] - ppx) / intrinsics.fx,
                        (pixel[1] - ppy) / intrinsics.fy)) * point[2])
                prev_sample[idx] = (point[2], pixel[0], pixel[1])

    if cv2 is not None:
        cv2.destroyAllWindows()

    print(f"\n{frames} frames, {tracked} with a pose "
          f"({100 * tracked / max(frames, 1):.0f}%)\n")
    if not tracked:
        print("No pose was ever detected -- nothing to diagnose. Were you in frame?")
        return 1

    print(f"{'joint':<12}{'present':>9}   why it was dropped")
    print("-" * 66)
    totals = collections.Counter()
    for idx in NEEDED:
        pct = 100 * present[idx] / tracked
        why = ", ".join(f"{k} {100 * v / tracked:.0f}%"
                        for k, v in counts[idx].most_common(3))
        totals.update(counts[idx])
        print(f"{overlay.NAMES.get(idx, idx):<12}{pct:8.0f}%   {why or '-'}")

    if totals:
        print("\nAggregate reason for every dropped joint:")
        total = sum(totals.values())
        for reason, count in totals.most_common():
            print(f"  {reason:<16}{100 * count / total:5.0f}%   ({count})")
    else:
        print("\nNo joints were dropped at all.")

    # Which error source dominates decides whether depth can help further.
    # Depth jitter is what a depth camera controls; lateral jitter is landmark
    # placement in the image, which depth cannot fix.
    rows = [(idx, np.median(depth_jitter[idx]) * 1000,
             np.median(lateral_jitter[idx]) * 1000)
            for idx in NEEDED
            if len(depth_jitter.get(idx, ())) >= 30]
    if rows:
        print(f"\n{'joint':<12}{'depth jitter':>14}{'lateral jitter':>16}"
              f"{'dominant':>10}")
        print("-" * 54)
        for idx, d, lat in rows:
            print(f"{overlay.NAMES.get(idx, idx):<12}{d:11.1f}mm{lat:14.1f}mm"
                  f"{'lateral' if lat > d else 'DEPTH':>10}")
        med_d = np.median([r[1] for r in rows])
        med_l = np.median([r[2] for r in rows])
        print("-" * 54)
        print(f"{'MEDIAN':<12}{med_d:11.1f}mm{med_l:14.1f}mm")
        print()
        if med_l > med_d * 1.3:
            print("Lateral (landmark placement) dominates -- depth is NOT the "
                  "bottleneck, and further depth work has little to win.")
        elif med_d > med_l * 1.3:
            print("Depth dominates -- depth-side improvements (limb-centre "
                  "fitting, better filtering) are worth building.")
        else:
            print("Comparable -- neither source dominates.")
    return 0


# --------------------------------------------------------------------------
# fetch -- download the pose models
# --------------------------------------------------------------------------

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")


def cmd_fetch(args, _parser):
    """Download the pose models. Pure stdlib, so it works everywhere Python does.

    This replaced a shell script, which only worked on Linux and macOS -- the
    models are needed on Windows too, and asking Windows users to find a bash is
    a poor first-run experience.
    """
    import urllib.request

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Say where: this is the repo's models/ from a checkout, but the user data
    # directory from a non-editable install, and a silent 45 MB download to an
    # unstated path is not something anyone should have to go looking for.
    print(f"into {MODEL_DIR}")
    wanted = MODELS if args.all else [args.model]
    for variant in wanted:
        target = pathlib.Path(model_path(variant))
        if target.exists() and not args.force:
            print(f"have  {target.name}")
            continue
        url = MODEL_URL.format(v=variant)
        print(f"fetch {target.name} ...", end="", flush=True)
        # Download to a temporary name and rename, so an interrupted download
        # never leaves a truncated file that looks valid to the next run.
        tmp = target.with_suffix(".partial")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, \
                    open(tmp, "wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp.replace(target)
            print(f" {target.stat().st_size / 1e6:.0f} MB")
        except Exception as exc:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            print(f" FAILED: {exc}")
            return 1
    return 0


# --------------------------------------------------------------------------
# listen -- confirm OSC is actually on
# --------------------------------------------------------------------------

def cmd_listen(args, _parser):
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import BlockingOSCUDPServer

    seen = set()

    def on_message(address, *osc_args):
        if address not in seen:
            seen.add(address)
            print(f"[new] {address:50s} {osc_args}", flush=True)

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(on_message, needs_reply_address=False)
    server = BlockingOSCUDPServer((args.bind, args.port), dispatcher)
    print(f"listening on {args.bind}:{args.port} -- ctrl-c to stop")
    print("(VRChat SENDS on 9001 when OSC is enabled; it RECEIVES on 9000)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped -- {len(seen)} distinct addresses seen")


# --------------------------------------------------------------------------
# measure -- stability comparison
# --------------------------------------------------------------------------

def record_session(args, seconds):
    frames = []
    with open_camera(args) as cam, PoseSolver(model_path(args.model)) as solver:
        for _ in range(15):
            cam.read()
        print(f"  preset applied: {cam.visual_preset_applied}"
              + (f"  warnings: {cam.sensor_warnings}" if cam.sensor_warnings else ""))
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            r = cam.read()
            if not r:
                continue
            t = time.monotonic() - t0
            sk = solver.solve(cam, r[0], r[1], t * 1000)
            if sk is not None:
                frames.append((t, {k: v.copy() for k, v in sk.joints.items()}))
    return frames


def evaluate(frames, stabilizer=None):
    prev = prev_raw = None
    steps, jumps, lags = [], [], []
    bone_lengths = collections.defaultdict(list)

    for t, joints in frames:
        current = stabilizer(joints, t) if stabilizer else joints
        # Lag proxy: how far the stabilised joint trails the raw one while the
        # raw one is genuinely moving. Measured only during motion, because
        # while standing still "trailing" is exactly the noise rejection wanted.
        if prev_raw is not None:
            for idx, p in current.items():
                if (idx in joints and idx in prev_raw
                        and float(np.linalg.norm(joints[idx] - prev_raw[idx])) > 0.01):
                    lags.append(float(np.linalg.norm(p - joints[idx])) * 1000)
        prev_raw = joints

        if prev is not None:
            for idx, p in current.items():
                if idx in prev:
                    d = float(np.linalg.norm(p - prev[idx])) * 1000
                    steps.append(d)
                    jumps.append(d)
        for a, b in ALL_BONES:
            if a in current and b in current:
                bone_lengths[(a, b)].append(
                    float(np.linalg.norm(current[a] - current[b])))
        prev = current

    spreads = [(np.percentile(v, 84) - np.percentile(v, 16)) * 1000
               for v in bone_lengths.values() if len(v) > 20]
    nan = float("nan")
    return {
        "jitter": float(np.median(steps)) if steps else nan,
        "lag": float(np.median(lags)) if lags else nan,
        "bone_spread": float(np.median(spreads)) if spreads else nan,
        "worst_jump": float(np.max(jumps)) if jumps else nan,
    }


def cmd_measure(args, _parser):
    if args.load:
        # allow_pickle is required for the object array of per-frame joint dicts.
        # Only ever point --load at a file this tool wrote with --save; a .npz
        # from an untrusted source can execute arbitrary code on load.
        data = np.load(args.load, allow_pickle=True)
        frames = [(float(t), {int(k): np.asarray(v) for k, v in dict(j).items()})
                  for t, j in zip(data["times"], data["joints"], strict=True)]
        print(f"Replaying {args.load}")
    else:
        print(f"Recording {args.seconds:.0f}s -- stand in frame and move naturally, "
              "then hold still for the last few seconds.")
        frames = record_session(args, args.seconds)
        if args.save:
            np.savez(args.save,
                     times=np.array([t for t, _ in frames]),
                     joints=np.array([j for _, j in frames], dtype=object))
            print(f"  saved to {args.save}")
    print(f"  {len(frames)} tracked frames\n")
    if len(frames) < 30:
        print("Not enough tracked frames to measure. Are you in frame?")
        return

    configs = [
        ("raw (no stabilisation)", None),
        ("gate only", SkeletonStabilizer(enable_bones=False, enable_fill=False)),
        ("bones only", SkeletonStabilizer(enable_gate=False, enable_fill=False)),
        ("full stabilisation", SkeletonStabilizer()),
    ]
    print(f"{'configuration':<26}{'jitter':>9}{'lag':>9}{'bone spread':>13}{'worst jump':>12}")
    print("-" * 69)
    for label, stab in configs:
        m = evaluate(frames, stab)
        print(f"{label:<26}{m['jitter']:7.1f}mm{m['lag']:7.1f}mm"
              f"{m['bone_spread']:11.1f}mm{m['worst_jump']:10.1f}mm")


# --------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="bodytracker",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="track and send to VRChat")
    run.add_argument("host", nargs="?", default="127.0.0.1",
                     help="where VRChat is listening (default: this machine)")
    run.add_argument("--port", type=int, default=VRCHAT_OSC_PORT)
    run.add_argument("--preview", action="store_true",
                     help="also show the live overlay window")
    run.add_argument("--roles", default=",".join(DEFAULT_ROLES),
                     help=f"comma-separated, from: {', '.join(TRACKER_ROLES)}")
    run.add_argument("--no-head", action="store_true",
                     help="omit the head anchor VRChat uses to align tracking space")
    run.add_argument("--no-rotations", action="store_true",
                     help="send identity rotations instead of derived ones")
    run.add_argument("--lead-ms", type=float, default=50.0,
                     help="extrapolate this far ahead to cancel pipeline latency "
                          "(measured ~50ms; 0 disables)")
    run.add_argument("--send-hz", type=float, default=0.0,
                     help="send rate. 0 (default) sends once per camera frame; "
                          "a positive value runs a sender thread at that rate, "
                          "filling the gaps between camera frames that a "
                          "72-90Hz headset would otherwise hold. Try 90")
    add_camera_args(run)
    add_model_args(run)
    add_stabilise_args(run)
    run.set_defaults(func=cmd_run)

    prev = sub.add_parser("preview", help="camera + overlay only, sends nothing")
    add_camera_args(prev)
    add_model_args(prev)
    add_stabilise_args(prev)
    prev.set_defaults(func=cmd_preview)

    fake = sub.add_parser("fake", help="synthetic trackers, no camera needed")
    fake.add_argument("host", nargs="?", default="127.0.0.1")
    fake.add_argument("--port", type=int, default=VRCHAT_OSC_PORT)
    fake.add_argument("--indices", metavar="LIST",
                      help="probe specific indices as a countable row, e.g. 1,2,3,4")
    fake.add_argument("--no-head", action="store_true")
    fake.set_defaults(func=cmd_fake)

    diag = sub.add_parser("diagnose", help="why joints get dropped")
    diag.add_argument("--seconds", type=float, default=20)
    diag.add_argument("--no-preview", dest="preview", action="store_false")
    add_camera_args(diag)
    add_model_args(diag)
    diag.set_defaults(func=cmd_diagnose, preview=True)

    est = sub.add_parser("depth-estimators",
                         help="compare depth patch reducers on a real body")
    est.add_argument("--seconds", type=float, default=20)
    est.add_argument("--no-preview", dest="preview", action="store_false")
    add_camera_args(est)
    add_model_args(est)
    est.set_defaults(func=cmd_depth_estimators, preview=True)

    fetch = sub.add_parser("fetch", help="download the pose models")
    fetch.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    fetch.add_argument("--all", action="store_true", help="fetch every variant")
    fetch.add_argument("--force", action="store_true", help="re-download if present")
    fetch.set_defaults(func=cmd_fetch)

    listen = sub.add_parser("listen", help="watch VRChat's outgoing OSC")
    listen.add_argument("--port", type=int, default=9001)
    listen.add_argument("--bind", default="0.0.0.0")
    listen.set_defaults(func=cmd_listen)

    measure = sub.add_parser("measure", help="compare stabilisation configurations")
    measure.add_argument("--seconds", type=float, default=15)
    measure.add_argument("--save", help="write the recording to an .npz for reuse")
    measure.add_argument("--load", help="replay a saved recording instead of capturing")
    add_camera_args(measure)
    add_model_args(measure)
    measure.set_defaults(func=cmd_measure)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        args.func(args, parser)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
