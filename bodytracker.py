#!/usr/bin/env python3
"""bodytracker -- RealSense D415 full-body tracking for VRChat over OSC.

One entry point, five subcommands:

    bodytracker.py run [host]      track and send to VRChat  (add --preview)
    bodytracker.py preview         camera + overlay only, sends nothing
    bodytracker.py measure         record/replay a session, compare stabilisation
    bodytracker.py fake [host]     synthetic trackers, no camera -- tests the network
    bodytracker.py listen          watch VRChat's outgoing OSC, proves OSC is enabled

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
import sys
import time

import numpy as np

import overlay
from capture import DepthCamera
from osc_out import VRCHAT_OSC_PORT, TrackerSender
from solver import DEFAULT_MODEL, MODELS, PoseSolver, Skeleton, model_path
from stabilize import ALL_BONES, SkeletonStabilizer
from transform import (
    DEFAULT_ROLES,
    TRACKER_ROLES,
    RotationSmoother,
    TrackerLatch,
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
    parser.add_argument("--upright", dest="rotate_180", action="store_false",
                        help="camera mounted the right way up (default: upside down)")
    parser.add_argument("--no-filter", dest="filtering", action="store_false",
                        help="disable RealSense depth post-processing")
    parser.set_defaults(rotate_180=True, filtering=True)


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
    return DepthCamera(args.width, args.height,
                       rotate_180=args.rotate_180, filtering=args.filtering)


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
    latch, rotation_latch = TrackerLatch(), TrackerLatch()
    rotation_smoother = RotationSmoother()

    if sending:
        print(f"sending to {args.host}:{args.port} at {SEND_HZ} Hz")
        for role in roles:
            print(f"  {role:<12} -> tracker index {TRACKER_ROLES[role]}")
        print("  (index->role mapping is unconfirmed -- see RESEARCH.md section 2)")
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

    with open_camera(args) as cam, PoseSolver(model_path(args.model)) as solver:
        for _ in range(10):
            cam.read()
        if cam.sensor_warnings:
            print(f"camera warnings: {cam.sensor_warnings}", flush=True)
        t0 = time.monotonic()

        while True:
            loop_start = time.monotonic()
            frame = cam.read()
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
                latch.update(trackers, t)
                rotation_latch.update(rotation_smoother(rotations), t)
                if head is not None:
                    latch.update({"head": head}, t)

                if sending:
                    # Emit EVERY tracker seen so far, every frame. OSC is
                    # stateless and VRChat has no "tracker removed" message, so
                    # going quiet does not retract a tracker -- it freezes it
                    # wherever it last was.
                    held = rotation_latch.all_current()
                    poses, head_pose = {}, None
                    for key, position in latch.all_current().items():
                        rotation = held.get(key, (0.0, 0.0, 0.0))
                        if key == "head":
                            if not args.no_head:
                                head_pose = (position, rotation)
                        else:
                            poses[key] = (position, rotation)
                    sender.send_frame(poses, head_pose)
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
                if sending:
                    stale = set(latch.stale_keys(t)) | set(rotation_latch.stale_keys(t))
                    names = {v: k for k, v in TRACKER_ROLES.items()}
                    live = [names.get(k, str(k)) for k in latch.all_current()
                            if k != "head"]
                    extra.append((f"sending {len(live)} -> {args.host}:{args.port}",
                                  overlay.CYAN))
                    if stale:
                        labels = sorted({names.get(k, str(k)) for k in stale})
                        extra.append((f"STALE: {', '.join(labels)}", overlay.AMBER))
                overlay.draw_lines(canvas, overlay.status_lines(skeleton, fps, extra))
                cv2.imshow("bodytracker", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    show_depth = not show_depth

            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

            if frames % 150 == 0 and not show:
                rate = 100.0 * sent / max(frames, 1)
                # Stale trackers are still being SENT, so they are invisible
                # unless reported -- a foot not measured for seconds looks fine
                # in VRChat while being wrong.
                stale = set(latch.stale_keys(t)) | set(rotation_latch.stale_keys(t))
                names = {v: k for k, v in TRACKER_ROLES.items()}
                note = ""
                if stale:
                    note = f"  [stale: {', '.join(sorted(names.get(k, str(k)) for k in stale))}]"
                print(f"{frames} frames, {sent} sent ({rate:.0f}% tracked){note}",
                      flush=True)

    if show:
        overlay.require_cv2().destroyAllWindows()


def cmd_preview(args, parser):
    args.host, args.port = "127.0.0.1", VRCHAT_OSC_PORT
    args.roles = ",".join(DEFAULT_ROLES)
    args.no_head = False
    args.no_rotations = False
    args.preview = True
    cmd_run(args, parser, sending=False)


# --------------------------------------------------------------------------
# fake -- synthetic trackers, no camera
# --------------------------------------------------------------------------

# All 8 trackers VRChat supports, placed where each role sits on a body. This is
# the COMPLETE set: there is no hand, wrist or head body tracker. Hands come from
# the controllers, and the head address is a space anchor, not a tracker.
ANATOMICAL = [
    (1, "hip", 0.00, 1.00, 0.5, 0.05),
    (2, "chest", 0.00, 1.35, 0.7, 0.05),
    (3, "left foot", -0.15, 0.08, 0.9, 0.06),
    (4, "right foot", 0.15, 0.08, 1.1, 0.06),
    (5, "left knee", -0.16, 0.50, 1.3, 0.06),
    (6, "right knee", 0.16, 0.50, 1.5, 0.06),
    (7, "left elbow", -0.35, 1.15, 1.7, 0.06),
    (8, "right elbow", 0.35, 1.15, 1.9, 0.06),
]


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
