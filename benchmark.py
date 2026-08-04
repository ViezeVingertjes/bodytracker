#!/usr/bin/env python3
"""Measure the camera and model settings, instead of assuming them.

Answers, with numbers rather than reasoning:

  presets     which RealSense visual preset gives the best depth for a body
  modes       resolution/framerate: what each costs and what it buys
  models      pose_landmarker lite vs full vs heavy -- speed against stability
  filters     what the post-processing chain is actually worth

Depth metrics need only a static scene. The `models` benchmark needs a PERSON in
frame, because inference cost and landmark stability both depend on whether there
is anything to track -- timing an empty room measures the detector's give-up path,
not the tracking path.

    .venv/bin/python benchmark.py all
    .venv/bin/python benchmark.py presets modes
"""

import argparse
import sys
import time

import numpy as np
import pyrealsense2 as rs

import overlay

from capture import DepthCamera
from solver import PoseSolver, model_path
from stabilize import ALL_BONES


def depth_stats(cam, frames=60, warmup=25):
    """Coverage and per-pixel temporal noise on whatever the camera is pointed at."""
    for _ in range(warmup):
        cam.read()
    stack = []
    for _ in range(frames):
        r = cam.read()
        if r:
            stack.append(r[1].astype(float) * cam.depth_scale)
    a = np.stack(stack)
    valid = a > 0
    always = valid.all(axis=0)
    stds = a[:, always].std(axis=0) if always.any() else np.array([0.0])
    return {
        "coverage": 100.0 * valid.mean(),
        "noise_mm": float(np.median(stds)) * 1000,
        "stable_px": int(always.sum()),
    }


def bench_presets(args):
    presets = [
        ("HIGH_DENSITY", rs.rs400_visual_preset.high_density),
        ("MEDIUM_DENSITY", rs.rs400_visual_preset.medium_density),
        ("HIGH_ACCURACY", rs.rs400_visual_preset.high_accuracy),
        ("DEFAULT", rs.rs400_visual_preset.default),
    ]
    print("\n=== visual presets (static scene) ===")
    print(f"{'preset':<16}{'coverage':>10}{'noise':>10}{'stable px':>12}")
    print("-" * 48)
    for name, preset in presets:
        cam = DepthCamera(rotate_180=args.rotate_180, filtering=True)
        cam.start()
        try:
            sensor = (cam._pipeline.get_active_profile()
                      .get_device().first_depth_sensor())
            # The preset control is busy during stream start-up; retry briefly.
            for _ in range(30):
                try:
                    sensor.set_option(rs.option.visual_preset, float(preset))
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(0.05)
            s = depth_stats(cam)
            print(f"{name:<16}{s['coverage']:9.1f}%{s['noise_mm']:8.2f}mm"
                  f"{s['stable_px']:12d}")
        finally:
            cam.stop()


def bench_modes(args):
    modes = [(1280, 720, 30), (848, 480, 30), (848, 480, 60), (640, 360, 30)]
    print("\n=== resolution / framerate ===")
    print(f"{'mode':<16}{'coverage':>10}{'noise':>10}{'capture ms':>12}{'achieved fps':>14}")
    print("-" * 62)
    for w, h, fps in modes:
        cam = DepthCamera(w, h, fps, rotate_180=args.rotate_180, filtering=True)
        try:
            cam.start()
        except Exception as exc:  # noqa: BLE001
            print(f"{f'{w}x{h}@{fps}':<16}  unsupported: {exc}")
            continue
        try:
            s = depth_stats(cam)
            t0 = time.perf_counter()
            n = 0
            times = []
            while n < 90:
                a = time.perf_counter()
                if cam.read():
                    times.append(time.perf_counter() - a)
                    n += 1
            wall = time.perf_counter() - t0
            print(f"{f'{w}x{h}@{fps}':<16}{s['coverage']:9.1f}%{s['noise_mm']:8.2f}mm"
                  f"{np.median(times) * 1000:11.2f}{n / wall:14.1f}")
        finally:
            cam.stop()


def bench_filters(args):
    print("\n=== depth post-processing ===")
    print(f"{'chain':<16}{'coverage':>10}{'noise':>10}")
    print("-" * 36)
    for label, on in (("off", False), ("on", True)):
        cam = DepthCamera(rotate_180=args.rotate_180, filtering=on)
        cam.start()
        try:
            s = depth_stats(cam)
            print(f"{label:<16}{s['coverage']:9.1f}%{s['noise_mm']:8.2f}mm")
        finally:
            cam.stop()


def bench_models(args):
    """Needs a person in frame -- see module docstring."""
    variants = ["lite", "full", "heavy"]
    print("\n=== pose model (STAND IN FRAME -- each run is "
          f"{args.seconds:.0f}s) ===")
    print(f"{'model':<10}{'inference':>12}{'detected':>11}{'jitter':>10}"
          f"{'bone spread':>13}{'max fps':>10}")
    print("-" * 68)
    for variant in variants:
        path = model_path(variant)
        cam = DepthCamera(rotate_180=args.rotate_180, filtering=True)
        cam.start()
        try:
            solver = PoseSolver(model_path=path)
        except FileNotFoundError:
            cam.stop()
            print(f"{variant:<10}  missing {path}")
            continue
        try:
            for _ in range(15):
                cam.read()
            cv2 = overlay.require_cv2() if args.preview else None
            if cv2 is not None:
                print(f"  {variant}: get in frame", flush=True)
            times, frames_seen, tracked = [], 0, []
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.seconds:
                r = cam.read()
                if not r:
                    continue
                frames_seen += 1
                a = time.perf_counter()
                sk = solver.solve(cam, r[0], r[1], (time.monotonic() - t0) * 1000)
                times.append(time.perf_counter() - a)
                if cv2 is not None:
                    canvas = r[0].copy()
                    overlay.draw_skeleton(canvas, sk, camera=cam)
                    overlay.draw_lines(canvas, [
                        (f"BENCHMARK  model={variant}", overlay.CYAN),
                        (f"{args.seconds - (time.monotonic() - t0):4.1f}s left",
                         overlay.WHITE),
                        *overlay.status_lines(sk, 0.0, frame_height=canvas.shape[0])[1:],
                    ])
                    cv2.imshow("bodytracker benchmark", canvas)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        raise KeyboardInterrupt
                if sk is not None:
                    tracked.append({k: v.copy() for k, v in sk.joints.items()})

            infer = np.median(times) * 1000
            det = 100.0 * len(tracked) / max(frames_seen, 1)
            steps, spreads = [], {}
            prev = None
            for joints in tracked:
                if prev:
                    for idx, p in joints.items():
                        if idx in prev:
                            steps.append(float(np.linalg.norm(p - prev[idx])) * 1000)
                for a_, b_ in ALL_BONES:
                    if a_ in joints and b_ in joints:
                        spreads.setdefault((a_, b_), []).append(
                            float(np.linalg.norm(joints[a_] - joints[b_])))
                prev = joints
            sp = [(np.percentile(v, 84) - np.percentile(v, 16)) * 1000
                  for v in spreads.values() if len(v) > 20]
            jit = np.median(steps) if steps else float("nan")
            bs = np.median(sp) if sp else float("nan")
            print(f"{variant:<10}{infer:10.2f}ms{det:10.0f}%{jit:9.1f}mm"
                  f"{bs:11.1f}mm{1000 / max(infer, 1e-6):10.1f}")
        finally:
            solver.close()
            cam.stop()


def bench_body(args):
    """Presets measured on a PERSON, which is the only test that decides this.

    The static-scene `presets` benchmark is misleading in both directions:

      - coverage on a textured desk says nothing about coverage on low-texture
        clothing and skin, which is the hard case for stereo and exactly where
        HIGH_ACCURACY is expected to lose most
      - temporal noise on a static scene is measured against a temporal filter
        that has converged, so it flatters filtering enormously

    So this measures what we actually care about: how many joints survive, and
    how steady they are, with a body in frame.
    """
    presets = [
        ("HIGH_DENSITY", rs.rs400_visual_preset.high_density),
        ("MEDIUM_DENSITY", rs.rs400_visual_preset.medium_density),
        ("HIGH_ACCURACY", rs.rs400_visual_preset.high_accuracy),
    ]
    print(f"\n=== presets ON A BODY (stand in frame, {args.seconds:.0f}s each) ===")
    print(f"{'preset':<16}{'detected':>10}{'joints/frame':>14}{'jitter':>10}"
          f"{'bone spread':>13}{'ankles':>9}")
    print("-" * 72)
    import skeleton as S
    for name, preset in presets:
        cam = DepthCamera(rotate_180=args.rotate_180, filtering=True)
        cam.start()
        solver = PoseSolver()
        try:
            sensor = (cam._pipeline.get_active_profile()
                      .get_device().first_depth_sensor())
            for _ in range(40):
                try:
                    sensor.set_option(rs.option.visual_preset, float(preset))
                    break
                except Exception:  # noqa: BLE001
                    cam.read()
            for _ in range(20):
                cam.read()

            # A benchmark that needs a person in frame is useless without a way
            # to see whether the person IS in frame.
            cv2 = overlay.require_cv2() if args.preview else None
            if cv2 is not None:
                print(f"  {name}: get in frame -- the window shows what it sees",
                      flush=True)

            seen, tracked, joint_counts, ankles = 0, [], [], 0
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.seconds:
                r = cam.read()
                if not r:
                    continue
                seen += 1
                sk = solver.solve(cam, r[0], r[1], (time.monotonic() - t0) * 1000)

                if cv2 is not None:
                    canvas = r[0].copy()
                    overlay.draw_skeleton(canvas, sk, camera=cam)
                    remaining = args.seconds - (time.monotonic() - t0)
                    overlay.draw_lines(canvas, [
                        (f"BENCHMARK  {name}", overlay.CYAN),
                        (f"{remaining:4.1f}s left   move around naturally",
                         overlay.WHITE),
                        *overlay.status_lines(sk, 0.0, frame_height=canvas.shape[0])[1:],
                    ])
                    cv2.imshow("bodytracker benchmark", canvas)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        raise KeyboardInterrupt

                if sk is None:
                    continue
                tracked.append({k: v.copy() for k, v in sk.joints.items()})
                joint_counts.append(len(sk.joints))
                ankles += sum(1 for a in (S.L_ANKLE, S.R_ANKLE) if a in sk.joints)

            steps, spreads = [], {}
            prev = None
            for joints in tracked:
                if prev:
                    for idx, p in joints.items():
                        if idx in prev:
                            steps.append(float(np.linalg.norm(p - prev[idx])) * 1000)
                for a_, b_ in ALL_BONES:
                    if a_ in joints and b_ in joints:
                        spreads.setdefault((a_, b_), []).append(
                            float(np.linalg.norm(joints[a_] - joints[b_])))
                prev = joints
            sp = [(np.percentile(v, 84) - np.percentile(v, 16)) * 1000
                  for v in spreads.values() if len(v) > 20]
            n = max(len(tracked), 1)
            print(f"{name:<16}{100 * len(tracked) / max(seen, 1):9.0f}%"
                  f"{np.mean(joint_counts) if joint_counts else 0:14.1f}"
                  f"{np.median(steps) if steps else float('nan'):9.1f}mm"
                  f"{np.median(sp) if sp else float('nan'):11.1f}mm"
                  f"{100 * ankles / (2 * n):8.0f}%")
        finally:
            solver.close()
            cam.stop()


BENCHMARKS = {
    "presets": bench_presets,
    "body": bench_body,
    "modes": bench_modes,
    "filters": bench_filters,
    "models": bench_models,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", default=["all"],
                    help=f"any of: {', '.join(BENCHMARKS)}, or 'all'")
    ap.add_argument("--seconds", type=float, default=12,
                    help="duration per model in the `models` benchmark")
    ap.add_argument("--flip", dest="rotate_180", action="store_true",
                    help="camera is mounted upside down")
    # Off by default: the overlay window hangs when the camera is reopened
    # between benchmark passes (reproducible on the third preset). Benchmarks
    # do not need it, and a hang mid-measurement is worse than no picture --
    # use `bodytracker.py preview` to check framing first, then benchmark.
    ap.add_argument("--preview", action="store_true",
                    help="show the overlay window (known to hang between passes)")
    ap.set_defaults(rotate_180=False, preview=False)
    args = ap.parse_args(argv)

    names = list(BENCHMARKS) if "all" in args.which else args.which
    unknown = [n for n in names if n not in BENCHMARKS]
    if unknown:
        ap.error(f"unknown benchmark(s): {', '.join(unknown)}")
    try:
        for name in names:
            BENCHMARKS[name](args)
    finally:
        if args.preview and overlay._cv2 is not None:
            overlay._cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
