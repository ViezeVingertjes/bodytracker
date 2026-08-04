#!/usr/bin/env python3
"""Head-to-head 2D landmark quality: global-shutter webcam vs the D415's RGB.

Tests one claim and nothing else: does a global-shutter camera give better 2D
landmarks than the D415's rolling-shutter RGB? That is the half of the pipeline a
global shutter can plausibly improve, and it is worth settling before trading
away measured depth for it.

Deliberately depth-free and calibration-free. Everything is measured in the
image, in units of the subject's own shoulder width, so the two cameras are
comparable despite different resolutions, fields of view and distances:

    jitter      frame-to-frame landmark movement / shoulder width
    bone spread apparent limb-length variation / shoulder width
    detection   fraction of frames with a pose

Rolling shutter skews a frame during fast motion, so the difference should show
up as MOTION-dependent: run it once holding still, once moving briskly. If the
global shutter only wins while moving, that is the rolling-shutter artefact and
it is real. If it wins in both, the sensor is simply better. If it wins in
neither, the D415's rolling shutter is not costing anything at these speeds.

    python compare_cameras.py --seconds 20            # both, one after another
    python compare_cameras.py --seconds 20 --only uvc
"""

import argparse
import collections
import sys
import threading
import time

import numpy as np

import skeleton as S

MOVING_THRESHOLD = 0.02  # shoulder-widths/frame; above this the subject is moving

# Recorded D415 result, for comparison when that camera is no longer connected.
# From `diagnose --seconds 20` on colour: 542 frames, 100% tracked, lateral
# jitter 3.0 mm at a measured shoulder width of 0.35 m.
#
# 3.0 mm / 350 mm = 0.0086 shoulder-widths, which is directly comparable to what
# this tool measures: both are image-plane landmark motion normalised by the
# subject's own shoulder width, so resolution and distance cancel.
#
# NOT comparable: bone spread. The D415 figure (26.8 mm) is a 3D limb length
# using measured depth; this tool measures the 2D projected length, which
# changes as a limb rotates toward or away from the camera even when tracking is
# perfect. Only the jitter columns can be read across the two.
D415_REFERENCE = {
    "detection": 100.0,
    "jitter": 3.0 / 350.0,
    "note": "recorded earlier; D415 not connected now",
}


def find_webcam(preferred=None):
    """Locate the UVC camera by name, not by a fixed index.

    /dev/videoN numbering shifts whenever devices are plugged or unplugged, and
    each camera exposes several nodes of which only some can capture. Hardcoding
    an index produces "can't open camera by index" the moment anything changes.
    """
    import glob
    import subprocess
    if preferred is not None:
        return preferred
    candidates = []
    for node in sorted(glob.glob("/dev/video*")):
        try:
            info = subprocess.run(["udevadm", "info", "--query=property",
                                   f"--name={node}"], capture_output=True,
                                  text=True, timeout=5).stdout
        except Exception:  # noqa: BLE001
            continue
        if "ID_V4L_CAPABILITIES=:capture:" not in info:
            continue
        model = ""
        for line in info.splitlines():
            if line.startswith("ID_MODEL="):
                model = line.split("=", 1)[1]
        index = int(node.replace("/dev/video", ""))
        # Prefer anything that is not the laptop's built-in webcam.
        score = 0 if "ACER" in model.upper() or "HD_Webcam" in model else 1
        candidates.append((score, index, model))
    if not candidates:
        raise RuntimeError("no capture-capable video device found")
    candidates.sort(key=lambda c: (-c[0], c[1]))
    score, index, model = candidates[0]
    print(f"  using /dev/video{index} ({model})")
    return index


class ThreadedWebcam:
    """Always expose the newest frame.

    A blocking read costs a whole frame period (measured 35.6 ms), which would
    then serialise with the ~15 ms inference and halve throughput. Capturing on
    its own thread lets inference overlap the wait.
    """

    def __init__(self, index, width, height):
        import cv2
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.frame = None
        self.frames_captured = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
                self.frames_captured += 1

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


BONES_2D = [
    (S.L_SHOULDER, S.R_SHOULDER), (S.L_SHOULDER, S.L_ELBOW),
    (S.R_SHOULDER, S.R_ELBOW), (S.L_ELBOW, S.L_WRIST), (S.R_ELBOW, S.R_WRIST),
    (S.L_SHOULDER, S.L_HIP), (S.R_SHOULDER, S.R_HIP), (S.L_HIP, S.R_HIP),
    (S.L_HIP, S.L_KNEE), (S.R_HIP, S.R_KNEE),
    (S.L_KNEE, S.L_ANKLE), (S.R_KNEE, S.R_ANKLE),
]


def collect(get_frame, seconds, label, preview=True):
    """Run the pose model over frames, returning per-landmark pixel tracks.

    Shows a preview, because without one there is no way to know you are framed
    -- the first run of this tool measured a spurious 93px "body" and reported a
    40x regression that was pure noise.
    """
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=S.model_path(S.DEFAULT_MODEL)),
        running_mode=RunningMode.VIDEO, num_poses=1))

    tracks = collections.defaultdict(list)
    world = collections.defaultdict(list)
    shoulders = []
    seen = detected = 0
    timestamp = 0
    cv2 = None
    if preview:
        import overlay
        cv2 = overlay.require_cv2()
    print(f"  {label}: recording {seconds:.0f}s -- stand in frame", flush=True)
    start = time.monotonic()
    try:
        while time.monotonic() - start < seconds:
            frame = get_frame()
            if frame is None:
                continue
            seen += 1
            height, width = frame.shape[:2]
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            timestamp += 34
            result = landmarker.detect_for_video(
                Image(image_format=ImageFormat.SRGB, data=rgb), timestamp)

            if cv2 is not None:
                canvas = frame.copy()
                marks_p = result.pose_landmarks[0] if result.pose_landmarks else None
                width_px = 0.0
                if marks_p:
                    pts = {i: (int(marks_p[i].x * width), int(marks_p[i].y * height))
                           for i in S.NEEDED}
                    for a, b in BONES_2D:
                        cv2.line(canvas, pts[a], pts[b], (0, 220, 0), 2)
                    for pt in pts.values():
                        cv2.circle(canvas, pt, 4, (0, 220, 0), -1)
                    width_px = float(np.hypot(
                        (marks_p[S.L_SHOULDER].x - marks_p[S.R_SHOULDER].x) * width,
                        (marks_p[S.L_SHOULDER].y - marks_p[S.R_SHOULDER].y) * height))
                good = width_px >= MIN_SHOULDER_PX
                lines = [
                    (f"{label}", (255, 220, 0)),
                    (f"{seconds - (time.monotonic() - start):4.1f}s left", (255, 255, 255)),
                    (f"shoulder width {width_px:.0f}px "
                     f"(need >{MIN_SHOULDER_PX})",
                     (0, 220, 0) if good else (60, 60, 255)),
                    ("SAMPLE VALID -- move around" if good
                     else "TOO SMALL / NOT FRAMED -- step closer",
                     (0, 220, 0) if good else (60, 60, 255)),
                ]
                y = 30
                for text, colour in lines:
                    cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, colour, 2, cv2.LINE_AA)
                    y += 30
                shown = canvas if canvas.shape[1] <= 1280 else cv2.resize(
                    canvas, (1280, int(1280 * canvas.shape[0] / canvas.shape[1])))
                cv2.imshow("camera comparison", shown)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if not result.pose_landmarks:
                continue
            detected += 1
            marks = result.pose_landmarks[0]
            for idx in S.NEEDED:
                lm = marks[idx]
                tracks[idx].append((lm.x * width, lm.y * height))
            # MediaPipe's metric 3D estimate -- inferred from limb proportions,
            # NOT measured. This is what replaces depth when there is no depth
            # sensor, so its self-consistency is the whole question.
            if result.pose_world_landmarks:
                wl = result.pose_world_landmarks[0]
                for idx in S.NEEDED:
                    world[idx].append((wl[idx].x, wl[idx].y, wl[idx].z))

            left, right = marks[S.L_SHOULDER], marks[S.R_SHOULDER]
            shoulders.append(float(np.hypot((left.x - right.x) * width,
                                            (left.y - right.y) * height)))
    finally:
        landmarker.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    return tracks, world, shoulders, seen, detected


MIN_SHOULDER_PX = 140   # below this the "subject" is too small to be a real body


def world_metrics(world):
    """3D limb-length spread from MediaPipe's world landmarks, in millimetres.

    Directly comparable to the D415 figures, because both are 3D limb lengths in
    metres -- unlike the 2D projected lengths, which change as a limb rotates.

    A limb cannot change length. Whatever spread appears here is error in the
    inferred depth, which is the axis a depth camera was measuring.
    """
    bones = [(S.L_SHOULDER, S.L_ELBOW), (S.L_ELBOW, S.L_WRIST),
             (S.R_SHOULDER, S.R_ELBOW), (S.R_ELBOW, S.R_WRIST),
             (S.L_HIP, S.L_KNEE), (S.R_HIP, S.R_KNEE),
             (S.L_KNEE, S.L_ANKLE), (S.R_KNEE, S.R_ANKLE),
             (S.L_SHOULDER, S.R_SHOULDER), (S.L_HIP, S.R_HIP)]
    spreads, jitters = [], []
    for a, b in bones:
        if len(world[a]) < 30 or len(world[b]) < 30:
            continue
        n = min(len(world[a]), len(world[b]))
        lengths = np.linalg.norm(
            np.array(world[a][:n]) - np.array(world[b][:n]), axis=1)
        spreads.append((np.percentile(lengths, 84) - np.percentile(lengths, 16)) * 1000)
    for idx in S.NEEDED:
        if len(world[idx]) < 30:
            continue
        arr = np.array(world[idx])
        jitters.append(np.median(np.linalg.norm(np.diff(arr, axis=0), axis=1)) * 1000)
    if not spreads:
        return None
    return {"bone_spread_mm": float(np.median(spreads)),
            "joint_jitter_mm": float(np.median(jitters)) if jitters else float("nan")}


def analyse(tracks, shoulders, seen, detected):
    """Scale-free metrics, so different resolutions and distances compare."""
    if detected < 30:
        return None
    # Guard against measuring a spurious detection. A 93px shoulder width on a
    # 1920px frame is not a person standing for full-body tracking, and the
    # resulting numbers looked like a 40x regression that was really just noise
    # being normalised by a tiny scale.
    if float(np.median(shoulders)) < MIN_SHOULDER_PX:
        return {"invalid": f"shoulder width only {np.median(shoulders):.0f}px "
                           f"(need >{MIN_SHOULDER_PX}) -- stand closer / in frame"}
    scale = float(np.median(shoulders))  # pixels per shoulder width

    steps_all = []
    steps_still, steps_moving = [], []
    for series in tracks.values():
        arr = np.array(series)
        if len(arr) < 30:
            continue
        deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1) / scale
        # Split by whether the subject was moving: rolling-shutter skew is a
        # motion artefact, so lumping the two together would hide it.
        steps_all.extend(deltas)
        steps_still.extend(deltas[deltas <= MOVING_THRESHOLD])
        steps_moving.extend(deltas[deltas > MOVING_THRESHOLD])

    bones = [(S.L_SHOULDER, S.L_ELBOW), (S.L_ELBOW, S.L_WRIST),
             (S.R_SHOULDER, S.R_ELBOW), (S.R_ELBOW, S.R_WRIST),
             (S.L_HIP, S.L_KNEE), (S.R_HIP, S.R_KNEE)]
    spreads = []
    for a, b in bones:
        if len(tracks[a]) < 30 or len(tracks[b]) < 30:
            continue
        n = min(len(tracks[a]), len(tracks[b]))
        lengths = np.linalg.norm(
            np.array(tracks[a][:n]) - np.array(tracks[b][:n]), axis=1) / scale
        spreads.append(np.percentile(lengths, 84) - np.percentile(lengths, 16))

    return {
        "detection": 100.0 * detected / max(seen, 1),
        "jitter_all": float(np.median(steps_all)) if steps_all else float("nan"),
        "jitter_still": float(np.median(steps_still)) if steps_still else float("nan"),
        "jitter_moving": float(np.median(steps_moving)) if steps_moving else float("nan"),
        "moving_frac": 100.0 * len(steps_moving) / max(len(steps_still) + len(steps_moving), 1),
        "bone_spread": float(np.median(spreads)) if spreads else float("nan"),
        "scale_px": float(np.median(shoulders)),
    }


def run_uvc(args):
    cam = ThreadedWebcam(find_webcam(args.device), args.width, args.height)
    time.sleep(1.5)
    try:
        return collect(lambda: cam.frame, args.seconds, "global-shutter webcam",
                       args.preview)
    finally:
        cam.close()


def run_realsense(args):
    from capture import DepthCamera
    cam = DepthCamera(rotate_180=args.rotate_180)
    cam.start()
    try:
        for _ in range(15):
            cam.read()

        def frame():
            got = cam.read()
            return got[0] if got else None

        return collect(frame, args.seconds, "D415 RGB (rolling shutter)",
                       args.preview)
    finally:
        cam.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--device", type=int, default=None,
                    help="V4L2 index; auto-detected by name if omitted")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--only", choices=("uvc", "realsense"))
    ap.add_argument("--no-preview", dest="preview", action="store_false")
    ap.add_argument("--flip", dest="rotate_180", action="store_true",
                    help="camera is mounted upside down")
    ap.set_defaults(rotate_180=False, preview=True)
    args = ap.parse_args(argv)

    runners = [("global-shutter webcam", run_uvc), ("D415 RGB", run_realsense)]
    if args.only:
        runners = [r for r in runners
                   if (args.only == "uvc") == (r[0] == "global-shutter webcam")]

    results, worlds = {}, {}
    for name, runner in runners:
        try:
            tracks, world, shoulders, seen, detected = runner(args)
            results[name] = analyse(tracks, shoulders, seen, detected)
            worlds[name] = world_metrics(world)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: unavailable ({exc})")

    print()
    print(f"{'camera':<26}{'detect':>8}{'jitter all':>12}{'still':>10}{'moving':>10}")
    print(f"{'':<26}{'':>8}{'(shoulder widths -- lower is better)':>32}")
    print("-" * 66)
    for name, m in results.items():
        if not m:
            print(f"{name:<26}  too few detections")
            continue
        if "invalid" in m:
            print(f"{name:<26}  NOT MEASURABLE: {m['invalid']}")
            continue
        print(f"{name:<26}{m['detection']:7.0f}%{m['jitter_all']:12.4f}"
              f"{m['jitter_still']:10.4f}{m['jitter_moving']:10.4f}")
    ref = D415_REFERENCE
    print(f"{'D415 RGB (rolling)':<26}{ref['detection']:7.0f}%{ref['jitter']:12.4f}"
          f"{'--':>10}{'--':>10}   <- {ref['note']}")
    for name, m in results.items():
        if m and "invalid" not in m:
            print(f"  {name}: subject was {m['moving_frac']:.0f}% moving, "
                  f"shoulder width {m['scale_px']:.0f} px")

    live = [m for m in results.values() if m and "invalid" not in m]
    if live:
        best = min(live, key=lambda m: m["jitter_all"])
        ratio = best["jitter_all"] / D415_REFERENCE["jitter"]
        print()
        if ratio < 0.8:
            print(f"Global shutter is {1/ratio:.1f}x steadier than the D415's RGB. "
                  "The rolling-shutter cost is real.")
        elif ratio > 1.25:
            print(f"Global shutter is {ratio:.1f}x SHAKIER than the D415's RGB, "
                  "so it is not buying landmark quality here.")
        else:
            print("Comparable to the D415's RGB -- rolling shutter is not costing "
                  "measurable landmark quality at these speeds.")
        print("The D415 reference lumps still and moving together, so read the "
              "'jitter all' column against it.")
        print()
        print("Remember what the D415 also provided and this camera cannot: "
              "MEASURED depth (4.0mm),")
        print("30fps in this lighting (vs ~15), and its own IR illumination.")

    print()
    print("=== 3D from MediaPipe world landmarks (INFERRED depth, not measured) ===")
    print(f"{'source':<30}{'bone spread':>14}{'joint jitter':>15}")
    print("-" * 60)
    for name, m in worlds.items():
        if m:
            print(f"{name:<30}{m['bone_spread_mm']:12.1f}mm"
                  f"{m['joint_jitter_mm']:13.1f}mm")
    print(f"{'D415, measured depth (raw)':<30}{84.9:12.1f}mm{10.2:13.1f}mm")
    print(f"{'D415, measured + stabilised':<30}{26.8:12.1f}mm{4.7:13.1f}mm")
    print()
    print("A limb cannot change length, so all of the spread is depth error.")
    print("Lower than the D415 rows means the inferred Z is good enough to "
          "replace measured depth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
