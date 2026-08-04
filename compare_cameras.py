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

import solver as S

MOVING_THRESHOLD = 0.02  # shoulder-widths/frame; above this the subject is moving


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


def collect(get_frame, seconds, label):
    """Run the pose model over frames, returning per-landmark pixel tracks."""
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
    shoulders = []
    seen = detected = 0
    timestamp = 0
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
            if not result.pose_landmarks:
                continue
            detected += 1
            marks = result.pose_landmarks[0]
            for idx in S.NEEDED:
                lm = marks[idx]
                tracks[idx].append((lm.x * width, lm.y * height))
            left, right = marks[S.L_SHOULDER], marks[S.R_SHOULDER]
            shoulders.append(float(np.hypot((left.x - right.x) * width,
                                            (left.y - right.y) * height)))
    finally:
        landmarker.close()
    return tracks, shoulders, seen, detected


def analyse(tracks, shoulders, seen, detected):
    """Scale-free metrics, so different resolutions and distances compare."""
    if detected < 30:
        return None
    scale = float(np.median(shoulders))  # pixels per shoulder width

    steps_still, steps_moving = [], []
    for series in tracks.values():
        arr = np.array(series)
        if len(arr) < 30:
            continue
        deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1) / scale
        # Split by whether the subject was moving: rolling-shutter skew is a
        # motion artefact, so lumping the two together would hide it.
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
        "jitter_still": float(np.median(steps_still)) if steps_still else float("nan"),
        "jitter_moving": float(np.median(steps_moving)) if steps_moving else float("nan"),
        "moving_frac": 100.0 * len(steps_moving) / max(len(steps_still) + len(steps_moving), 1),
        "bone_spread": float(np.median(spreads)) if spreads else float("nan"),
        "scale_px": float(np.median(shoulders)),
    }


def run_uvc(args):
    cam = ThreadedWebcam(args.device, args.width, args.height)
    time.sleep(1.5)
    try:
        return collect(lambda: cam.frame, args.seconds, "global-shutter webcam")
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

        return collect(frame, args.seconds, "D415 RGB (rolling shutter)")
    finally:
        cam.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--device", type=int, default=2, help="V4L2 index of the webcam")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--only", choices=("uvc", "realsense"))
    ap.add_argument("--upright", dest="rotate_180", action="store_false")
    ap.set_defaults(rotate_180=True)
    args = ap.parse_args(argv)

    runners = [("global-shutter webcam", run_uvc), ("D415 RGB", run_realsense)]
    if args.only:
        runners = [r for r in runners
                   if (args.only == "uvc") == (r[0] == "global-shutter webcam")]

    results = {}
    for name, runner in runners:
        try:
            results[name] = analyse(*runner(args))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: unavailable ({exc})")

    print()
    print(f"{'camera':<26}{'detect':>8}{'jitter still':>14}{'jitter moving':>15}"
          f"{'bone spread':>13}")
    print(f"{'':<26}{'':>8}{'(shoulder widths, lower is better)':>42}")
    print("-" * 76)
    for name, m in results.items():
        if not m:
            print(f"{name:<26}  too few detections")
            continue
        print(f"{name:<26}{m['detection']:7.0f}%{m['jitter_still']:13.4f}"
              f"{m['jitter_moving']:15.4f}{m['bone_spread']:13.4f}")
    for name, m in results.items():
        if m:
            print(f"  {name}: subject was {m['moving_frac']:.0f}% moving, "
                  f"shoulder width {m['scale_px']:.0f} px")

    if len(results) == 2 and all(results.values()):
        a, b = results.values()
        print()
        print("Rolling shutter is a MOTION artefact, so the meaningful comparison "
              "is the 'moving' column.")
        print("If the global shutter only wins there, that is the artefact and it "
              "is real.")
        print("If it wins in neither, rolling shutter costs nothing at these speeds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
