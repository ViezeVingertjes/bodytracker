"""Skeleton solving: MediaPipe Pose landmarks, lifted to real 3D using depth.

MediaPipe gives us 33 landmarks in *image* space plus a guessed Z. The guessed Z
is the weak part of every RGB-only tracker -- it is inferred from apparent limb
proportions, and it is exactly the axis a depth camera can measure instead. So we
keep MediaPipe's (x, y) and throw its Z away, reading the real depth at that pixel
and deprojecting through the camera intrinsics.

NOTE ON THE API: MediaPipe 1.0 removed `mp.solutions` entirely. Nearly all
tutorials and community projects (ju1ce's included) use `mp.solutions.pose`, which
no longer exists -- this uses the Tasks API and an explicit .task model file.
"""

import os

# MUST be set before mediapipe is imported -- these configure the C++ glog/absl
# backend at load time, and setting them afterwards has no effect. Without them
# every run spams "Feedback manager requires a model with a single signature"
# and "NORM_RECT without IMAGE_DIMENSIONS", neither of which is actionable:
# both are internal to MediaPipe's own graph, not caused by how we call it.
os.environ.setdefault("GLOG_minloglevel", "2")  # 0=INFO 1=WARNING 2=ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_stderrthreshold", "2")

import collections  # noqa: E402
import contextlib  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
from mediapipe import Image, ImageFormat  # noqa: E402
from mediapipe.tasks.python import BaseOptions  # noqa: E402
from mediapipe.tasks.python.vision import (  # noqa: E402
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# Accuracy-first default. MediaPipe ships three pose models; `heavy` is the most
# accurate. It is affordable here: the pipeline is camera-limited at 30 fps with
# roughly 20 ms/frame spare, so the extra inference cost comes out of idle time.
# Step down with --model if your machine cannot hold 30 fps.
MODELS = ("lite", "full", "heavy")
DEFAULT_MODEL = "heavy"


def model_path(name):
    return f"models/pose_landmarker_{name}.task"


MODEL_PATH = model_path(DEFAULT_MODEL)

# MediaPipe pose landmark indices we care about. Left/right are the SUBJECT's.
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32  # foot index (toe) -- gives foot direction

# Landmarks we need a real depth reading for. Everything else is ignored.
#
# Wrists are tracked but NOT sent as trackers -- VRChat has no hand or wrist
# tracker slot (the controllers drive the hands). They are here because elbow
# orientation comes from the shoulder->elbow->wrist chain, and because seeing the
# forearm in the preview is the quickest read on whether tracking is healthy.
NEEDED = (
    NOSE, L_EAR, R_EAR,
    L_SHOULDER, R_SHOULDER,
    L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST,
    L_HIP, R_HIP,
    L_KNEE, R_KNEE,
    L_ANKLE, R_ANKLE,
    L_FOOT, R_FOOT,
)


@contextlib.contextmanager
def _captured_stderr():
    """Silence C-level stderr, but replay it if something actually fails.

    MediaPipe emits its remaining load-time noise ("Created TensorFlow Lite
    XNNPACK delegate", "Feedback manager requires...") *before* glog is
    initialised -- it says so itself -- so no log-level env var can filter it.
    Only redirecting the file descriptor works.

    The captured text is replayed on exception, so a genuine model-loading
    failure (missing .task file, corrupt download) is still visible rather than
    swallowed into a silent crash.
    """
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+") as sink:
        try:
            os.dup2(sink.fileno(), 2)
            yield
        except BaseException:
            os.dup2(saved, 2)
            sink.seek(0)
            captured = sink.read()
            if captured.strip():
                sys.stderr.write(captured)
            raise
        finally:
            os.dup2(saved, 2)
            os.close(saved)


class Skeleton:
    """3D joints in camera space (metres). Missing joints are absent from `joints`."""

    def __init__(self, joints, visibility, pixels=None, rejected=None, body_depth=None):
        self.joints = joints          # {landmark_index: np.array([X, Y, Z])}
        self.visibility = visibility  # {landmark_index: float 0..1}
        self.pixels = pixels or {}    # {landmark_index: (px, py)} -- for overlays
        # Landmarks MediaPipe found but we dropped, with the reason. Worth
        # surfacing: a joint silently missing looks identical to a joint the
        # model never saw, and they need different fixes.
        self.rejected = rejected or {}
        self.body_depth = body_depth  # torso depth reference, metres

    def has(self, *indices):
        return all(i in self.joints for i in indices)

    def midpoint(self, a, b):
        if not self.has(a, b):
            return None
        return (self.joints[a] + self.joints[b]) * 0.5

    def get(self, index):
        return self.joints.get(index)


class PoseSolver:
    def __init__(self, model_path=MODEL_PATH, min_visibility=0.5, depth_tolerance=0.45):
        self.min_visibility = min_visibility
        # How far a landmark's depth may sit from the torso depth before it is
        # treated as background. 0.45 m comfortably covers a limb extended toward
        # or away from the camera while still rejecting a wall behind the subject.
        self.depth_tolerance = depth_tolerance

        # Plausibility gating exists to exclude furniture-shaped false positives
        # (an early session produced a "body" 0.10 m across with a 0.24 m torso),
        # NOT to police body shape.
        #
        # ABSOLUTE bounds are deliberately very loose, because measured body
        # dimensions vary enormously with pose and viewing angle on the SAME
        # person. Measured here: standing and framed, torso 0.55 m and shoulders
        # 0.35 m; seated and closer, the same person measures torso 0.31 m and
        # shoulders 0.24 m. Turning sideways shrinks shoulder width further still,
        # because the far shoulder is occluded and its depth collapses onto the
        # near body.
        #
        # A tight absolute floor therefore cannot distinguish "not a person" from
        # "person in a different posture" -- a 0.30 m torso floor rejected 36.5%
        # of frames from a real subject, and every rejected frame is a frozen
        # limb in VRChat. So absolute bounds only catch gross nonsense, and the
        # real discrimination is RELATIVE to this person (see _is_plausible).
        self.min_shoulder_width, self.max_shoulder_width = 0.10, 0.80
        self.min_torso, self.max_torso = 0.15, 1.00

        # Learned per-subject scale. Rejecting deviation from THIS person's
        # measured dimensions adapts to build, posture and distance in a way no
        # fixed threshold can.
        self._torso_history = collections.deque(maxlen=150)
        self.min_scale_samples = 40
        self.max_scale_deviation = 0.40
        # Consecutive rejections before concluding the baseline is stale (~0.7 s
        # at 30 Hz) rather than the measurements being wrong.
        self.max_scale_rejects = 20
        self._scale_rejects = 0
        # Max hip movement between frames before it is treated as a different
        # subject. ~2.5 m/s at 30 Hz -- faster than anyone walks past a camera.
        self.max_torso_jump = 0.35
        self._last_torso = None
        self.last_rejection = None

        # Frames to keep stderr captured while waiting for the first detection
        # (~30 s at 30 Hz), after which the noise is accepted rather than paying
        # the redirect cost indefinitely.
        self._quiet_frames = 900
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            # Exactly one person. MediaPipe returns only its highest-confidence
            # pose, so a second person in frame cannot produce a second skeleton.
            # Which person it picks between frames is not guaranteed, though --
            # that is what the torso-continuity check in _is_plausible defends.
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"pose model not found at {model_path!r}. Download it with:\n"
                "  curl -sSL -o models/pose_landmarker_full.task "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            )
        with _captured_stderr():
            self._landmarker = PoseLandmarker.create_from_options(options)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def solve(self, camera, color_bgr, depth_raw, timestamp_ms):
        """Return a Skeleton, or None if no person was found.

        `camera` is a DepthCamera -- used for its intrinsics and depth sampling,
        not re-read here.
        """
        rgb = np.ascontiguousarray(color_bgr[:, :, ::-1])
        image = Image(image_format=ImageFormat.SRGB, data=rgb)

        # MediaPipe emits a one-off NORM_RECT/IMAGE_DIMENSIONS warning from inside
        # its own graph. It comes from the landmark projection stage, which only
        # runs once a pose is actually FOUND -- so suppressing the first call is
        # not enough; it has to be the first successful detection. Capped so a
        # long personless run does not redirect fds at 30 Hz forever.
        if self._quiet_frames > 0:
            self._quiet_frames -= 1
            with _captured_stderr():
                result = self._landmarker.detect_for_video(image, int(timestamp_ms))
            if result.pose_landmarks:
                self._quiet_frames = 0  # warning has been emitted and swallowed
        else:
            result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        h, w = depth_raw.shape

        # Pass 1: gather candidate depths per landmark, without committing.
        candidates, rejected, pixels = {}, {}, {}
        for idx in NEEDED:
            lm = landmarks[idx]
            vis = getattr(lm, "visibility", 1.0)
            px, py = lm.x * w, lm.y * h
            pixels[idx] = (px, py)

            if vis < self.min_visibility:
                rejected[idx] = "low visibility"
                continue

            # MediaPipe returns normalised coords, and they can fall outside the
            # frame when a limb is cut off -- clamping would invent a depth
            # reading from whatever is at the frame edge, so drop instead.
            if not (0 <= px < w and 0 <= py < h):
                rejected[idx] = "off frame"
                continue

            samples = camera.depth_samples(depth_raw, px, py)
            if samples.size == 0:
                rejected[idx] = "no depth"  # stereo hole
                continue
            candidates[idx] = (px, py, float(vis), samples)

        if not candidates:
            return None

        body_depth = self._estimate_body_depth(candidates)
        if body_depth is None:
            return None

        # Pass 2: keep only depths consistent with the body, not the background.
        joints, visibility = {}, {}
        for idx, (px, py, vis, samples) in candidates.items():
            near = samples[np.abs(samples - body_depth) <= self.depth_tolerance]
            if near.size == 0:
                rejected[idx] = "background"
                continue
            depth_m = float(np.median(near))
            joints[idx] = np.array(camera.deproject(px, py, depth_m), dtype=float)
            visibility[idx] = vis

        if not joints:
            return None

        skeleton = Skeleton(joints, visibility, pixels, rejected, body_depth)
        if not self._is_plausible(skeleton):
            return None
        return skeleton

    def _is_plausible(self, skeleton):
        """Reject detections that are not a real human body.

        MediaPipe will confidently fit a pose to furniture, a poster, or a
        reflection. An earlier session produced a "body" 10 cm across the
        shoulders with a 24 cm torso -- obvious nonsense in metres, completely
        invisible in normalised image coordinates, which is why this check has to
        live here and not in the model.

        Only fires when the measurement exists: a partially-framed real person
        must not be thrown away just because their shoulders are occluded.
        """
        shoulders = skeleton.get(L_SHOULDER), skeleton.get(R_SHOULDER)
        if all(p is not None for p in shoulders):
            width = float(np.linalg.norm(shoulders[0] - shoulders[1]))
            if not self.min_shoulder_width <= width <= self.max_shoulder_width:
                self.last_rejection = f"shoulder width {width:.2f} m"
                return False

        chest = skeleton.midpoint(L_SHOULDER, R_SHOULDER)
        hips = skeleton.midpoint(L_HIP, R_HIP)
        if chest is not None and hips is not None:
            torso = float(np.linalg.norm(chest - hips))
            if not self.min_torso <= torso <= self.max_torso:
                self.last_rejection = f"torso length {torso:.2f} m"
                return False

            # Relative to what this person actually measures. The median is
            # taken over accepted frames only, so a burst of bad frames cannot
            # drag the baseline onto itself.
            if len(self._torso_history) >= self.min_scale_samples:
                learned = float(np.median(self._torso_history))
                if learned > 1e-6 and (
                    abs(torso - learned) / learned > self.max_scale_deviation
                ):
                    self._scale_rejects += 1
                    # A learned baseline must be able to go stale. Sitting down
                    # then standing up genuinely changes measured torso length
                    # (0.29 m -> 0.49 m observed), and rejecting against the old
                    # value would lock tracking out for as long as you stand.
                    # Sustained disagreement means the BASELINE is wrong, not the
                    # measurement, so relearn instead of rejecting forever.
                    if self._scale_rejects <= self.max_scale_rejects:
                        self.last_rejection = (
                            f"torso {torso:.2f} m vs learned {learned:.2f} m"
                        )
                        return False
                    self._torso_history.clear()
                self._scale_rejects = 0
            self._torso_history.append(torso)

            # Continuity: one person, not a new one every frame. A real torso
            # cannot teleport, so a large jump means the model latched onto
            # something else -- another person walking through, or a false
            # positive. Rejecting keeps identity stable without any tracking-by-
            # detection machinery.
            if self._last_torso is not None:
                jump = float(np.linalg.norm(hips - self._last_torso))
                if jump > self.max_torso_jump:
                    self.last_rejection = f"torso jumped {jump:.2f} m"
                    # Accept the new position as the reference anyway, or a
                    # genuine fast movement would lock tracking out permanently.
                    self._last_torso = hips
                    return False
            self._last_torso = hips

        self.last_rejection = None
        return True

    def _estimate_body_depth(self, candidates):
        """Depth of the torso, used as the reference for rejecting background.

        Torso landmarks are used because they sit well inside the silhouette, so
        their patches are least likely to straddle the background. The nearest
        plausible reading is taken rather than the median: where a patch does
        straddle an edge, the erroneous samples are always *further* away.
        """
        core = [
            samples
            for idx, (_px, _py, _vis, samples) in candidates.items()
            if idx in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
        ]
        if not core:
            core = [s for (_px, _py, _vis, s) in candidates.values()]
        per_joint = [float(np.percentile(s, 25)) for s in core if s.size]
        if not per_joint:
            return None
        return float(np.median(per_joint))
