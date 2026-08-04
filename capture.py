"""D415 capture: aligned colour + depth, and pixel -> 3D metres.

Depth and colour come from physically different lenses, so a pixel in the colour
image is not the same pixel in the depth image. rs.align fixes that -- after it,
colour pixel (x, y) and depth pixel (x, y) look at the same thing, which is what
lets us take a 2D landmark from any RGB pose model and read its real depth.

deproject() is the primitive both solver options need: it turns (pixel, depth)
into a 3D point in metres in camera space, using the camera's own intrinsics
rather than a guessed focal length.
"""


import numpy as np
import pyrealsense2 as rs

# 848x480 is the D415's native stereo aspect and streams at 90 Hz; Nuitrack's own
# guidance calls it the optimum size. 30 Hz is what we send to VRChat anyway.
DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30


class DepthCamera:
    def __init__(
        self,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        fps=DEFAULT_FPS,
        rotate_180=False,
        filtering=True,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        # For a camera mounted upside down. This rotates the frames AND the
        # intrinsics together -- rotating only the image would leave the
        # principal point in the old place, and deprojection would silently
        # return mirrored 3D coordinates while the preview looked perfectly fine.
        self.rotate_180 = rotate_180
        # Post-processing can be switched off to measure what it is buying.
        self.filtering = filtering
        self.sensor_warnings = []
        self.laser_power = None
        self.visual_preset_applied = False
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._align = rs.align(rs.stream.color)
        self._depth_scale = None
        self._intrinsics = None

        # RealSense post-processing chain. Raw stereo depth is noticeably noisy
        # and full of holes; these are the filters the hardware is designed to be
        # used with, and skipping them throws away most of the camera's quality.
        #
        # Order matters and is Intel's recommended one. The disparity round-trip
        # is the important part: stereo error is roughly uniform in DISPARITY
        # space and grows quadratically in depth space, so smoothing in disparity
        # weights near and far samples correctly. Smoothing raw metres instead
        # over-smooths close geometry and under-smooths far, which at 2.5 m is
        # exactly where a body is.
        self._to_disparity = rs.disparity_transform(True)
        self._to_depth = rs.disparity_transform(False)

        # Edge-preserving. Deliberately moderate: heavy spatial smoothing drags
        # limb edges into the background, which would reintroduce the very
        # foreground/background mixing the solver works to reject.
        self._spatial = rs.spatial_filter()
        self._spatial.set_option(rs.option.filter_magnitude, 2)
        self._spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self._spatial.set_option(rs.option.filter_smooth_delta, 20)
        self._spatial.set_option(rs.option.holes_fill, 2)

        # The main jitter win: averages each pixel over time, which is what stops
        # a stationary joint from shimmering. alpha 0.4 keeps it responsive enough
        # not to smear real motion.
        self._temporal = rs.temporal_filter()
        self._temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self._temporal.set_option(rs.option.filter_smooth_delta, 20)

        # Fills residual holes from the nearest valid neighbour. Runs last so it
        # only touches pixels the real filters could not recover.
        self._hole_filling = rs.hole_filling_filter()
        self._hole_filling.set_option(rs.option.holes_fill, 1)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        self._config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        self._config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        profile = self._pipeline.start(self._config)

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        self._configure_sensor(depth_sensor)
        return profile

    def _configure_sensor(self, depth_sensor):
        """Push the depth sensor toward coverage over precision.

        A body at 2.5 m is mostly low-texture clothing and skin, which is the
        hard case for stereo matching -- the default preset leaves large holes on
        exactly the torso and limbs we need. HIGH_DENSITY trades some per-pixel
        accuracy for far more valid pixels, which is the right trade when a
        downstream median over a patch is going to average the noise away anyway
        but cannot invent data that is not there.

        Options are set individually and tolerantly: which ones exist varies with
        firmware, and an unsupported option raises rather than being ignored.
        """
        def try_set(option, value, label):
            try:
                if depth_sensor.supports(option):
                    depth_sensor.set_option(option, value)
                    return True
            except Exception as exc:  # noqa: BLE001 - non-fatal by design
                self.sensor_warnings.append(f"{label}: {exc}")
            return False

        # visual_preset goes over a USB extension unit that is busy while the
        # stream spins up, and early attempts fail with "Resource temporarily
        # unavailable". Retrying inside start() is NOT reliable -- observed
        # failing after a full second of retries on a freshly replugged camera,
        # leaving the camera silently on its default depth settings.
        #
        # So the attempt is retried from read() as well, against live frames,
        # until it takes. `visual_preset_applied` is the honest record of
        # whether it ever did.
        self._depth_sensor = depth_sensor
        self._preset_attempts = 0
        self._try_visual_preset()

        # The IR projector is what gives stereo something to match on blank
        # surfaces. Full power is the single biggest coverage win indoors.
        try_set(rs.option.emitter_enabled, 1.0, "emitter_enabled")
        try:
            if depth_sensor.supports(rs.option.laser_power):
                rng = depth_sensor.get_option_range(rs.option.laser_power)
                depth_sensor.set_option(rs.option.laser_power, rng.max)
                self.laser_power = rng.max
        except Exception as exc:  # noqa: BLE001
            self.sensor_warnings.append(f"laser_power: {exc}")

    def _try_visual_preset(self):
        """Attempt HIGH_DENSITY once. Returns True when it has been applied.

        MEDIUM_DENSITY, chosen by measurement on a real body rather than by
        reasoning. Measured over 5 s per preset with a person in frame:

            preset          detected  joints/frame  jitter  bone spread  ankles
            HIGH_DENSITY         97%          12.4   7.6mm       55.3mm     58%
            MEDIUM_DENSITY      100%          14.3   5.1mm       41.4mm     96%
            HIGH_ACCURACY        65%          11.4  20.3mm      116.2mm     21%

        MEDIUM_DENSITY wins on every metric, most importantly ankle availability
        (96% vs 58%) -- and ankles are two of the three trackers we send.

        Note this is the OPPOSITE of what a static scene suggests: on a textured
        desk, HIGH_ACCURACY looks best (3.61mm noise vs 6.53mm) and MEDIUM looks
        unremarkable. A desk is nothing like a body -- clothing and skin are
        low-texture, which is exactly where HIGH_ACCURACY loses the coverage that
        a pose model needs. Benchmark on a person, not on furniture.
        """
        if self.visual_preset_applied or self._depth_sensor is None:
            return True
        if self._preset_attempts >= 200:
            return False
        self._preset_attempts += 1
        try:
            if self._depth_sensor.supports(rs.option.visual_preset):
                self._depth_sensor.set_option(
                    rs.option.visual_preset,
                    float(rs.rs400_visual_preset.medium_density),
                )
                self.visual_preset_applied = True
                self.sensor_warnings = [
                    w for w in self.sensor_warnings if "visual_preset" not in w
                ]
                return True
        except Exception:  # noqa: BLE001 - expected while the stream settles
            if self._preset_attempts == 200:
                self.sensor_warnings.append(
                    "visual_preset: never applied (running on camera default)"
                )
        return False

    def stop(self):
        self._pipeline.stop()

    @property
    def depth_scale(self):
        """Metres per raw depth unit (0.001 on the D415, but read it, don't assume)."""
        return self._depth_scale

    def read(self, timeout_ms=5000):
        """Return (color_bgr, depth_raw, depth_frame) with depth aligned to colour."""
        frames = self._pipeline.wait_for_frames(timeout_ms)
        # Keep trying until the preset takes; it often only becomes settable
        # once frames are genuinely flowing.
        if not self.visual_preset_applied:
            self._try_visual_preset()
        aligned = self._align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        # Filter AFTER aligning, so filtered depth stays pixel-aligned with the
        # colour image the pose model runs on. (Decimation is deliberately not in
        # the chain -- it changes resolution and would break that correspondence.)
        if self.filtering:
            depth_frame = self._to_disparity.process(depth_frame)
            depth_frame = self._spatial.process(depth_frame)
            depth_frame = self._temporal.process(depth_frame)
            depth_frame = self._to_depth.process(depth_frame)
            depth_frame = self._hole_filling.process(depth_frame)
            depth_frame = depth_frame.as_depth_frame()

        if self._intrinsics is None:
            intr = depth_frame.profile.as_video_stream_profile().intrinsics
            self._intrinsics = self._rotated_intrinsics(intr) if self.rotate_180 else intr

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        if self.rotate_180:
            # Reversing both axes is a 180 deg rotation. Copy, because MediaPipe
            # and cv2 both need contiguous buffers and a reversed view is not.
            color = np.ascontiguousarray(color[::-1, ::-1])
            depth = np.ascontiguousarray(depth[::-1, ::-1])

        return color, depth, depth_frame

    @staticmethod
    def _rotated_intrinsics(intr):
        """Intrinsics for the same camera turned 180 deg about its optical axis.

        A 180 deg image rotation maps pixel (x, y) -> (W-1-x, H-1-y). Deprojecting
        the rotated pixel with the principal point moved the same way yields
        (-X, -Y, Z) -- i.e. exactly the upright-world coordinates we want, with
        focal lengths unchanged.

        Safe here only because every distortion coefficient on this camera is
        zero. With tangential distortion, p1/p2 would need their signs flipped.
        """
        rotated = rs.intrinsics()
        rotated.width, rotated.height = intr.width, intr.height
        rotated.fx, rotated.fy = intr.fx, intr.fy
        rotated.ppx = intr.width - 1 - intr.ppx
        rotated.ppy = intr.height - 1 - intr.ppy
        rotated.model = intr.model
        rotated.coeffs = list(intr.coeffs)
        if any(intr.coeffs):
            raise NotImplementedError(
                "rotate_180 assumes zero distortion; this camera reports "
                f"coeffs={list(intr.coeffs)} and would need p1/p2 sign flips"
            )
        return rotated

    @property
    def intrinsics(self):
        return self._intrinsics

    def deproject(self, x, y, depth_metres):
        """Pixel (x, y) + depth in metres -> (X, Y, Z) in metres, camera space.

        Camera space is right-handed: +X right, +Y *down*, +Z forward from the
        lens. Converting to VRChat's left-handed +Y-up space is transform.py's
        job -- do not do it here.

        Signs verified on this camera, not assumed from convention (getting them
        wrong is the classic mirrored/inverted-avatar bug):
            pixel above centre -> Y = -0.237     pixel below centre -> Y = +0.231
            pixel left of centre -> X = -0.953   pixel right -> X = +0.410
        So the Y flip into Unity space is a negation, and X passes through --
        handedness is fixed by negating Z, not X.
        """
        if self._intrinsics is None:
            raise RuntimeError("call read() at least once before deproject()")
        return rs.rs2_deproject_pixel_to_point(
            self._intrinsics, [float(x), float(y)], float(depth_metres)
        )

    def project(self, point):
        """Camera-space metres -> pixel. Inverse of deproject().

        Needed to draw what is actually being SENT: after stabilisation and
        occlusion filling, a joint's 3D position no longer corresponds to the
        landmark pixel it came from, and a reconstructed joint has no landmark
        pixel at all. Drawing the original pixels would show a preview that
        disagrees with the data leaving the machine.
        """
        if self._intrinsics is None:
            raise RuntimeError("call read() at least once before project()")
        return rs.rs2_project_point_to_pixel(self._intrinsics, [float(v) for v in point])

    def depth_samples(self, depth_raw, x, y, patch=3):
        """All valid depths (metres) in a small patch around (x, y), unsorted.

        Returned raw rather than reduced because the right reduction depends on
        context: near a body silhouette the patch straddles subject and
        background, and the median then lands on the wall. Callers that know the
        subject's approximate depth should filter these against it.
        """
        h, w = depth_raw.shape
        x0, x1 = max(0, int(x) - patch), min(w, int(x) + patch + 1)
        y0, y1 = max(0, int(y) - patch), min(h, int(y) + patch + 1)
        window = depth_raw[y0:y1, x0:x1]
        valid = window[window > 0]
        if valid.size == 0:
            return np.empty(0, dtype=float)
        return valid.astype(float) * self._depth_scale

