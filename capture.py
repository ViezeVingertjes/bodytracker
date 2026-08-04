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

try:
    import pyrealsense2 as rs
except ImportError as _exc:  # pragma: no cover - environment-specific
    # This is the first error most new users hit, and the bare ImportError says
    # nothing about why it is not simply a pip install on every platform.
    raise ImportError(
        "pyrealsense2 is required for camera capture but is not installed.\n"
        "\n"
        "  Windows:  pip install pyrealsense2\n"
        "  Linux:    there is often no wheel for your Python, and it must be\n"
        "            built from source. See SETUP.md for the exact build --\n"
        "            two cmake flags are required or it fails at link time --\n"
        "            and the udev rule the camera needs to work without root.\n"
        "\n"
        "Everything except capture works without it, including `pytest`,\n"
        "`bodytracker fetch`, `bodytracker fake` and `bodytracker listen`."
    ) from _exc

# 848x480 is the D415's native stereo aspect and streams at 90 Hz; Nuitrack's own
# guidance calls it the optimum size. 30 Hz is what we send to VRChat anyway.
DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30


# Sensor gain used in IR mode. The stereo module's auto-exposure optimises for
# DEPTH -- it targets the contrast of the projector's dots, not a viewable image
# -- so it pins gain at the minimum (16 of 16..248) and leaves IR starved. That
# is why the dim lower body tracked so badly: the signal was freely available and
# simply not being taken.
#
# Measured, exposure held at its 30fps maximum of 33000us:
#
#     gain   IR mean   depth coverage   depth noise
#       16       8.2           85.9%        10.58mm
#      128      41.0           85.8%        11.28mm
#      200      62.1           86.2%        11.44mm
#
# 128 is a deliberate midpoint: 5x the IR brightness for ~7% more depth noise and
# no coverage loss, keeping headroom before highlights clip. Depth quality still
# matters -- this is a depth tracker.
DEFAULT_IR_GAIN = 128

IR_MODES = ("global", "masked", "clahe", "masked_clahe")
DEFAULT_IR_MODE = "masked"


def normalise_ir(ir, depth=None, mode=DEFAULT_IR_MODE, depth_scale=0.001,
                 min_distance=0.3, max_distance=4.0):
    """Single-channel IR -> a 3-channel image the pose model can use.

    The stereo imagers are exposed for DEPTH, not for a photograph, so the raw
    frame is very dark (measured mean 21/255, p95 38) and MediaPipe needs
    recognisable contrast. Raising the sensor's own exposure would brighten IR at
    the cost of the depth quality that shares those imagers, so this is done in
    software.

    The projector is the illumination, and its light falls off as 1/distance^2.
    That produces a strong gradient: on a real subject, head and shoulders track
    at ~98% while ankles manage ~30% and toes 6%, with "low visibility" the
    dominant rejection. The lower body is simply too dim.

    Modes, in increasing order of how hard they fight that:

    global       percentile stretch over the whole frame. Simple, and the
                 original -- but a bright nearby object sets the scale and
                 leaves the actual subject dark.
    masked       percentiles computed only over pixels inside the working
                 volume, so the SUBJECT sets the exposure rather than the
                 furniture. This is what depth is for.
    clahe        local histogram equalisation. DO NOT USE with the emitter on:
                 it amplifies the projector's dot pattern along with everything
                 else, and took tracking from 48% to zero. Kept only for use
                 with an emitter-off / externally-lit IR image.
    masked_clahe both -- restrict the range to the subject, then equalise
                 locally.
    """
    ir = np.asarray(ir)
    if mode in ("masked", "masked_clahe") and depth is not None:
        in_volume = (depth * depth_scale > min_distance) & \
                    (depth * depth_scale < max_distance)
        sample = ir[in_volume] if in_volume.sum() > 500 else ir
    else:
        sample = ir

    lo, hi = np.percentile(sample, (2, 98))
    if hi - lo < 1:
        lo, hi = float(ir.min()), max(float(ir.max()), lo + 1)
    stretched = np.clip((ir.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    out = stretched.astype(np.uint8)

    if mode in ("clahe", "masked_clahe"):
        import cv2
        out = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(out)

    return np.repeat(out[:, :, None], 3, axis=2)


class DepthCamera:
    def __init__(
        self,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        fps=DEFAULT_FPS,
        rotate_180=False,
        filtering=True,
        min_distance=0.3,
        max_distance=4.0,
        source="color",
        ir_mode=DEFAULT_IR_MODE,
        ir_gain=DEFAULT_IR_GAIN,
        ir_emitter="on",
    ):
        self.width = width
        self.height = height
        self.fps = fps
        # For a camera mounted upside down. This rotates the frames AND the
        # intrinsics together -- rotating only the image would leave the
        # principal point in the old place, and deprojection would silently
        # return mirrored 3D coordinates while the preview looked perfectly fine.
        self.rotate_180 = rotate_180
        # Which image the pose model sees: "color" or "ir".
        #
        # IR matters for two reasons. It works in a DARK ROOM -- the projector is
        # the illumination, so the imagers see a well-lit person with the lights
        # off, whereas RGB sees nothing and MediaPipe finds nothing to place.
        #
        # And depth is computed in the left IR imager's own frame, so IR needs no
        # rs.align at all: a landmark pixel and its depth pixel are the same
        # pixel by construction, instead of being made to correspond.
        #
        # A rendered depth map would NOT work here. BlazePose is trained on
        # photographs; a false-colour depth image is far outside that
        # distribution. IR is still a real photograph, just at 850nm.
        self.source = source
        self.ir_mode = ir_mode
        self.ir_gain = ir_gain
        # Projector state in IR mode -- the central tension of IR tracking:
        #
        #   on   depth works (97% coverage) but the projector's dot pattern is
        #        painted over everything, and the pose model cannot read a body
        #        through it. Measured 48% detection, and 0% once CLAHE amplified
        #        the dots.
        #   off  a clean, natural IR photograph -- what the model actually wants
        #        -- but the projector WAS the illumination, so depth coverage
        #        collapses to ~30-75%.
        #
        # Mode 2 ("alternate") is accepted by this device but measured as
        # behaving like OFF: laser metadata reads 0 on every frame and coverage
        # drops to 29.7%. Not a usable middle ground here.
        self.ir_emitter = ir_emitter
        # Post-processing can be switched off to measure what it is buying.
        self.filtering = filtering
        # Working volume. 4 m comfortably covers standing at 2.3-3 m plus arm
        # reach, while excluding the rest of the room.
        self.min_distance = min_distance
        self.max_distance = max_distance
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
        # Clip to the working volume BEFORE anything else. Everything past
        # `max_distance` is wall, furniture or another room, and letting it into
        # the chain actively hurts: the spatial filter drags far values across
        # limb edges, and hole filling pulls them into gaps on the body. Cutting
        # first means the later stages only ever see plausible subject depth.
        self._threshold = rs.threshold_filter()
        self._threshold.set_option(rs.option.min_distance, min_distance)
        self._threshold.set_option(rs.option.max_distance, max_distance)

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

        # Runs last, so it only touches pixels the real filters could not recover.
        #
        # Mode 2 = nearest_from_around, NOT the default 1 = farthest_from_around.
        # This matters and the default is actively wrong for us: filling a hole
        # with the FURTHEST neighbour means a gap on a person's torso or leg gets
        # filled with the wall behind them. The solver then either rejects that
        # joint as background (losing it) or, worse, believes it. We track a
        # foreground subject, so the nearest neighbour is the right guess.
        self._hole_filling = rs.hole_filling_filter()
        self._hole_filling.set_option(rs.option.holes_fill, 2)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        self._config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        if self.source == "ir":
            # Imager 1 is the left one, which is depth's reference frame.
            self._config.enable_stream(
                rs.stream.infrared, 1, self.width, self.height,
                rs.format.y8, self.fps
            )
        else:
            self._config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
        try:
            profile = self._pipeline.start(self._config)
        except RuntimeError as exc:
            raise RuntimeError(
                f"could not start the camera: {exc}\n"
                "  - is a RealSense connected? (lsusb | grep 8086)\n"
                "  - is another process using it? only one may open it at a time\n"
                "  - after repeated open/close cycles the stream can degrade; "
                "unplug and replug the camera"
            ) from exc

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

        if self.source == "ir" and self.ir_gain:
            # Auto-exposure must go, or it immediately winds gain back down.
            try_set(rs.option.enable_auto_exposure, 0.0, "ir_auto_exposure")
            try_set(rs.option.exposure, 33000.0, "ir_exposure")
            try_set(rs.option.gain, float(self.ir_gain), "ir_gain")

        # ORDER MATTERS. Setting laser_power re-enables the emitter, so it must
        # come FIRST and be skipped entirely when the emitter is meant to be off
        # -- otherwise `--ir-emitter off` silently does nothing, which is exactly
        # what happened (coverage read 97% when it should have collapsed).
        emitter = {"on": 1.0, "off": 0.0, "alternate": 2.0}[
            self.ir_emitter if self.source == "ir" else "on"]

        if emitter:
            # Full power is the single biggest depth-coverage win indoors: the
            # projector is what gives stereo something to match on blank walls,
            # clothing and skin.
            try:
                if depth_sensor.supports(rs.option.laser_power):
                    rng = depth_sensor.get_option_range(rs.option.laser_power)
                    depth_sensor.set_option(rs.option.laser_power, rng.max)
                    self.laser_power = rng.max
            except Exception as exc:  # noqa: BLE001
                self.sensor_warnings.append(f"laser_power: {exc}")

        try_set(rs.option.emitter_enabled, emitter, "emitter_enabled")

    def _try_visual_preset(self):
        """Attempt MEDIUM_DENSITY once. Returns True when it has been applied.

        MEDIUM_DENSITY, chosen by measurement on a real body rather than by
        reasoning. Measured over 5 s per preset with a person in frame:

            preset          detected  joints/frame  jitter  bone spread  ankles
            HIGH_DENSITY         97%          12.4   7.6mm       55.3mm     58%
            MEDIUM_DENSITY      100%          14.3   5.1mm       41.4mm     96%
            HIGH_ACCURACY        65%          11.4  20.3mm      116.2mm     21%

        MEDIUM_DENSITY wins on every metric, most importantly ankle availability
        (96% vs 58%) -- and both feet are default trackers.

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

    def light_margin(self):
        """How much low-light headroom the colour sensor still has.

        Returns (exposure_us, exposure_max, gain, gain_max, stops_left) or None.

        Worth surfacing because the intuition is wrong: in a normally-lit room
        auto-exposure sits at ~166us of a 10000us maximum, so there is roughly
        60x of exposure headroom plus 2x of gain before the sensor runs out --
        about 100x, or ~7 stops. A "dim" room is nowhere near that limit, which
        is why colour keeps working long after it feels too dark to the eye.

        The real low-light failure is not loss of detection but motion blur once
        exposure grows long, and noise once gain is high.
        """
        if self.source != "color":
            return None
        try:
            device = self._pipeline.get_active_profile().get_device()
            sensor = next(s for s in device.sensors
                          if "RGB" in s.get_info(rs.camera_info.name))
            exposure = sensor.get_option(rs.option.exposure)
            exp_max = sensor.get_option_range(rs.option.exposure).max
            gain = sensor.get_option(rs.option.gain)
            gain_max = sensor.get_option_range(rs.option.gain).max
        except Exception:  # noqa: BLE001
            return None
        headroom = (exp_max / max(exposure, 1)) * (gain_max / max(gain, 1))
        return exposure, exp_max, gain, gain_max, float(np.log2(max(headroom, 1)))

    def stop(self):
        self._pipeline.stop()

    @property
    def depth_scale(self):
        """Metres per raw depth unit (0.001 on the D415, but read it, don't assume)."""
        return self._depth_scale

    def health_warning(self, delivery_fps):
        """Detect the stream degradation that repeated open/close causes.

        Measured: identical code and identical inference time, but the third
        camera open in one session delivered 0.7 fps instead of 28. It recovers
        only by replugging. Users will hit this and have no way to know it is a
        hardware state rather than their configuration, so say so explicitly.

        `delivery_fps` must be how fast the CAMERA delivers, not how fast the
        caller's pipeline runs. Those differ whenever inference is the
        bottleneck, and passing the latter reports a hardware fault for what is
        really a heavier model or a slower CPU.
        """
        if delivery_fps < self.fps * 0.5:
            return (f"only {delivery_fps:.1f} fps of an expected {self.fps} -- "
                    "this is usually the camera stream degrading after repeated "
                    "open/close cycles. UNPLUG AND REPLUG the camera; it is not "
                    "a configuration problem.")
        return None

    def read(self, timeout_ms=5000):
        """Return (color_bgr, depth_raw, depth_frame) with depth aligned to colour."""
        frames = self._pipeline.wait_for_frames(timeout_ms)
        # Keep trying until the preset takes; it often only becomes settable
        # once frames are genuinely flowing.
        if not self.visual_preset_applied:
            self._try_visual_preset()

        if self.source == "ir":
            # No alignment: depth already lives in this imager's frame.
            depth_frame = frames.get_depth_frame()
            image_frame = frames.get_infrared_frame(1)
        else:
            aligned = self._align.process(frames)
            depth_frame = aligned.get_depth_frame()
            image_frame = aligned.get_color_frame()
        if not depth_frame or not image_frame:
            return None

        # Filter AFTER aligning, so filtered depth stays pixel-aligned with the
        # colour image the pose model runs on. (Decimation is deliberately not in
        # the chain -- it changes resolution and would break that correspondence.)
        if self.filtering:
            depth_frame = self._threshold.process(depth_frame)
            depth_frame = self._to_disparity.process(depth_frame)
            depth_frame = self._spatial.process(depth_frame)
            depth_frame = self._temporal.process(depth_frame)
            depth_frame = self._to_depth.process(depth_frame)
            depth_frame = self._hole_filling.process(depth_frame)
            depth_frame = depth_frame.as_depth_frame()

        if self._intrinsics is None:
            intr = depth_frame.profile.as_video_stream_profile().intrinsics
            self._intrinsics = self._rotated_intrinsics(intr) if self.rotate_180 else intr

        color = np.asanyarray(image_frame.get_data())
        if self.source == "ir":
            color = normalise_ir(
                color, np.asanyarray(depth_frame.get_data()), self.ir_mode,
                self._depth_scale or 0.001, self.min_distance, self.max_distance)
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

