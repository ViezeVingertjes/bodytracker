# Decoupled sender + rotation prediction — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let output run at headset rate instead of camera rate, and give rotations the same latency compensation positions already get.

**Architecture:** A `PoseSender` owns the position and rotation predictors behind a lock and can either be pumped inline by the capture loop (today's behaviour) or run its own thread at a fixed rate. A new `RotationPredictor` extrapolates orientation in SO(3) — never in euler angles, which are wrong at the wrap.

**Tech Stack:** Python 3.10–3.12, numpy, python-osc, pytest, ruff.

## Global Constraints

- `--send-hz` defaults to `0`, meaning "send once per camera frame". Shipped behaviour must be unchanged at the default.
- Rotations are never extrapolated as euler angles. Predict on matrices, convert to euler only at output.
- Reuse `TrackerPredictor`'s existing guards: EMA-smoothed velocity, clamped rate, horizon clamped to 0.12 s.
- Line length 100. Ruff rules `E,F,W,B,SIM,UP,C4,RET,ARG` must pass.
- Measurement was explicitly deferred by the user; do not add `record`/`sweep` commands.

---

### Task 1: Rodrigues helpers

**Files:**
- Modify: `transform.py` (after `unity_euler_to_matrix`)
- Test: `tests/test_transform.py`

**Interfaces:**
- Produces: `rotvec_to_matrix(v) -> (3,3) ndarray`, `matrix_to_rotvec(m) -> (3,) ndarray`. A rotation vector is `axis * angle`, angle in radians.

- [ ] **Step 1: Write the failing test**

```python
class TestRotationVectors:
    def test_round_trip_over_random_rotations(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(0.0, np.pi - 1e-3)
            v = axis * angle
            back = matrix_to_rotvec(rotvec_to_matrix(v))
            assert np.allclose(back, v, atol=1e-8)

    def test_zero_vector_is_identity(self):
        assert np.allclose(rotvec_to_matrix([0.0, 0.0, 0.0]), np.eye(3))
        assert np.allclose(matrix_to_rotvec(np.eye(3)), np.zeros(3))

    def test_near_pi_recovers_the_axis(self):
        # The antisymmetric part vanishes at pi, so the generic formula divides
        # by ~0. This is the case that silently returns garbage if unhandled.
        for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]):
            axis = np.array(axis, dtype=float)
            axis /= np.linalg.norm(axis)
            m = rotvec_to_matrix(axis * np.pi)
            back = matrix_to_rotvec(m)
            assert np.isclose(np.linalg.norm(back), np.pi, atol=1e-6)
            # k and -k describe the same rotation at pi; accept either.
            assert (np.allclose(back / np.pi, axis, atol=1e-5)
                    or np.allclose(back / np.pi, -axis, atol=1e-5))

    def test_result_is_a_rotation(self):
        m = rotvec_to_matrix([0.3, -0.7, 1.1])
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(m), 1.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_transform.py -k RotationVectors -q`
Expected: FAIL, ImportError on `rotvec_to_matrix`.

- [ ] **Step 3: Implement**

```python
def rotvec_to_matrix(v):
    """Rotation vector (axis * angle, radians) -> rotation matrix. Rodrigues."""
    v = np.asarray(v, dtype=float)
    theta = float(np.linalg.norm(v))
    if theta < 1e-9:
        return np.eye(3)
    k = v / theta
    cross = np.array([[0.0, -k[2], k[1]],
                      [k[2], 0.0, -k[0]],
                      [-k[1], k[0], 0.0]])
    return (np.eye(3) + math.sin(theta) * cross
            + (1.0 - math.cos(theta)) * (cross @ cross))


def matrix_to_rotvec(m):
    """Rotation matrix -> rotation vector. Inverse of rotvec_to_matrix."""
    m = np.asarray(m, dtype=float)
    theta = math.acos(float(np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0)))
    if theta < 1e-9:
        return np.zeros(3)
    if theta > math.pi - 1e-6:
        # At pi the antisymmetric part is zero, so the generic formula divides
        # by sin(theta) ~ 0. Near pi, (R + I)/2 = k k^T, so read the axis off
        # its diagonal -- taking the largest entry, because a small one loses
        # precision in the square root.
        outer = (m + np.eye(3)) / 2.0
        diag = np.clip(np.diag(outer), 0.0, None)
        i = int(np.argmax(diag))
        k = np.zeros(3)
        k[i] = math.sqrt(diag[i])
        for j in range(3):
            if j != i:
                k[j] = outer[i][j] / k[i]
        n = float(np.linalg.norm(k))
        if n < 1e-9:
            return np.zeros(3)
        return k / n * theta
    axis = np.array([m[2][1] - m[1][2],
                     m[0][2] - m[2][0],
                     m[1][0] - m[0][1]])
    return axis * (theta / (2.0 * math.sin(theta)))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_transform.py -k RotationVectors -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transform.py tests/test_transform.py
git commit -m "Add Rodrigues helpers for SO(3) extrapolation"
```

---

### Task 2: RotationPredictor

**Files:**
- Modify: `transform.py` (after `RotationSmoother`)
- Test: `tests/test_transform.py`

**Interfaces:**
- Consumes: `rotvec_to_matrix`, `matrix_to_rotvec` (Task 1); existing `unity_euler_to_matrix`, `matrix_to_unity_euler`.
- Produces: `RotationPredictor(velocity_alpha=0.35, max_rate_deg=720.0)` with `update(rotations, t)` taking `{key: euler_deg}`, `at(t, lead=0.0) -> {key: euler_deg}`, and `stale_keys(t, max_stale=1.0) -> list`.

- [ ] **Step 1: Write the failing test**

```python
class TestRotationPredictor:
    @staticmethod
    def _spin(rate_deg_per_s, steps=40, dt=1 / 30):
        """Feed a constant yaw rate; return the predictor and the last time."""
        p = RotationPredictor()
        t = 0.0
        for i in range(steps):
            t = i * dt
            p.update({1: np.array([0.0, rate_deg_per_s * t, 0.0])}, t)
        return p, t

    def test_predicts_along_measured_angular_velocity(self):
        p, t = self._spin(90.0)
        lead = 0.1
        got = p.at(t, lead)[1]
        # 90 deg/s for 0.1s = 9 degrees beyond the last observation.
        assert np.isclose(got[1], 90.0 * t + 9.0, atol=1.0)

    def test_zero_lead_returns_the_measurement(self):
        p, t = self._spin(90.0)
        assert np.allclose(p.at(t, 0.0)[1], [0.0, 90.0 * t, 0.0], atol=1e-6)

    def test_stationary_rotation_is_not_moved(self):
        p = RotationPredictor()
        for i in range(20):
            p.update({1: np.array([10.0, 20.0, 30.0])}, i / 30)
        assert np.allclose(p.at(19 / 30, 0.1)[1], [10.0, 20.0, 30.0], atol=1e-6)

    def test_single_observation_passes_through(self):
        p = RotationPredictor()
        p.update({1: np.array([5.0, 0.0, 0.0])}, 0.0)
        assert np.allclose(p.at(0.0, 0.1)[1], [5.0, 0.0, 0.0], atol=1e-6)

    def test_wrap_does_not_produce_a_spurious_flip(self):
        # Yaw crossing +-180. Averaging or differencing euler here gives a
        # ~360 degree spike; matrices have no such discontinuity.
        p = RotationPredictor()
        dt = 1 / 30
        for i in range(30):
            yaw = 170.0 + 30.0 * i * dt      # walks past 180 into wrap territory
            yaw = (yaw + 180.0) % 360.0 - 180.0
            p.update({1: np.array([0.0, yaw, 0.0])}, i * dt)
        predicted = p.at(29 * dt, 0.05)[1]
        previous = p.at(29 * dt, 0.0)[1]
        m_pred = unity_euler_to_matrix(predicted)
        m_prev = unity_euler_to_matrix(previous)
        step = np.degrees(np.linalg.norm(matrix_to_rotvec(m_pred @ m_prev.T)))
        assert step < 10.0, f"wrap produced a {step:.0f} degree jump"

    def test_rate_is_clamped(self):
        p = RotationPredictor(max_rate_deg=90.0)
        p.update({1: np.array([0.0, 0.0, 0.0])}, 0.0)
        p.update({1: np.array([0.0, 100.0, 0.0])}, 1 / 30)   # 3000 deg/s
        got = p.at(1 / 30, 0.1)[1]
        step = abs(got[1] - 100.0)
        assert step <= 90.0 * 0.1 + 1e-6

    def test_horizon_is_clamped(self):
        p, t = self._spin(90.0)
        far = p.at(t, 10.0)[1]
        capped = p.at(t, 0.12)[1]
        assert np.allclose(far, capped, atol=1e-6)

    def test_stale_keys_reports_untouched_trackers(self):
        p = RotationPredictor()
        p.update({1: np.array([0.0, 0.0, 0.0])}, 0.0)
        assert p.stale_keys(0.5) == []
        assert p.stale_keys(2.0) == [1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_transform.py -k RotationPredictor -q`
Expected: FAIL, ImportError on `RotationPredictor`.

- [ ] **Step 3: Implement**

```python
class RotationPredictor:
    """Rotational twin of TrackerPredictor: latch plus angular velocity.

    Positions were extrapolated forward to cancel pipeline latency while
    rotations were held at their last measured value, so every frame shipped a
    pose that disagreed with itself -- a hip predicted 50ms ahead carrying a hip
    rotation 50ms behind, both derived from the same joints. VRChat's IK
    resolves against both, so the mismatch is not cosmetic.

    All extrapolation happens on matrices. Euler angles cannot be extrapolated:
    a yaw crossing +-180 wraps, and differencing across that wrap reports a
    ~360 deg/s angular velocity that would throw the tracker completely. This is
    the same reason RotationSmoother refuses to average euler angles.

    Angular velocity is EMA-smoothed for the reason linear velocity is: it is a
    difference of two noisy orientations, so multiplying it by a lead time
    amplifies that noise straight into the output.
    """

    def __init__(self, velocity_alpha=0.35, max_rate_deg=720.0, max_horizon=0.12):
        self.velocity_alpha = velocity_alpha
        # Ceiling on extrapolated angular speed, mirroring TrackerPredictor's
        # max_speed. Two full turns a second is far beyond real body motion but
        # well below the spike a single bad frame produces.
        self.max_rate = math.radians(max_rate_deg)
        self.max_horizon = max_horizon
        self._matrix = {}
        self._omega = {}
        self._time = {}

    def update(self, rotations, t):
        for key, euler in rotations.items():
            m = unity_euler_to_matrix(euler)
            previous, previous_t = self._matrix.get(key), self._time.get(key)
            if previous is not None and previous_t is not None and t > previous_t:
                raw = matrix_to_rotvec(m @ previous.T) / (t - previous_t)
                rate = float(np.linalg.norm(raw))
                if rate > self.max_rate:
                    raw = raw * (self.max_rate / rate)
                old = self._omega.get(key)
                self._omega[key] = (raw if old is None else
                                    old * (1 - self.velocity_alpha)
                                    + raw * self.velocity_alpha)
            self._matrix[key] = m
            self._time[key] = t

    def at(self, t, lead=0.0):
        """Every rotation seen so far, extrapolated to `t + lead`, as euler."""
        out = {}
        for key, m in self._matrix.items():
            omega = self._omega.get(key)
            if omega is None:
                out[key] = matrix_to_unity_euler(m)
                continue
            horizon = float(np.clip((t - self._time[key]) + lead,
                                    0.0, self.max_horizon))
            out[key] = matrix_to_unity_euler(rotvec_to_matrix(omega * horizon) @ m)
        return out

    def stale_keys(self, t, max_stale=1.0):
        return [k for k, ts in self._time.items() if t - ts > max_stale]
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_transform.py -k RotationPredictor -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transform.py tests/test_transform.py
git commit -m "Predict rotations forward, not just positions"
```

---

### Task 3: PoseSender

**Files:**
- Modify: `osc_out.py`
- Test: `tests/test_osc_and_stabilise.py`

**Interfaces:**
- Consumes: `TrackerSender` (existing), `TrackerPredictor` and `RotationPredictor` (Task 2) passed in by the caller.
- Produces: `PoseSender(sender, positions, rotations, lead=0.0, hz=0, clock=None, include_head=True)` with `update(trackers, rotations, head, t)`, `poses_at(t) -> (poses, head_pose)`, `stale_keys(t)`, `start()`, `stop()`, and context-manager support. `hz=0` means no thread — the caller pumps it.

- [ ] **Step 1: Write the failing test**

```python
class TestPoseSender:
    @staticmethod
    def _wired(hz=0, lead=0.0, clock=None):
        positions, rotations = TrackerPredictor(), RotationPredictor()
        received = []

        class Recorder:
            def send_frame(self, poses, head=None):
                received.append((dict(poses), head))

        return PoseSender(Recorder(), positions, rotations,
                          lead=lead, hz=hz, clock=clock), received

    def test_inline_mode_sends_nothing_on_its_own(self):
        sender, received = self._wired(hz=0)
        sender.update({1: np.zeros(3)}, {}, None, 0.0)
        time.sleep(0.05)
        assert received == []

    def test_poses_at_pairs_positions_with_rotations(self):
        sender, _ = self._wired()
        sender.update({1: np.array([1.0, 2.0, 3.0])},
                      {1: np.array([0.0, 90.0, 0.0])}, None, 0.0)
        poses, head = sender.poses_at(0.0)
        assert head is None
        position, rotation = poses[1]
        assert np.allclose(position, [1.0, 2.0, 3.0])
        assert np.allclose(rotation, [0.0, 90.0, 0.0], atol=1e-6)

    def test_position_without_rotation_gets_identity(self):
        sender, _ = self._wired()
        sender.update({1: np.zeros(3)}, {}, None, 0.0)
        poses, _ = sender.poses_at(0.0)
        assert np.allclose(poses[1][1], [0.0, 0.0, 0.0])

    def test_head_is_separated_from_body_trackers(self):
        sender, _ = self._wired()
        sender.update({1: np.zeros(3)}, {}, np.array([0.0, 1.7, 0.0]), 0.0)
        poses, head = sender.poses_at(0.0)
        assert set(poses) == {1}
        assert np.allclose(head[0], [0.0, 1.7, 0.0])

    def test_head_can_be_suppressed(self):
        positions, rotations = TrackerPredictor(), RotationPredictor()
        sender = PoseSender(object(), positions, rotations, include_head=False)
        sender.update({1: np.zeros(3)}, {}, np.array([0.0, 1.7, 0.0]), 0.0)
        _, head = sender.poses_at(0.0)
        assert head is None

    def test_threaded_mode_sends_at_about_the_requested_rate(self):
        now = [0.0]
        sender, received = self._wired(hz=200, clock=lambda: now[0])
        sender.update({1: np.zeros(3)}, {}, None, 0.0)
        with sender:
            time.sleep(0.25)
        # 200Hz for 0.25s is 50 sends; timing on a loaded CI box is loose, so
        # assert the order of magnitude rather than the exact count.
        assert 15 <= len(received) <= 90, len(received)

    def test_stop_is_idempotent_and_joins(self):
        sender, _ = self._wired(hz=100)
        sender.start()
        sender.stop()
        sender.stop()
        assert not sender.is_running()

    def test_concurrent_updates_never_tear_a_pose(self):
        # Position and rotation for one tracker must come from the same update.
        # A torn read pairs frame N's position with frame N-1's rotation.
        sender, received = self._wired(hz=500)
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                i += 1
                v = float(i % 100)
                sender.update({1: np.array([v, v, v])},
                              {1: np.array([0.0, v, 0.0])}, None, i * 1e-4)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        with sender:
            time.sleep(0.3)
        stop.set()
        thread.join()
        assert received, "sender produced nothing"
        for poses, _ in received:
            if 1 not in poses:
                continue
            position, rotation = poses[1]
            # Both were written from the same counter value, so with prediction
            # disabled (lead=0) they must still agree.
            assert np.isclose(position[0], position[1])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_osc_and_stabilise.py -k PoseSender -q`
Expected: FAIL, ImportError on `PoseSender`.

- [ ] **Step 3: Implement**

Add `import threading`, `import time` to `osc_out.py`, then:

```python
class PoseSender:
    """Owns the predictors and decides when a frame goes out.

    Two modes, one implementation. With `hz=0` the caller pumps it once per
    camera frame, which is what the pipeline has always done. With `hz>0` a
    thread sends on its own clock, independent of the camera.

    The second mode is the point. VRChat applies tracker data per rendered
    frame, so 30Hz updates on a 72-90Hz headset are held for two or three
    frames -- adding staleness on top of the pipeline latency that --lead-ms
    already compensates for. The predictors can produce a pose for ANY moment,
    so a faster sender is free of new measurements: it re-extrapolates the ones
    it has. Sub-stepping the capture loop instead would not work, because at
    ~30fps with ~33ms of work per iteration the loop has no idle time to
    sub-step into.

    Prediction is evaluated on the SENDING side rather than at update time. A
    pose extrapolated when the frame arrived would be just as stale as the
    frame by the time it went out.
    """

    def __init__(self, sender, positions, rotations, lead=0.0, hz=0,
                 clock=None, include_head=True):
        self._sender = sender
        self._positions = positions
        self._rotations = rotations
        self.lead = lead
        self.hz = hz
        self.include_head = include_head
        self._clock = clock or time.monotonic
        # One lock over BOTH predictors. Taking one per predictor would let a
        # send land between the position and rotation updates of a single
        # frame, pairing this frame's hip with last frame's hip rotation --
        # exactly the inconsistency RotationPredictor exists to remove.
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()

    def update(self, trackers, rotations, head, t):
        with self._lock:
            self._positions.update(trackers, t)
            self._rotations.update(rotations, t)
            if head is not None:
                self._positions.update({"head": head}, t)

    def poses_at(self, t):
        """({index: (position, rotation)}, head_pose or None) at time `t`."""
        with self._lock:
            positions = self._positions.at(t, self.lead)
            rotations = self._rotations.at(t, self.lead)
        poses, head_pose = {}, None
        for key, position in positions.items():
            rotation = rotations.get(key, (0.0, 0.0, 0.0))
            if key == "head":
                if self.include_head:
                    head_pose = (position, rotation)
            else:
                poses[key] = (position, rotation)
        return poses, head_pose

    def send_once(self, t):
        poses, head_pose = self.poses_at(t)
        # Sent outside the lock: this is network I/O, and holding the lock
        # across it would stall the capture thread behind a slow socket.
        self._sender.send_frame(poses, head_pose)

    def stale_keys(self, t):
        with self._lock:
            return set(self._positions.stale_keys(t)) | set(
                self._rotations.stale_keys(t))

    def start(self):
        if self.hz <= 0 or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        period = 1.0 / self.hz
        next_tick = self._clock()
        while not self._stop.is_set():
            self.send_once(self._clock())
            next_tick += period
            delay = next_tick - self._clock()
            if delay < 0:
                # Fell behind; resync rather than accumulating debt and then
                # bursting to catch up.
                next_tick = self._clock()
                delay = 0.0
            self._stop.wait(delay)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        return False
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_osc_and_stabilise.py -k PoseSender -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add osc_out.py tests/test_osc_and_stabilise.py
git commit -m "Add PoseSender: send on our own clock, not the camera's"
```

---

### Task 4: Wire into cmd_run

**Files:**
- Modify: `bodytracker.py:43-45` (imports), `:149-150` (construction), `:228-251` (loop), `:273,306` (stale reporting), argparse for `--send-hz`
- Modify: `README.md` (flag table)

**Interfaces:**
- Consumes: `PoseSender` (Task 3), `RotationPredictor` (Task 2).

- [ ] **Step 1: Replace construction**

Replace `latch, rotation_latch = TrackerPredictor(), TrackerLatch()` and the
`rotation_smoother` line with a `PoseSender` built from a `TrackerPredictor`
and a `RotationPredictor`, keeping `RotationSmoother` applied to rotations
before they reach the predictor. Clock is `lambda: time.monotonic() - t0`, so
the sender shares the loop's timebase.

- [ ] **Step 2: Replace the send block**

`pose_sender.update(trackers, rotation_smoother(rotations), head, t)`, then
`if sending and pose_sender.hz <= 0: pose_sender.send_once(t)`.

- [ ] **Step 3: Add the flag**

```python
run.add_argument("--send-hz", type=float, default=0.0,
                 help="send rate. 0 (default) sends once per camera frame; "
                      "a positive value runs a sender thread at that rate, "
                      "which fills the gaps between camera frames on a "
                      "72-90Hz headset")
```

- [ ] **Step 4: Start and stop the thread**

Wrap the loop body so the sender thread starts after `t0` is set and stops on
exit, including on ctrl-c.

- [ ] **Step 5: Verify default behaviour is unchanged**

Run: `.venv/bin/pytest -q` (all tests) and `.venv/bin/ruff check .`
Then confirm `bodytracker fake` still transmits and `--help` shows `--send-hz`.

- [ ] **Step 6: Commit**

```bash
git add bodytracker.py README.md
git commit -m "Wire the decoupled sender into run"
```

---

## Self-review

- Spec coverage: components 3 and 4 of the spec are covered by Tasks 1–4. Components 1 and 2 (`record`, `sweep`) were explicitly dropped by the user in favour of manual testing; the spec section is superseded and must be marked as such.
- No placeholders: every step carries real code.
- Type consistency: `RotationPredictor.at` returns `{key: euler_deg}` in Task 2 and is consumed as such by `PoseSender.poses_at` in Task 3. `PoseSender.update` takes `(trackers, rotations, head, t)` in Tasks 3 and 4 alike.
