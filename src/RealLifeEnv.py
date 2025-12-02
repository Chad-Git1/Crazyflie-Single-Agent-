"""

Real-world environment wrapper for a Bitcraze Crazyflie that mirrors the
observation/action interface used by the MuJoCo CrazyFlieEnv.

Mapping:
  - Observation vector (13D) = [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
  - Action (4D) = [thrust_scalar, mx, my, mz] (same shape as sim)
    Mix to four motor outputs and convert to PWM/percent for the Crazyflie.

"""

import threading
import time
import numpy as np

try:
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.transport import uri_helper
except Exception:
    Crazyflie = None
    SyncCrazyflie = None
    LogConfig = None
    uri_helper = None

class CrazyflieRealEnv:
    """
    Real environment wrapper exposing:
      - connect(uri)
      - reset() -> obs
      - step(action) -> obs, reward_estimate, done, info
      - get_obs() -> latest 13D obs
      - estop(): emergency stop (cuts motors)
    """

    # --- TUNE THESE CONSTANTS BEFORE FIRST FLIGHT ---
    CONTROL_HZ = 100.0               # loop rate for sending commands
    HOVER_THRUST_SCALE = 1.0         # how we map your policy's 'thrust' to motor mix (scale factor)
    PWM_BASE = 42000                 # example hover PWM for CF2 (very implementation dependent)
    PWM_SCALE = 60000                # maps [-1, +1] motor mix output to PWM range (change carefully)
    PWM_MIN = 20000                  # conservative min PWM (no motor = 0 on some firmwares — tune carefully)
    PWM_MAX = 65535                  # conservative max PWM for your hardware
    SAFETY_MAX_TILT_DEG = 45.0       # tilt threshold to cut motors
    SAFE_ALTITUDE_MIN = 0.01         # meters: if sensor reports below this, treat as near ground
    # -------------------------------------------------

    def __init__(self, uri: str, use_logging=True, safe_mode=True):
        if Crazyflie is None:
            raise RuntimeError("cflib not available; install with `pip install cflib`")

        self.uri = uri
        self._scf = None 
        self._cf = None
        self._log_conf = None
        self._log_thread = None
        self._running = False
        self._connected = False
        self._use_logging = use_logging
        self._safe_mode = bool(safe_mode)

        # Latest sensor-derived state (13D) stored here
        self._obs_lock = threading.Lock()
        self._latest_obs = np.zeros(13, dtype=np.float32)

        self._last_log_ts = 0.0
        self._hover_thrust = 0.27  # approximate hover thrust from sim; used only for scaling hints
        self._motor_scale = 1.0    # to reflect any per-episode motor scaling (if used)

        # E-stop
        self._estopped = False

        # last motor values (for monitoring)
        self._last_motors = np.zeros(4, dtype=np.float32)

    # -------------------------
    # Connection & Logging
    # -------------------------
    def connect(self, timeout_s=10.0):
        """Open link and start logging telemetry. Blocks until connected or timeout."""
        if self._connected:
            return

        # Create SyncCrazyflie and open link
        self._scf = SyncCrazyflie(self.uri, cf=Crazyflie())
        self._scf.open_link()
        self._cf = self._scf.cf
        self._connected = True

        if not self._use_logging:
            return

        # Build a LogConfig - try to include multiple commonly-available keys.
        log_conf = LogConfig(name="state", period_in_ms=20)  # 50 Hz default; we'll sample faster in control loop
        # Add candidate variables; many Crazyflie firmwares expose these names.
        # We'll read several variants and the logger callback will use whatever is present.
        keys = [
            "stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
            "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz",
            "stateEstimate.qw", "stateEstimate.qx", "stateEstimate.qy", "stateEstimate.qz",
            "stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw",
            "gyro.x", "gyro.y", "gyro.z",
            "acc.x", "acc.y", "acc.z",
        ]
        # Only add keys that exist on the Crazyflie (SyncLogger will ignore unknown keys,
        # but some older firmwares may error — so we wrap add_variable).
        for k in keys:
            try:
                log_conf.add_variable(k)
            except Exception:
                # ignore variables not present on the connected firmwares
                pass

        self._log_conf = log_conf

        # Start SyncLogger in a thread (so we can handle async updates)
        self._running = True
        self._log_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self._log_thread.start()

        # Give some time for first logs to populate
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self._obs_lock:
                # if z has become non-zero we assume logging populated
                if abs(self._latest_obs[2]) > 1e-6:
                    return
            time.sleep(0.05)
        # timed out, but connection is open; still use it
        return

    def _logging_worker(self):
        """Runs SyncLogger and updates self._latest_obs whenever a packet arrives."""
        if self._scf is None or self._log_conf is None:
            return
        try:
            with self._scf as scf:
                with scf.cf.log.create_log(self._log_conf) as log:
                    for timestamp, data in log:
                        # Build 13D observation from whatever fields are available.
                        # Use heuristics + fallbacks for keys that vary by firmware.
                        obs = self._build_obs_from_packet(data)
                        with self._obs_lock:
                            self._latest_obs = obs
                            self._last_log_ts = timestamp / 1e6
                        if not self._running:
                            break
        except Exception as e:
            # logger ended or had an error
            print("Crazyflie logging worker stopped:", e)
        finally:
            self._running = False

    def _build_obs_from_packet(self, data: dict) -> np.ndarray:
        """
        Convert a telemetry dict into the 13D vector:
        [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
        We fall back gracefully when some keys are missing.
        """
        # position
        x = data.get("stateEstimate.x", data.get("state.x", 0.0))
        y = data.get("stateEstimate.y", data.get("state.y", 0.0))
        z = data.get("stateEstimate.z", data.get("state.z", 0.0))

        # quaternion
        qw = data.get("stateEstimate.qw", data.get("state.qw",
             data.get("stabilizer.qw", 1.0)))
        qx = data.get("stateEstimate.qx", data.get("state.qx", 0.0))
        qy = data.get("stateEstimate.qy", data.get("state.qy", 0.0))
        qz = data.get("stateEstimate.qz", data.get("state.qz", 0.0))

        # linear velocity
        vx = data.get("stateEstimate.vx", data.get("state.vx", data.get("vel.x", 0.0)))
        vy = data.get("stateEstimate.vy", data.get("state.vy", data.get("vel.y", 0.0)))
        vz = data.get("stateEstimate.vz", data.get("state.vz", data.get("vel.z", 0.0)))

        # angular rates: try gyro first
        wx = data.get("gyro.x", data.get("state.wz", 0.0))
        wy = data.get("gyro.y", data.get("state.wx", 0.0))
        wz = data.get("gyro.z", data.get("state.wy", 0.0))

        # Pack and ensure dtype
        obs = np.array([x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz], dtype=np.float32)
        # ensure quaternion is normalized (if present)
        q = obs[3:7]
        nq = np.linalg.norm(q)
        if nq > 1e-6:
            obs[3:7] = q / nq
        return obs

    #  API
    def reset(self):
        with self._obs_lock:
            obs = self._latest_obs.copy()
        return obs

    def get_obs(self):
        with self._obs_lock:
            return self._latest_obs.copy()

    def step(self, action: np.ndarray):
        
        if self._estopped:
            return self.get_obs(), 0.0, True, {"estop": True}

        # Map action -> motors & send
        motors = self._action_to_motors(action)
        self._send_motor_pwm(motors)

        # small sleep to allow motors to apply — control loop should call step at CONTROL_HZ
        # return latest obs
        obs = self.get_obs()
        # optionally include safety checks
        info = {}
        done = False
        # tilt safety
        tilt = self._tilt_from_quat(obs[3:7])
        if self._safe_mode and (np.rad2deg(tilt) > self.SAFETY_MAX_TILT_DEG):
            print("[SAFETY] Tilt exceeded; estopping motors.")
            self.estop()
            done = True
            info["crash"] = "tilt_exceed"
        # ground safety
        if obs[2] < 0.0 and self._safe_mode:
            print("[SAFETY] Negative altitude reading; stopping.")
            self.estop()
            done = True
            info["ground"] = True
        return obs, 0.0, done, info

    # -------------------------
    # Action -> motors / send
    # -------------------------
    def _action_to_motors(self, action: np.ndarray) -> np.ndarray:
        """
        Mix [thrust, mx, my, mz] -> 4 motor mix outputs in [-1, 1] range.
        Standard quad X-configuration mixing:
            m1 = thrust + mx + my - mz
            m2 = thrust - mx + my + mz
            m3 = thrust - mx - my - mz
            m4 = thrust + mx - my + mz
        The agent's thrust is expected to be within the sim's action bounds (e.g. [0, 0.35]).
        We normalize accordingly and output (motor1..4) in [-1,1].
        """
        a = np.asarray(action, dtype=np.float32).squeeze()
        if a.shape != (4,):
            raise ValueError("Action must be shape (4,)")

        thrust = float(a[0])
        mx, my, mz = float(a[1]), float(a[2]), float(a[3])

        # Simple mixing
        m1 = thrust + mx + my - mz
        m2 = thrust - mx + my + mz
        m3 = thrust - mx - my - mz
        m4 = thrust + mx - my + mz

        mot = np.array([m1, m2, m3, m4], dtype=np.float32)

        # Normalize mot to [-1,1] centered around the hover_thrust (if thrust is positive)
        # Here we assume thrust values are small positive numbers; scale by HOVER_THRUST_SCALE if needed
        # We'll shift/scale later when converting to PWM
        # Clip for safety to avoid runaway numbers
        mot = np.clip(mot, -2.0, 2.0)

        return mot

    def _send_motor_pwm(self, motors: np.ndarray):
        """Convert [-2..2] motor mix to PWM and send to Crazyflie via commander."""
        if self._scf is None or self._cf is None:
            return

        # Simple mapping:
        # normalized motor in [-2,2] -> scale to PWM range using PWM_SCALE and base
        # you MUST tune PWM_BASE and PWM_SCALE per your drone and bench tests
        scaled = motors * (self.PWM_SCALE / 2.0)  # motors of 2.0 -> PWM_SCALE
        pwm = (self.PWM_BASE + scaled).astype(np.int32)

        # Clip to safe PWM range
        pwm = np.clip(pwm, self.PWM_MIN, self.PWM_MAX)

        # Save last
        self._last_motors = pwm

        # Send using commander. Some firmwares provide send_motor_power() or send_setpoint() APIs.
        # We'll try several safe approaches.
        try:
            # Newer cflib: commander.send_motor_power expects floats [0..1], but many firmwares differ.
            # We attempt to call a motor PWM method if available; otherwise use setpoint (roll,pitch,yaw,thrust).
            if hasattr(self._cf.commander, "send_motor_power"):
                # convert pwm to 0..1 fraction (best-effort)
                # This conversion is hardware-specific; you will almost certainly replace/adapt it.
                motor_frac = (pwm - self.PWM_MIN) / float(self.PWM_MAX - self.PWM_MIN)
                motor_frac = np.clip(motor_frac, 0.0, 1.0)
                # send floats
                try:
                    self._cf.commander.send_motor_power(motor_frac[0], motor_frac[1], motor_frac[2], motor_frac[3])
                except TypeError:
                    # some implementations expect int
                    self._cf.commander.send_motor_power(int(motor_frac[0]*1000), int(motor_frac[1]*1000),
                                                        int(motor_frac[2]*1000), int(motor_frac[3]*1000))
            else:
                # Fallback: try send_setpoint (roll, pitch, yawrate, thrust)
                # Warning: this goes through the stabilizer; not raw motors. Use only for safe tests.
                thrust_pct = np.clip((pwm.mean() - self.PWM_MIN) / (self.PWM_MAX - self.PWM_MIN), 0.0, 1.0)
                # send_setpoint expects roll, pitch, yawrate, thrust (thrust often 0..60000)
                try:
                    self._cf.commander.send_setpoint(0, 0, 0, int(thrust_pct * 60000))
                except Exception:
                    # last resort: a direct parameter write is not attempted here
                    pass
        except Exception as e:
            print("Error sending motor command:", e)

    # -------------------------
    # Safety / utilities
    # -------------------------
    def estop(self):
        """Emergency stop — cut motors immediately."""
        self._estopped = True
        try:
            if self._cf is not None and hasattr(self._cf.commander, "send_motor_power"):
                # set zeros
                try:
                    self._cf.commander.send_motor_power(0.0, 0.0, 0.0, 0.0)
                except Exception:
                    pass
        except Exception:
            pass

    def _tilt_from_quat(self, q: np.ndarray):
        """Compute tilt angle (rad) from quaternion q = [qw,qx,qy,qz] (ignoring yaw)."""
        qw, qx, qy, qz = q
        sin_tilt = np.sqrt(qx * qx + qy * qy)
        sin_tilt = np.clip(sin_tilt, 0.0, 1.0)
        tilt_angle = 2.0 * np.arcsin(sin_tilt)
        return tilt_angle

    def close(self):
        """Shutdown logger and close link."""
        self._running = False
        time.sleep(0.05)
        if self._scf is not None:
            try:
                self._scf.close_link()
            except Exception:
                pass
        self._connected = False
