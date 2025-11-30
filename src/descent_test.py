# slow_descent_demo.py
#
# Standalone MuJoCo script:
# - Loads Crazyflie model
# - Starts it in the air
# - Uses a simple PD controller to:
#       * keep it upright
#       * slowly descend straight down
# - No RL, no Gym – just physics + control.

import os
import time
import numpy as np
import mujoco
from mujoco import viewer


def main():
    here = os.path.dirname(__file__)
    xml_path = os.path.abspath(
        os.path.join(here, "..", "Assets", "bitcraze_crazyflie_2", "scene.xml")
    )

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    # Load model + data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # --- Initial state: start in the air, upright, zero velocity ---
    data.qpos[:] = np.array([0.0, 0.0, 1.0,   # x, y, z  (1m high)
                             1.0, 0.0, 0.0, 0.0],  # qw, qx, qy, qz (upright)
                            dtype=np.float64)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    # --- Control parameters ---
    # These should feel familiar: similar to your landing controller.
    HOVER_THRUST = 0.27          # approx hover thrust from your env
    TMIN, TMAX = 0.0, 0.35       # thrust limits from your XML
    M_MIN, M_MAX = -1.0, 1.0     # moment range for [mx, my, mz]

    frame_skip = 10              # how many physics steps per control step
    dt_sim = model.opt.timestep
    dt_step = dt_sim * frame_skip

    # Descent & attitude gains
    landing_descent_rate = 0.2   # m/s downward (slow)
    kq = 6.0                     # attitude (qx, qy) proportional gain
    kw = 1.0                     # angular velocity (wx, wy) damping
    k_v = 0.7                    # vertical velocity gain

    # We’ll keep a target vertical velocity v_des and try to track it.
    v_des = -abs(landing_descent_rate)

    print("Starting slow descent demo...")
    print("Controls: thrust = ctrl[0], moments = ctrl[1:4]")

    with viewer.launch_passive(model, data) as v:
        t0 = time.time()
        last_print = t0

        while v.is_running():
            # Current state
            x, y, z = data.qpos[0:3]
            qw, qx, qy, qz = data.qpos[3:7]
            vx, vy, vz = data.qvel[0:3]
            wx, wy, wz = data.qvel[3:6]

            # --- Compute tilt angle (ignore yaw) ---
            tilt_sin = np.sqrt(qx * qx + qy * qy)
            tilt_sin = np.clip(tilt_sin, 0.0, 1.0)
            tilt_angle = 2.0 * np.arcsin(tilt_sin)  # radians

            # -------------------------
            # 1) Vertical control: track v_des (slow downward speed)
            # -------------------------
            # error in vertical velocity
            err_v = float(vz - v_des)
            # thrust around hover to correct vertical speed
            u = HOVER_THRUST - k_v * err_v

            # clamp thrust
            u = float(np.clip(u, TMIN, TMAX))

            # -------------------------
            # 2) Attitude control: keep upright
            # -------------------------
            mx = -kq * float(qx) - kw * float(wx)
            my = -kq * float(qy) - kw * float(wy)
            mz = -0.3 * float(wz)  # just damp yaw rate

            m = np.array([mx, my, mz], dtype=np.float32)
            m = np.clip(m, M_MIN, M_MAX)

            # -------------------------
            # 3) Near-ground behavior
            # -------------------------
            if z < 0.05 and abs(vz) < 0.2 and tilt_angle < np.deg2rad(10.0):
                # almost landed: cut thrust & moments
                u = 0.0
                m[:] = 0.0

            # Apply to ctrl
            data.ctrl[0] = u
            if model.nu >= 4:
                data.ctrl[1:4] = m
                if model.nu > 4:
                    data.ctrl[4:] = 0.0

            # Step physics
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)

            # Update viewer
            v.sync()

            # Print status ~1 Hz
            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                print(
                    f"t={now - t0:4.1f}s | z={z:+.3f} m  vz={vz:+.3f} m/s  "
                    f"tilt={np.rad2deg(tilt_angle):5.1f} deg  "
                    f"u={u:.3f}  (x,y)=({x:+.3f},{y:+.3f})"
                )

            # Stop once really landed and settled
            if z < 0.03 and abs(vz) < 0.05 and tilt_angle < np.deg2rad(10.0):
                print("Landed safely. Stopping simulation.")
                break

            # Keep real-time pace
            time.sleep(dt_step)


if __name__ == "__main__":
    main()
