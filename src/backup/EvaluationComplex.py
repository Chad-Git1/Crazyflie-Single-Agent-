
import os
import time
import numpy as np
import mujoco
from mujoco import viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

##change xml value for thrust range
##parallel environment implementation
##introduce complexity to action space 
##for drone api figure out which things we can control
##use weights and biases then run hyperparameter suite over night
#reduce entropy gain/loss
##reduce episode length and increase episode count
## have at least 3 seeds

##in action space add penalty for difference in previous and current action to prevent large thrust
##for input to pokicy add a history of states called frame stacking
##next 2 weeks, add guasian noise for domain randomization
##alter mujocco step for 10 steps rather than 1 physics step



##change xml value for thrust range
##parallel environment implementation
##introduce complexity to action space 
##for drone api figure out which things we can control
##use weights and biases then run hyperparameter suite over night
#reduce entropy gain/loss
##reduce episode length and increase episode count
## have at least 3 seeds

##in action space add penalty for difference in previous and current action to prevent large thrust
##for input to pokicy add a history of states called frame stacking
##next 2 weeks, add guasian noise for domain randomization
##alter mujocco step for 10 steps rather than 1 physics step


##Week 10
##look at seeds for hyperparameter tuning 
##randomize hover heights and phases to help drone learnign with height 
##add some noises
##domain randomization
##randomize the observations that came back from the drone, randomize the actions, randomize the the physics steps
##add safety controls for certain pitch/rolls to catch issues or if it goes past a certain point, limit
##


##add some gausian noise(randomize actions, rnadomize the physcs steps etc.)
##do hyperparameter sweep over night with some seeds
##add safety controls for certain pitch/rolls to catch issues or if it goes past a certain point
##work on actual environment

from CrazyFlieEnvComplex import CrazyFlieEnv



import numpy as np

def level_action(env, z_ref: float) -> tuple[np.ndarray, float]:
    """
    Phase 1: attitude leveling while roughly holding altitude.

    Returns:
        action: np.array([thrust, mx, my, mz])
        tilt_angle: current tilt in radians
    """
    # Read clean state from MuJoCo
    x, y, z = env.data.qpos[0:3]
    qw, qx, qy, qz = env.data.qpos[3:7]
    vx, vy, vz = env.data.qvel[0:3]
    wx, wy, wz = env.data.qvel[3:6]

    # --- Tilt angle (roll/pitch only) ---
    tilt_sin = np.sqrt(qx * qx + qy * qy)
    tilt_sin = np.clip(tilt_sin, 0.0, 1.0)
    tilt_angle = 2.0 * np.arcsin(tilt_sin)

    # --- Attitude PD: kill tilt and angular rates (roll/pitch) ---
    # Gains are deliberately modest; we also clamp torques.
    kq = 4.0   # position gain
    kw = 1.0   # rate gain
    m_max = 0.3

    mx = -kq * float(qx) - kw * float(wx)
    my = -kq * float(qy) - kw * float(wy)
    mz = -0.2 * float(wz)   # very light yaw damping

    m = np.array([mx, my, mz], dtype=np.float32)
    m = np.clip(m, -m_max, m_max)

    # --- Vertical PD around z_ref: try not to drop while leveling ---
    # We only want to hold altitude roughly constant here.
    k_z = 2.0
    k_vz = 0.8
    err_z = z_ref - z       # positive if below reference

    u = env.HOVER_THRUST + k_z * err_z - k_vz * vz
    u = float(np.clip(u, env.action_space.low[0], env.action_space.high[0]))

    action = np.array([u, m[0], m[1], m[2]], dtype=np.float32)
    return action, tilt_angle


def descent_action(env) -> tuple[np.ndarray, float]:
    """
    Phase 2: gentle descent with attitude stabilization.

    Returns:
        action: np.array([thrust, mx, my, mz])
        tilt_angle: current tilt in radians
    """
    # Read clean state
    x, y, z = env.data.qpos[0:3]
    qw, qx, qy, qz = env.data.qpos[3:7]
    vx, vy, vz = env.data.qvel[0:3]
    wx, wy, wz = env.data.qvel[3:6]

    # Tilt
    tilt_sin = np.sqrt(qx * qx + qy * qy)
    tilt_sin = np.clip(tilt_sin, 0.0, 1.0)
    tilt_angle = 2.0 * np.arcsin(tilt_sin)

    # Attitude PD (same idea, can be a bit softer)
    kq = 3.0
    kw = 0.8
    m_max = 0.3

    mx = -kq * float(qx) - kw * float(wx)
    my = -kq * float(qy) - kw * float(wy)
    mz = -0.2 * float(wz)

    m = np.array([mx, my, mz], dtype=np.float32)
    m = np.clip(m, -m_max, m_max)

    # Vertical speed profile: descend like an elevator
    h = z - env.safe_ground_height()
    if h < 0.0:
        h = 0.0

    if h > 0.8:
        v_des = -0.30
    elif h > 0.4:
        v_des = -0.22
    elif h > 0.2:
        v_des = -0.16
    else:
        v_des = -0.10   # very gentle near ground

    k_v = 0.4
    err_v = vz - v_des
    u = env.HOVER_THRUST - k_v * err_v

    # Keep some minimum thrust to avoid free-fall
    u_min = 0.12
    u_max = env.action_space.high[0]
    u = float(np.clip(u, u_min, u_max))

    action = np.array([u, m[0], m[1], m[2]], dtype=np.float32)
    return action, tilt_angle


# Build a tiny vec env only to load VecNormalize stats
def _make_norm_loader(xml_path: str, target_z: float, max_steps: int):##factor function to load the environemnt
    def _thunk():
        return Monitor(CrazyFlieEnv(xml_path=xml_path, target_z=target_z, max_steps=max_steps,hover_required_steps=600))
    return DummyVecEnv([_thunk])


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    xml_path = os.path.abspath(os.path.join(here, "..", "Assets", "bitcraze_crazyflie_2", "scene.xml"))
    ##path to models and the specific model zip file
    models_dir = os.path.abspath(os.path.join(here, "..", "models", "Complex2_DR"))
    model_path = os.path.join(models_dir, "complex_dr.zip")
    norm_path  = os.path.join(models_dir, "vecnormalize_dr.pkl")

    TARGET_Z  = 1
    MAX_STEPS = 1500

    #this loads the trained model(which is the neural network weights and training hyperparameters fromt he disk)
    model = PPO.load(model_path)

    ##Load VecNormalize stats so we can feed it to the model we loaded so we can normalize observations manually
    norm_loader = _make_norm_loader(xml_path, TARGET_Z, MAX_STEPS)
    vecnorm = VecNormalize.load(norm_path, norm_loader)
    vecnorm.training = False ##freezes normalization statistics to prevent updating
    vecnorm.norm_reward = False##don't normalize rewards for real simulation

    ##Create a single (non-vectorized) env so we can open the live viewer
    env = CrazyFlieEnv(
        xml_path=xml_path,
        target_z=TARGET_Z,
        max_steps=MAX_STEPS,
        n_stack=4,
        hover_required_steps=600,
       obs_noise_std=0.0,      # base scale for white noise
    obs_bias_std=0.0,       # episode-level offsets
    action_noise_std=0.0,   # very small jitter
    motor_scale_std=0.0,    # ±3% gain
    frame_skip=10,
    frame_skip_jitter=0,     # [9, 11]


        auto_landing=True,
    landing_descent_rate=0.2,    # nice and slow
    landing_upright_gain=1.0,    # stronger upright
    landing_rate_gain=0.7,       # stronger rate damping
       
    )
## we don't wrap it with dummyVecEnv because mujocco only works with a single env
    obs_raw, _ = env.reset()
    dt_sim = env.model.opt.timestep
    dt_step = dt_sim * env.frame_skip 

    #  Launch MuJoCo's interactive viewer; step the env and sync the window
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        phase = "HOVER"          # "HOVER" -> "LANDING" -> "LANDED"
        landing_step = 0
        landing_steps = 600      # more steps = slower, smoother descent

        terminated = False
        truncated = False
        ep_return = 0.0

        t0 = time.time()
        last_print = t0

        while phase != "LANDED":
            if phase == "HOVER":
                # ---- RL controls hover ----
                obs_norm = vecnorm.normalize_obs(obs_raw[None, :])
                action, _ = model.predict(obs_norm, deterministic=True)

                action = np.asarray(action, dtype=np.float32)
                if action.ndim == 2:
                    a = action[0]
                else:
                    a = action

                obs_raw, reward, terminated, truncated, info = env.step(a)
                ep_return += float(reward)

                done = terminated or truncated
                if done:
                    phase = "LANDING"
                    landing_step = 0

            elif phase == "LANDING":
                # ---- Thrust-only vertical-speed landing ----
                landing_action = env.landing_action(landing_step, landing_steps)
                obs_raw, _, _, _, _ = env.step(landing_action)
                landing_step += 1

                z  = env.get_altitude()
                vz = float(env.data.qvel[2])

                low_enough  = (z <= env.safe_ground_height())
                slow_enough = (abs(vz) < 0.10)   # want it really gentle near ground

                if (low_enough and slow_enough) or (landing_step >= landing_steps):
                    env.cut_motors()
                    phase = "LANDED"

            # Viewer
            viewer.sync()

            # HUD
            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                z  = env.get_altitude()
                vz = float(env.data.qvel[2])
                R  = float(ep_return)
                print(
                    f"t={int(now - t0):2d}s | "
                    f"phase={phase} "
                    f"z={z:+.3f} m  vz={vz:+.3f} m/s  "
                    f"thrust={env.u_cmd:.3f} "
                    f"R={R:.1f} "
                    f"info={info}"
                )

            time.sleep(dt_step)

    print(
        f"\nEpisode finished. Return={ep_return:.2f}, "
        f"terminated={terminated}, truncated={truncated}, info: {info}"
    )
