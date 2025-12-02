"""
python run_real_eval.py --uri radio://0/80/2M/E7E7E7E7E7

Replace the URI with the real Crazyflie radio URI.
"""

import argparse
import time
import numpy as np
import signal
import threading
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from RealLifeEnv import CrazyflieRealEnv

# Shutown flag
_shutdown = False

def _signal_handler(sig, frame):
    global _shutdown
    print("Signal received, shutting down...")
    _shutdown = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def load_model_and_norm(model_path: str, norm_path: str):
    """Load PPO and VecNormalize"""
    model = PPO.load(model_path)
    # Create a tiny DummyVecEnv wrapper to load VecNormalize state
    dummy = DummyVecEnv([lambda: None])
    vecnorm = VecNormalize.load(norm_path, dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return model, vecnorm

def main(args):
    model_path = args.model
    vecnorm_path = args.vecnorm
    uri = args.uri

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.isfile(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")

    model, vecnorm = load_model_and_norm(model_path, vecnorm_path)

    # Create real env and connect
    env = CrazyflieRealEnv(uri, use_logging=True, safe_mode=True)
    print("Connecting to Crazyflie...")
    env.connect()

    # Wait a short time for the logger to populate sensor values
    print("Waiting for initial telemetry...")
    t0 = time.time()
    while time.time() - t0 < 5.0:
        obs = env.get_obs()
        if np.any(np.isfinite(obs)) and abs(obs[2]) > 1e-6:
            break
        time.sleep(0.05)

    # main control loop
    ctrl_dt = 1.0 / env.CONTROL_HZ
    print(f"Starting control loop at ~{env.CONTROL_HZ:.1f} Hz. Ctrl-C to stop.")
    ep_return = 0.0
    step = 0

    # Safety: start with very conservative scaling factor
    thrust_scale = 1.0  # you may multiply policy thrust by small factor at first (0.5..1.0)

    try:
        while not _shutdown:
            obs_raw = env.get_obs()  # shape (13,)
            # vecnorm.normalize_obs expects shape (n_envs, obs_dim)
            obs_norm = vecnorm.normalize_obs(obs_raw[None, :])  # (1, obs_dim)
            action, _ = model.predict(obs_norm, deterministic=True)
            action = np.asarray(action, dtype=np.float32).squeeze()
            # If model outputs shape (1,4), squeeze to (4,)
            if action.ndim == 2:
                a = action[0]
            else:
                a = action

            # Optional: scale thrust down for first tests
            a[0] = a[0] * thrust_scale

            # Send to env, which will mix -> motors -> send
            obs, reward, done, info = env.step(a)
            ep_return += float(reward)
            step += 1

            # Print periodic telemetry
            if step % int(env.CONTROL_HZ) == 0:
                print(f"[{step}] z={obs[2]:+.3f} m  vz={obs[9]:+.3f} m/s  return={ep_return:.2f} info={info}")

            # Safety: break if estopped or done
            if done:
                print("Controller stopping (done/estop):", info)
                break

            # sleep to match real-time
            time.sleep(ctrl_dt)

    except KeyboardInterrupt:
        print("Keyboard interrupt, shutting down.")

    finally:
        # Ensure motors off
        print("Shutting down: estopping and closing link.")
        env.estop()
        time.sleep(0.1)
        env.close()
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to PPO model .zip")
    parser.add_argument("--vecnorm", type=str, required=True, help="Path to vecnormalize .pkl")
    parser.add_argument("--uri", type=str, required=True, help="Crazyflie radio URI (e.g. radio://0/80/2M/E7E7E7E7E7)")
    args = parser.parse_args()
    main(args)
