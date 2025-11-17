# src/train_thrust_ppo_dr.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from CrazyFlieEnvComplex import CrazyFlieEnv


# factory stays the same – still uses DR_PARAMS
def make_env(xml_path: str, target_z: float, max_steps: int = 1500, rank: int = 0):
    def _f():
        env = CrazyFlieEnv(
            xml_path=xml_path,
            target_z=target_z,
            max_steps=max_steps,
            n_stack=4,
            hover_required_steps=600,
            **DR_PARAMS,  # will now be NON-ZERO noise
        )
        env = Monitor(env)
        env.reset(seed=rank)
        return env
    return _f


if __name__ == "__main__":
    here = os.path.dirname(__file__)

    # ---------- 1) Set STRONGER DR params for Stage 2 ----------
    DR_PARAMS = dict(
        obs_noise_std=0.007,
        obs_bias_std=0.01,
        action_noise_std=0.008,
        motor_scale_std=0.05,  # ±2% thrust gain
        frame_skip=10,
        frame_skip_jitter=2,   # frame skip in [9, 11]
    )

    # MuJoCo XML path
    xml_path = os.path.abspath(os.path.join(here, "..", "Assets", "bitcraze_crazyflie_2", "scene.xml"))

    # ---------- 2) Paths to OLD (Stage 1) model + vecnorm ----------
    base_models_dir = os.path.abspath(os.path.join(here, "..", "models", "Complex2"))
    old_model_path = os.path.join(base_models_dir, "complex.zip")
    old_vecnorm_path = os.path.join(base_models_dir, "vecnormalize.pkl")

    # ---------- 3) NEW paths to save Stage 2 noisy model ----------
    models_dir = os.path.abspath(os.path.join(here, "..", "models", "Complex2_DR"))
    logs_dir = os.path.abspath(os.path.join(here, "..", "logsComplex2_DR"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Env constants
    TARGET_Z = 1.0
    MAX_STEPS = 1500
    N_ENVS = 8

    # ---------- 4) Build noisy envs ----------
    env_fns = [make_env(xml_path, TARGET_Z, MAX_STEPS, rank=i) for i in range(N_ENVS)]
    venv = DummyVecEnv(env_fns=env_fns)

    # ---------- 5) Load OLD VecNormalize stats on top of this venv ----------
    venv = VecNormalize.load(old_vecnorm_path, venv)
    # Make sure it's in training mode and we keep updating stats
    venv.training = True
    venv.norm_reward = True

    # ---------- 6) Load OLD PPO model, attach NEW noisy env ----------
    model = PPO.load(old_model_path, env=venv)



    # ---------- 7) Continue training with noise ----------
    model.learn(
        total_timesteps=5_000_000,   # extra steps for fine-tuning
        progress_bar=True,
        reset_num_timesteps=False,   # <--- IMPORTANT: continue timestep count
    )

    # ---------- 8) Save new noisy-model + updated vecnorm ----------
    model.save(os.path.join(models_dir, "complex_dr.zip"))
    venv.save(os.path.join(models_dir, "vecnormalize_dr.pkl"))
    print(f"Saved DR model + VecNormalize into {models_dir}")
