import os
import random
import numpy as np
import torch as th

import wandb
from wandb.integration.sb3 import WandbCallback

from tqdm import trange

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from CrazyFlieEnvComplex import CrazyFlieEnv


# ---------- CLEAN ENV: domain randomization OFF FOR TUNING ----------
DR_PARAMS = dict(
    obs_noise_std=0.0,
    obs_bias_std=0.0,
    action_noise_std=0.0,
    motor_scale_std=0.0,
    frame_skip=10,
    frame_skip_jitter=0,
)


def make_env(xml_path: str, target_z: float, max_steps: int, rank: int, base_seed: int):
    """Factory that creates a clean env with per-env seeding."""
    def _f():
        env = CrazyFlieEnv(
            xml_path=xml_path,
            target_z=target_z,
            max_steps=max_steps,
            n_stack=4,
            hover_required_steps=600,
            **DR_PARAMS,
        )
        env = Monitor(env)
        env.reset(seed=base_seed + rank)
        return env
    return _f


def train():
    """One training run for a single hyperparameter set (one sweep trial)."""
    here = os.path.dirname(__file__)

    # Paths
    xml_path = os.path.abspath(os.path.join(here, "..", "Assets", "bitcraze_crazyflie_2", "scene.xml"))
    base_models_dir = os.path.abspath(os.path.join(here, "..", "models", "Complex2_sweep"))
    base_logs_dir = os.path.abspath(os.path.join(here, "..", "logsComplex2_sweep"))
    os.makedirs(base_models_dir, exist_ok=True)
    os.makedirs(base_logs_dir, exist_ok=True)

    TARGET_Z = 1.0
    MAX_STEPS = 1500
    N_ENVS = 8

    # ----- 1) Init wandb run -----
    run = wandb.init(
        project="crazyflie-ppo-clean",
        config={
            "total_timesteps": 1_000_000,   # training per trial
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.001,
            "vf_coef": 0.5,
            "hidden_size": 64,
            "seed": 0,
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    config = wandb.config
    seed = int(config.seed)

    # ----- 2) Global seeding -----
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ----- 3) Build vectorized clean env -----
    env_fns = [
        make_env(xml_path, TARGET_Z, MAX_STEPS, rank=i, base_seed=seed)
        for i in range(N_ENVS)
    ]
    venv = DummyVecEnv(env_fns=env_fns)
    try:
        venv.seed(seed)
    except AttributeError:
        pass

    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # Per-run TB log dir
    logs_dir = os.path.join(base_logs_dir, run.name)
    os.makedirs(logs_dir, exist_ok=True)

    # ----- 4) PPO model using hyperparams from wandb config -----
    policy_kwargs = dict(net_arch=[config.hidden_size, config.hidden_size])

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=logs_dir,
        verbose=1,
        seed=seed,  # SB3 internal RNG
    )

    # ----- 5) Train (progress bar at sweep level, so keep this off) -----
    model.learn(
        total_timesteps=config.total_timesteps,
        progress_bar=False,
        callback=WandbCallback(
            gradient_save_freq=0,
            model_save_path=base_models_dir,
            model_save_freq=0,
            verbose=1,
        ),
    )

    # ----- 6) Save final model + VecNormalize for this run -----
    model_path = os.path.join(base_models_dir, f"{run.name}_final.zip")
    vecnorm_path = os.path.join(base_models_dir, f"{run.name}_vecnorm.pkl")
    model.save(model_path)
    venv.save(vecnorm_path)

    print(f"[{run.name}] Saved final model to {model_path}")
    print(f"[{run.name}] Saved VecNormalize to {vecnorm_path}")

    run.finish()


if __name__ == "__main__":
    # ----- 7) Sweep config -----
    sweep_config = {
        "method": "bayes",  # can change to "random" if you like
        "metric": {
            "name": "rollout/ep_rew_mean",
            "goal": "maximize",
        },
        "parameters": {
            "total_timesteps": {"value": 1_500_000},
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-3,
            },
            "n_steps": {"values": [1024, 2048, 4096]},
            "batch_size": {"values": [64, 128, 256]},
            "gamma": {"values": [0.98, 0.99, 0.995]},
            "gae_lambda": {"min": 0.9, "max": 0.99},
            "clip_range": {"values": [0.1, 0.2, 0.3]},
            "ent_coef": {"values": [0.0, 0.001, 0.01]},
            "vf_coef": {"values": [0.3, 0.5, 0.7]},
            "hidden_size": {"values": [64, 128]},
            "seed": {
                "values": [0, 1, 2, 3, 4],
            },
        },
    }

    # Create sweep (you already fixed login, so no entity needed)
    sweep_id = wandb.sweep(sweep_config, project="crazyflie-ppo-clean")

    NUM_RUNS = 30  # make this big for overnight

    # ----- 8) Global sweep progress bar -----
    for _ in trange(NUM_RUNS, desc="Sweep runs", unit="run"):
        # Each call with count=1 runs exactly ONE trial
        wandb.agent(sweep_id, function=train, count=1)
