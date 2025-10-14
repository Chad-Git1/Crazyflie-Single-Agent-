
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
import CrazyFlieEnv
import os
from stable_baselines3.common.vec_env import SubprocVecEnv


##function ot create CrazyFlieEnv
def make_env(xmlpath, target):
    return CrazyFlieEnv.CrazyFlieEnv(target, xmlpath)

models_dir_ppo: str = "models/PPO"
model_dir_SAC: str = "models/SAC"
log_dir: str = "logs"

xml_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Assets",
    "bitcraze_crazyflie_2",
    "scene.xml"
)
TARGET = 0.5
xml_path = os.path.abspath(xml_path)


##wrap the environment we made with DUmmyVecEnv for vector of environments that are being monitored
#sb3 only takes vectored environments, this allows for multiple enviornments, and we pass in a callable(a funciton to call which instantiates the enivornments)
##then normalize the vecotred environment
N_ENVS = 4
venv = make_vec_env(lambda: Monitor(make_env(xmlpath=xml_path, target=TARGET)), n_envs=N_ENVS)
venv = VecNormalize(venv=venv, norm_obs=True, norm_reward=True)

##for the neural network create larger hidden layers for better learning
policy_kwargs_ppo = dict(net_arch=[256, 256])
policy_kwargs_sac = dict(net_arch=[400, 300])

##Stable baseline 3 PPO implementation, all these prameters are default, pass in the policy("MLP = multilayered perceptron")
model = PPO(
    "MlpPolicy",
    venv,
    learning_rate=5e-5,
    n_steps=4096,
    batch_size=128,
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    vf_coef=0.1,
    policy_kwargs=policy_kwargs_ppo,
    verbose=1,
    tensorboard_log=log_dir
)

''' 
Old SAC implementation, kept for reference

model2 = SAC(
    "MlpPolicy",
    venv,
    learning_rate=5e-5,
    batch_size=256,
    gamma=0.99,
    tau=0.02,
    ent_coef="auto",
    policy_kwargs=policy_kwargs_sac,
    verbose=1,
    tensorboard_log=log_dir
)
'''

# New SAC implementation with adjusted hyperparameters

model2 = SAC(
    "MlpPolicy",
    venv,
    learning_rate=3e-4,  # Increased learning rate for potentially faster convergence
    learning_starts=10000,  # More initial random steps for better exploration
    batch_size=256,
    gamma=0.98,  # Slightly lower discount factor to prioritize recent rewards
    tau=0.05,  # Increased tau for more stable target updates
    ent_coef="auto",    
    policy_kwargs=policy_kwargs_sac,
    verbose=1,
    tensorboard_log=log_dir,
    target_entropy = -float(venv.action_space.shape[0]) # Negative target entropy encourages sufficient exploration without excessive randomness

)

total_time: int = 10
TIMESTEPS = 2000

# Add evaluation callback to monitor and save best models
eval_env = make_vec_env(lambda: Monitor(make_env(xmlpath=xml_path, target=TARGET)), n_envs=1)
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True)
eval_callback_ppo = EvalCallback(eval_env, best_model_save_path=models_dir_ppo,
                                 log_path=log_dir, eval_freq=TIMESTEPS, deterministic=True, render=False)
eval_callback_sac = EvalCallback(eval_env, best_model_save_path=model_dir_SAC,
                                 log_path=log_dir, eval_freq=TIMESTEPS, deterministic=True, render=False)

import time
best_ppo_reward = float('-inf')
best_sac_reward = float('-inf')
no_improve_count = 0
early_stop_patience = 3
crash_threshold = -30
for i in range(1, total_time):
    print(f"=== Training iteration {i}/{total_time-1} ===")
    start_time = time.time()
    ppo_result = model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=eval_callback_ppo)
    sac_result = model2.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC", callback=eval_callback_sac)
    duration = time.time() - start_time
    print(f"Iteration {i} training time: {duration:.2f} seconds")
    # Save checkpoints
    model.save(f"{models_dir_ppo}/checkpoint_{TIMESTEPS*i}")
    model2.save(f"{model_dir_SAC}/checkpoint_{TIMESTEPS*i}")
    # Log episode rewards for diagnostics
    mean_ppo_reward = None
    mean_sac_reward = None
    if hasattr(ppo_result, 'ep_info_buffer'):
        ppo_rewards = [ep_info['r'] for ep_info in ppo_result.ep_info_buffer if 'r' in ep_info]
        if ppo_rewards:
            mean_ppo_reward = sum(ppo_rewards) / len(ppo_rewards)
            print(f"PPO mean episode reward: {mean_ppo_reward:.2f}")
            if mean_ppo_reward > best_ppo_reward:
                best_ppo_reward = mean_ppo_reward
                no_improve_count = 0
            else:
                no_improve_count += 1
            if mean_ppo_reward < crash_threshold:
                print("Warning: PPO agent may be crashing frequently. Consider tuning reward or hyperparameters.")
    if hasattr(sac_result, 'ep_info_buffer'):
        sac_rewards = [ep_info['r'] for ep_info in sac_result.ep_info_buffer if 'r' in ep_info]
        if sac_rewards:
            mean_sac_reward = sum(sac_rewards) / len(sac_rewards)
            print(f"SAC mean episode reward: {mean_sac_reward:.2f}")
            if mean_sac_reward > best_sac_reward:
                best_sac_reward = mean_sac_reward
                no_improve_count = 0
            else:
                no_improve_count += 1
            if mean_sac_reward < crash_threshold:
                print("Warning: SAC agent may be crashing frequently. Consider tuning reward or hyperparameters.")
    # Early stopping if no improvement
    if no_improve_count >= early_stop_patience:
        print(f"No improvement in mean reward for {early_stop_patience} iterations. Stopping early.")
        break


##to run model type python src/Train.py
##too see tensorboard graphs run tensorboard logdir='The File path of the logs folder'


## Save the VecNormalize statistics for evaluation or deployment
# vecnorm_dir = "logs/VectorNormalization"
# os.makedirs(vecnorm_dir, exist_ok=True)  # Create directory if missing
# venv.save(os.path.join(vecnorm_dir, "normalization.pkl"))


##evalaution function portion now
# eval_env = DummyVecEnv([lambda: Monitor(make_env(target=0.5,xmlpath=xml_path))])
# eval_env = VecNormalize.load("logs/vecnormalize.pkl", eval_env)
# eval_env.training = False
# eval_env.norm_reward = False

# obs = eval_env.reset()
# done, truncated = False, False
# while not (done or truncated):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = eval_env.step(action)


# print(gym.registry.keys)##just to test 


# vec_env = make_vec_env("", n_envs=4)




    

