# src/train_thrust_ppo.py
import os
from stable_baselines3 import PPO ## using PPO from SB3
from stable_baselines3.common.monitor import Monitor## SB3 wrapped that tracks statistics like reward and successe/failure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize 
##----dummyVecEnv is a wrapped for a single gym.Env implementation
##that converts it to a vectorized environemnt
##since SB3 expects vectorized enviornments
## VecNormalize normalizes the environment

from CrazyFlieEnvComplex import CrazyFlieEnv

##factory function, a function that returns another function
def make_env(xml_path: str, target_z: float, max_steps: int = 1500, rank: int = 0):
    def _f():
        env = CrazyFlieEnv(
            xml_path=xml_path,
            target_z=target_z,
            max_steps=max_steps,
            n_stack=4,
            hover_required_steps=600,
            # <-- add domain randomization params here
            **DR_PARAMS,
        )
        env = Monitor(env)
        env.reset(seed=rank)
        return env
    return _f



if __name__ == "__main__":
    # Domain randomization settings used for TRAINING
    DR_PARAMS = dict(
        obs_noise_std=0.5,
        obs_bias_std=0.5,
        action_noise_std=0.0,
        motor_scale_std=0.0,  # ±5% thrust gain
        frame_skip=10,
        frame_skip_jitter=0,   # frame skip in [8, 12]
          start_xy_range=0.6,
        # vertical: between 0.15 m and 1.1 m (assuming TARGET_Z = 1.0)
        start_z_min=0.01,
        start_z_max=1.10,
    )
    here = os.path.dirname(__file__)

    # Point to your MuJoCo scene.xml (adjust path to your repo layout)
    ##xml path for the crazyflydrone
    xml_path = os.path.abspath(os.path.join(here, "..", "Assets", "bitcraze_crazyflie_2", "scene.xml"))
    ##directory to save the models from training
    models_dir = os.path.abspath(os.path.join(here, "..", "models", "ComplexRandomized"))
    ##where to save the logging for tensorboard
    logs_dir = os.path.abspath(os.path.join(here, "..", "logsComplexRandomized"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    ##environment paramater constants
    TARGET_Z = 1
    MAX_STEPS = 1500
    N_ENVS = 24  # or 4, 16, etc.
    SEEDS = [0]

    
    
  
    env_fns = [make_env(xml_path, TARGET_Z, MAX_STEPS, rank=i) for i in range(N_ENVS)]
    ##venv is vectorized environment, dummyVec expects an array of factory functions for a single envrionemnt to vectorize the environment
    venv = DummyVecEnv(env_fns=env_fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)##scales the venv to normalized values for rewards and observations

    model = PPO(
        "MlpPolicy", ##multilayered perceptron(which is default in sb3)
        venv, ##pass in our vectorized environment, PPO expects this
        learning_rate=3e-4, ##how fast weights update during gradient descent(smaller = slower but more stable)
        n_steps=2048, ## how many steps of experience to collect before each training update. Runs the environment for n_steps and then stores all the info(obs,action,reward) and do gradient updates (large leads to mroe stable gradients but more memory use)
        batch_size=64, ##minibatch size for each epoch, splits the large n_steps batch into mini-batches and traisn the neural network on that
        n_epochs=10,##the training doesn't scan the batch once, it goes over multiple times to train data, each iteration is an epoch. too many can lead to overfitting(10 is a good value)
        gamma=0.99,##discount factor, how much agent values future rewards over immediete
        gae_lambda=0.95,##generalized advantage estimation(GAE) it reduces noise when estimating how good an action was. 
        clip_range=0.2, ##PPO clips how much the new policy can change from the old one, it prevents instability and huge policy shifts
        ent_coef=0.001, ###entropy coefficient, entropy is the randomness which encourages exploraiton in the loss funciton, higher means more exploraiton, lower means exploit what is already known
        vf_coef=0.5,##how much value funciton loss(critic) contributes in compairson wiht policy loss and entropy bonus
        policy_kwargs=dict(net_arch=[64, 64]),##architecture for neural network, two hidden layers of 64 neurons,
        tensorboard_log=logs_dir,##logs tensorboard log files into the logs director
        verbose=1##prints training progress into the console(1 is minimal)
        
        
    )
    
    ##train the model by running it for a total_timsetep amount of simulation steps, each step is a single env.step() call
    ##total_timsteps/episode_steps = (100,000)/1500=66 episodesroughly
    model.learn(total_timesteps=2000000, progress_bar=True)

    model.save(os.path.join(models_dir, "complex.zip"))##saved the train model
    venv.save(os.path.join(models_dir, "vecnormalize.pkl"))##also saved normalization stats
    print(f"Saved model + VecNormalize into {models_dir}")


## the models folder will contain the network weights, hyperparameters and policy config.


##Achieve hover for long time first
##Either 1. Increase action space complexity
## or 2. Introduce landing task
##do parallel environemnts 