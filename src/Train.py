
from stable_baselines3 import PPO,SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import CrazyFlieEnv
import os




##function ot create CrazyFlieEnv
def make_env(xmlpath,target):
    return CrazyFlieEnv.CrazieFlieEnv(target, xmlpath)


models_dir_ppo:str = "models/PPO"
model_dir_SAC:str = "models/SAC"
log_dir:str = "logs"

xml_path = os.path.join(
    os.path.dirname(__file__),   # current file directory
    "..",                        # go up one folder (from src to project root)
    "Assets",
    "bitcraze_crazyflie_2",
    "scene.xml"
)
TARGET = 0.5
xml_path = os.path.abspath(xml_path)


##wrap the environment we made with DUmmyVecEnv for vector of environments that are being monitored
#sb3 only takes vectored environments, this allows for multiple enviornments, and we pass in a callable(a funciton to call which instantiates the enivornments)
##then normalize the vecotred environment
venv:DummyVecEnv = DummyVecEnv([lambda:Monitor(make_env(xmlpath=xml_path,target=TARGET))])
venv = VecNormalize(venv=venv,norm_obs=True,norm_reward=True)

##for the neural network create hidden layers of 64x64 shape according to docs
policy_kwargs = dict(net_arch=[64, 64])

##Stable baseline 3 PPO implementation, all these prameters are default, pass in the policy("MLP = multilayered perceptron")
model = PPO(##just default PPO parameters no strategy now
    "MlpPolicy",
    venv,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.1,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=log_dir
)

##do same for SAC algorithm but with default parameters
model2 = SAC(
    "MlpPolicy",
    venv,
    verbose=1,
    tensorboard_log=log_dir)

total_time:int = 10
TIMESTEPS=1000

#each is 1000 timesteps and then for 10 iterations, learn a model then save it for both PPO and SAC
for i in range(1,total_time):
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name="PPO")
    model2.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name="SAC")
    
    model.save(f"{models_dir_ppo}/{TIMESTEPS*i}")
    model.save(f"{model_dir_SAC}/{TIMESTEPS*i}")

##to run model type python src/Train.py
##too see tensorboard graphs run tensorboard logdir='The File path of the logs folder'







##evalaution function portion now
# eval_env = DummyVecEnv([lambda: Monitor(make_env(target=0.5,xmlpath=xml_path))])
# eval_env = VecNormalize.load("vecnormalize.pkl", eval_env)
# eval_env.training = False
# eval_env.norm_reward = False

# obs = eval_env.reset()
# done, truncated = False, False
# while not (done or truncated):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = eval_env.step(action)


# print(gym.registry.keys)##just to test 


# vec_env = make_vec_env("", n_envs=4)




    

