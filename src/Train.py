
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import CrazyFlieEnv
import os



##function ot create CrazyFlieEnv
def make_env(xmlpath,target):
    return CrazyFlieEnv.CrazieFlieEnv(target, xmlpath)



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
)

model.learn(total_timesteps=10000)

model.save("ppo_crazyflie")
venv.save("vecnormalize.pkl")



##evalaution function portion now
eval_env = DummyVecEnv([lambda: Monitor(make_env(target=0.5,xmlpath=xml_path))])
eval_env = VecNormalize.load("vecnormalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

obs = eval_env.reset()
done, truncated = False, False
while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)


print(gym.registry.keys)##just to test 


# vec_env = make_vec_env("", n_envs=4)




    

