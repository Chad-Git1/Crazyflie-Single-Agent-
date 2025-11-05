
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


from CrazyFlieEnvMain import CrazyFlieEnv


# Build a tiny vec env only to load VecNormalize stats
def _make_norm_loader(xml_path: str, target_z: float, max_steps: int):##factor function to load the environemnt
    def _thunk():
        return Monitor(CrazyFlieEnv(xml_path=xml_path, target_z=target_z, max_steps=max_steps))
    return DummyVecEnv([_thunk])


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    xml_path = os.path.abspath(os.path.join(here, "..", "Assets", "bitcraze_crazyflie_2", "scene.xml"))
    ##path to models and the specific model zip file
    models_dir = os.path.abspath(os.path.join(here, "..", "models", "Test"))
    model_path = os.path.join(models_dir, "test.zip")
    norm_path  = os.path.join(models_dir, "vecnormalize.pkl")

    TARGET_Z  = 0.5
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
        max_steps=MAX_STEPS
      
        # render_mode=None because we'll use the interactive viewer below
    )## we don't wrap it with dummyVecEnv because mujocco only works with a single env
    obs, _ = env.reset()
    dt_sim = env.model.opt.timestep

    #  Launch MuJoCo's interactive viewer; step the env and sync the window
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        terminated = False
        truncated = False
        ep_return = 0.0
      
        # simple 1 Hz HUD print
        t0 = time.time()
        last_print = t0

        while not (terminated or truncated):
            # normalize observation like during training (batch dimension required)
           
            obs_n = vecnorm.normalize_obs(obs[None, :])

            # policy prediction (deterministic for evaluation)
            action, _ = model.predict(obs_n, deterministic=True)
            # action can be shape (1,) or (1,1); reduce to scalar
            u = float(np.asarray(action).squeeze())

            # step env with scalar thrust
            obs, reward, terminated, truncated, info = env.step(u)
            print(obs,info)
            ep_return += reward

            # update viewer
            viewer.sync()

            # Print a tiny HUD each second
            now = time.time()
           
            if now - last_print >= 1.0:
                z = float(obs[2])
                vz = float(obs[9])
                print(f"t={int(now - t0):2d}s | z={z:+.3f} m  vz={vz:+.3f} m/s  thrust={u:.3f}  R={ep_return:.1f}")
                last_print = now
            time.sleep(dt_sim*10)
            

        print(f"\nEpisode finished. Return={ep_return:.2f}, terminated={terminated}, truncated={truncated}")
