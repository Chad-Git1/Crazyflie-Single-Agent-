
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


# Build a tiny vec env only to load VecNormalize stats
def _make_norm_loader(xml_path: str, target_z: float, max_steps: int):##factor function to load the environemnt
    def _thunk():
        return Monitor(CrazyFlieEnv(xml_path=xml_path, target_z=target_z, max_steps=max_steps,hover_required_steps=600))
    return DummyVecEnv([_thunk])


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    xml_path = os.path.abspath(os.path.join(here, "..", "Assets", "bitcraze_crazyflie_2", "scene.xml"))
    ##path to models and the specific model zip file
    models_dir = os.path.abspath(os.path.join(here, "..", "models", "ComplexMain_DR"))
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
        # Evaluation: no extra noise (or use milder values)
        obs_noise_std=0.05,
        obs_bias_std=0.05,
        action_noise_std=0.05,
        motor_scale_std=0.05,  # ±2% thrust gain
        frame_skip=10,
        frame_skip_jitter=2,  
    )
## we don't wrap it with dummyVecEnv because mujocco only works with a single env
    obs_raw, _ = env.reset()
    dt_sim = env.model.opt.timestep
    dt_step = dt_sim * env.frame_skip 

    #  Launch MuJoCo's interactive viewer; step the env and sync the window
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        terminated = False
        truncated = False
        ep_return = 0.0

       
      
        # simple 1 Hz HUD print
        t0 = time.time()
        last_print = t0
        while not (terminated or truncated):
            
            obs_norm = vecnorm.normalize_obs(obs_raw[None, :])  # shape (1, obs_dim)

            # Policy prediction (deterministic for evaluation)
            action, _ = model.predict(obs_norm, deterministic=True)

            # Make sure it's a NumPy array
            action = np.asarray(action, dtype=np.float32)
            if action.ndim == 2:
                a = action[0]     # (4,)
            else:
                a = action        # (4,)

            thrust = float(a[0])
            mx = float(a[1])
            my = float(a[2])
            mz = float(a[3])

            # Step the *real* env with this action (un-normalized env)
            obs_raw, reward, terminated, truncated, info = env.step(a)
            ep_return += float(reward)

            # Logging values from *raw* obs
            x, y, z = float(obs_raw[0]), float(obs_raw[1]), float(obs_raw[2])
            vx, vy, vz = float(obs_raw[7]), float(obs_raw[8]), float(obs_raw[9])


            # Update viewer
            viewer.sync()

            # HUD at ~1 Hz
            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                R = float(ep_return)
                print(
                    f"t={int(now - t0):2d}s | "
                    f"z={z:+.3f} m  vz={vz:+.3f} m/s  "
                    f"thrust={env.last_du:.3f}, "
                    f"(x,y,z)=({x:+.3f},{y:+.3f},{z:+.3f})  "
                    f"R={R:.1f}"
                    f"info={info}"
                )

            # Match real time: 1 env.step = frame_skip * dt_sim seconds
            time.sleep(dt_step)

    print(f"\nEpisode finished. Return={ep_return:.2f}, terminated={terminated}, truncated={truncated}, Info: {info}")