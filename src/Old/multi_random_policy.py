# Multi (2) drone, random policy

import os
import time
import mujoco.viewer
from CrazyFlieEnv import CrazyflieEnv


scenePath = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Assets", 
    "bitcraze_crazyflie_2", 
    "scene_multi.xml"
)

env = CrazyflieEnv(scenePath, 2)


# Training example (using random policy for now)
num_episodes = 5
max_steps = 500

for ep in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # We can replace this random sampling with a policy
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        if done:
            print(f"Episode {ep+1} ended after {step+1} steps with reward: {total_reward:.2f}")
            break
    else:
        print(f"Episode {ep+1} ran all steps ({max_steps}) with reward: {total_reward:.2f}")

env.close()


# Evaluation / Visualization - use learned policy and visualize in the sim
obs, _ = env.reset()
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    print("\nRunning final visualization based on learned policy (RANDOM FOR NOW)")
    for step in range(max_steps):
        # Replace the random sampling here with a trained policy
        action = env.action_space.sample()  

        obs, reward, done, _, _ = env.step(action)

        viewer.sync()
        time.sleep(1/60)

        if done:
            print(f"Simulation ended early at step {step+1}")
            break
    else:
        print(f"Simulation ran all {max_steps} steps\n")
