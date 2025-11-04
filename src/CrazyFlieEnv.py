import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco as mj

import os

##creates an environment inheriting from gymnasium's gym.Env class
class CrazyFlieEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self, target, xmlPath):
        super().__init__()
        self.model = mj.MjModel.from_xml_path(xmlPath)
        self.data = mj.MjData(self.model)
        self.target_height = target

        ##defining the action space as the 4 actuators of the drone thrust, yaw(z-axis rotate in place), pitch(y-axis forward/back), roll(x-axis left/right)
        ##low and high values taken from the cf2.xml actuator ctrl ranges
        ##here we specify the action space using spaces.Box to illustrate a continous range with a range of
        #low values representing each actuator [0,-1,-1,-1] = [thrust,roll,pitch,yaw]
        ##high values are for each actuator too [0.35,1,1,1] = [thrust,roll,pitch,yaw]
        self.action_space :spaces.Box = spaces.Box(low=np.array([0,-1,-1,-1],dtype=np.float32),high=np.array([0.35,1,1,1], dtype =np.float32),dtype=np.float32)
        
        #here we create the observaiton space which is the locaitonal and speed values of which there are 13
        #these valeus have no limits and can range from -infinity to infinity so we have to have an array 
        #representing all 13 of these elements as a way to represent the state where each value has a range from (inf,-inf)
        obs = np.inf*np.ones(13,dtype=np.float32)
        self.observation_space:spaces.Box = spaces.Box(low=-obs ,high=obs,dtype=np.float32)
        self.state:np.array = np.array([0,0,0,0.01])
           
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset model data, then reset the control data so that there is no thrust, or any rotational movement
        mj.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0
        # Start at a safer height above ground
        self.data.qpos[0:] = np.array([0, 0, 0.3, 1, 0, 0, 0], dtype=np.float32)
        self.data.qvel[0:] = 0
        current_obs = self._get_obs()
        current_info = {}
        return current_obs, current_info
        
    def step(self, action):
        # Clip the action and apply it
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mj.mj_step(self.model, self.data)
        current_observation = self._get_obs()
        # Reward shaping
        distance = abs(self.target_height - current_observation[2])
        velocity = abs(current_observation[5])  # z-velocity
        reward = -distance - 0.02 * velocity
        # Small reward for staying above ground
        if current_observation[2] > 0.01:
            reward += 1.0
        if current_observation[2] < 0.01:
            reward -= 10  # further reduced crash penalty
            done = True
            print("drone crashed")
        elif distance < 0.05:
            reward += 10  # further increased bonus for hovering close to target
            done = False
        else:
            done = False
        truncated = False
        info = {}
        return current_observation, reward, done, truncated, info
       



    def _get_obs(self):
        ##here our observation for the environment consist of position(x,y,z), quaternion position(w,x,y,z),
        ##linear velocity (vx,vy,vz), angular velocity (wx,wy,wz)
        ##linear velocity is the velocity in a particular direction and angular velocity is rotational speed along an axis
        pos = self.data.qpos[0:3]
        quat_pos = self.data.qpos[3:7]
        linear_velocity= self.data.qvel[0:3]
        angular_velocity=self.data.qvel[3:6]
        ##pos (x,y,z) then quat pos(w,x,y,z) then lin velocity(vx,vy,vz) then angular velocity (wx,wy,wz)
        observation = np.concatenate([pos,quat_pos,linear_velocity,angular_velocity])
        return observation.astype(np.float32)
    
   



if __name__ == "__main__":##this part is for testing purely
        
    xml_path = os.path.join(
        os.path.dirname(__file__),   # current file directory
        "..",                        # go up one folder (from src to project root)
        "Assets",
        "bitcraze_crazyflie_2",
        "scene.xml"
    )
    xml_path = os.path.abspath(xml_path)
    env = CrazyFlieEnv(0.5,xml_path)
    obs, info= env.reset()
    print(obs)
    low = env.action_space.low
    high = env.action_space.high
  
    action = env.action_space.sample()
    observation, reward, done,truncated,info = env.step(action)
    print(f"action {action}")
    print(f"observation: {observation}")
    total_reward = reward
    while not done and not truncated:
        action = env.action_space.sample()
        observation, reward, done,truncated,info = env.step(action)
        
        total_reward += reward
    print(f"final observation {observation} with total reward: {total_reward}")
  
        


        
   
    
    


