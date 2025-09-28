import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco as mj

import os

##creates an environment inhereting from gymnasium's gym.Env class
class CrazieFlieEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self,target, xmlPath):##initialization takes in a xml path for the crazy flies model and then binds it to a model pa
        #parameter and also a data parameter
        super().__init__
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
           
    def reset(self,seed = None, options = None):

        super().reset(seed=seed)

        ##reset model data, then reset the control data so that there is no thrust, or any rotaitonal movement
        mj.mj_resetData(self.model, self.data)

        self.data.ctrl[:] = 0
        self.data.qpos[0:]=np.array([0,0,0.1,1,0,0,0],dtype=np.float32)
    

        self.data.qvel[0:]=0

        current_obs = self._get_obs()
        current_info = {}

        return current_obs, current_info
        
    def step(self,action):## the agent applies actions to the environemnt through this method
       
       ##clip the action then apply the action, check the current status of the environment, then determine reward based on that
       action= np.clip(action,self.action_space.low,self.action_space.high)

       self.data.ctrl[:] = action

       mj.mj_step(self.model,self.data)

       current_observation = self._get_obs()

       distance = self.target_height-current_observation[2]
       reward = -abs(distance)
       done=False
       truncated=False
       info = {}
       ##when drone crashes
       if(current_observation[2] < 0.01 ):
           done=True
           reward =-100
           print("dorne crashed")

        ##when drone achieves hover distance
       elif(distance==0):
           done=True
           print("drone achieved")
       return current_observation,reward,done,truncated,info
       



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
    env = CrazieFlieEnv(0.5,xml_path)
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
  
        


        
   
    
    


