# using os for file pathing and file loading
##using tuple,Dict,and deque as datastructures
##nump is also used to create arrays and perform math operations, we'll use these arrays to hold data like observations and actions
##gymnasium is used to implement the gym.Env interface and also gives us access to the Spaces data structures for actiona and observation space
### mujoco imported for our physics engine, it is what we are applying the environment and actions too
import os
from typing import Optional, Tuple, Dict, Any
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco as mj

##this class right here we define our custome RL environment by first inheriting from the gym itnerface gym.Env
## which demands the implementaiton of __init__, step(action), reset functions
class CrazyFlieEnv(gym.Env):
    """
    Thrust-only Crazyflie hover task (MuJoCo), with anti-overshoot shaping:
      • Action: scalar thrust u in [tmin, tmax]
      • Autodetects actuators:
          - 4 per-motor: sets all motors = u
          - [thrust, mx, my, mz]: sets thrust=u, moments=0
      • Smoothing: slew-rate + low-pass on thrust
      • Reward:
          - closeness to target (bounded)
          - **directional progress**: (target_z - z) * vz  -> only rewards moving TOWARD target
          - strong penalty for being/staying ABOVE target
          - vertical-speed penalty near target
          - gentle tilt/lateral + thrust-jump penalties
      • Safety:
          - soft ceiling above target
          - hard ceiling termination
      • Early success after smooth hover for K steps
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}## the metadata here is just for rendering info 

    def __init__(
        self,
        xml_path: str,
        target_z: float = 0.5,
        max_steps: int = 1500,
        ##render options ofr camera, render mode, as well as width ahd height of camera as well as the name and positions of the camera
        ##remove these???
        render_mode: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        camera_name: Optional[str] = None,
        follow_body: Optional[str] = None,
        cam_distance: float = 1.4,
        cam_elevation: float = -20.0,
        cam_azimuth: float = 110.0,
        # Smoothing params, alpha and slew, it's how we soften/limit the thrust at each step
        thrust_lowpass_alpha: float = 0.25,##
        thrust_slew_per_step: float = 0.02,##max change per step
        # Hover success parameters
        hover_band: float = 0.04,             # ±4 cm band
        hover_required_steps: int = 600,       # 60 hz so 1 second to fullfill a steady hover(can increase this)
        smooth_window: int = 600,
        # Anti-overshoot guards defining a soft ceiling for the drone to not go over(small penalty) and a hard ceiling where the episode terminates and big penalty
        soft_ceiling_margin: float = 0.20,    # start penalizing > target+0.20 m
        hard_ceiling_margin: float = 0.40,    # terminate if > target+0.40 m
        print_actuators: bool = False,##remove?
    ):
        
        """ Initilaization function to first create the CrazyFlyEnv
        
        arguments:
        self -- the instance of the environemnt, self is pased in for every class 
        xml_path: string -- the file path for the crazyfly model, where the model lives so we can open it and access the physics properties
        target_z: float -- the height in meters we want to hover at
        max_steps:int -- the maximum steps (iterations of performng acitons) for a single episode. One episode consists of many steps
        thrust_lowpass_alpha:float -- value from 0-1, represents the alpha value for the lowpass filter formula used to filter thrust with
        thrust_slew_per_step: float -- represents the maximum change allowed in the thrust value per step to prevent going from 0.1-0.3 in one go
        hover_band:float -- the band defining how close the drone needs to be to it's target value to be counted as a successful hover( +-hover_band)
        hover_required_steps:int -- the number of steps required for the drone to be within the hover band to count as a success
        smooth_window : int -- number of steps for how long the creation of the history will be(60 is 1 second)
        Return: return_description
        """
        
        super().__init__()## call the base class initialization
        if not os.path.exists(xml_path):## check if the model exists and if not raise an exception
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        
        ##creating the model from the xml path for the model then taking the model and binding the data to two properties of the class model, and data(initialized as self.model and self.data)
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        ##intialize env properties for the target height and max steps
        self.target_z = float(target_z)
        self.max_steps = int(max_steps)

    #     # --- Actuator autodetect ---
    #     ##remove maybe?
    #     self.nu = int(self.model.nu)##self.model.nu defines the number of actuators
    #     if self.nu < 4:
    #         raise ValueError("Model must expose ≥ 4 actuators (motors or virtual controls).")
    #     self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
    #     self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

    #     rng4 = np.c_[self.ctrl_low[:4], self.ctrl_high[:4]]
    #     per_motor_like = np.allclose(rng4, rng4[0])
    #  #----------------------------
        self.mode = "thrust_plus_moments"## in the mode where the control is the thrust + x,y,z moments
        tmin, tmax = 0.0, 0.35 ##define thrust minimum and maximum values according to mj model (0,0.35)

        # give headroom but not ballistic
        tmax = min(tmax, 0.30)##can potentially remove, this just caps the thrust
        
        ##defining the action space as the 4 actuators of the drone thrust, yaw(z-axis rotate in place), pitch(y-axis forward/back), roll(x-axis left/right)
        ##low and high values taken from the cf2.xml actuator ctrl ranges
        ##here we specify the action space using spaces.Box to illustrate a continous range with a range of
        #low values representing each actuator [0,-1,-1,-1] = [thrust,roll,pitch,yaw]
        ##high values are for each actuator too [0.35,1,1,1] = [thrust,roll,pitch,yaw]
        ##we will be limiting aciton space to only thrust so the drone only understands vertical movement, and no rotaitonal tilting, so it'sa  1D array
        self.action_space = spaces.Box(
            low=np.array([tmin], dtype=np.float32),
            high=np.array([tmax], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space, define an array of 13 dimensions representing our observations and have them be infinity since any of the observations don't have a cap

        hi = np.inf * np.ones(13, dtype=np.float32)
        # observation: 13D [pos3, quat4, linv3, angv3]
        # [x,y,z,  qw,qx,qy,qz,  vx,vy,vz,  wx,wy,wz]
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)

        # Hover prior
        ##choosing a starting hover thrust that makes training easier, this hover thrust is according to the weight of the drone
        ##helps start a stable episode, this thrust should roughly balance gravity
        self.HOVER_THRUST = float(np.clip(0.27, tmin, tmax))

        ##Smoothing
        ##alpha and max_du are properties we have for the slew rate and lowpass filter
        self.alpha = float(thrust_lowpass_alpha)
        self.max_du = float(thrust_slew_per_step)
        ##u_cmd is our filtered thrust which we will pass into the data control rather than the raw action
        self.u_cmd = self.HOVER_THRUST
        ##last_du is the thrust of the last episode
        
        self.last_du = 0.0

        # tracking for successful hover
        self.band = float(hover_band)
        self.hover_required = int(hover_required_steps)
        self.hover_count = 0 ##count for hover time

        #  track thrust jerk and vertical speed across the smooth_window(60 steps = 1s)to detect smooth hover with a deque datastructure
        self.du_hist = deque(maxlen=int(smooth_window))
        self.vz_hist = deque(maxlen=int(smooth_window))
        self.frame_skip = 10
        # Ceiling values
        self.soft_ceiling = self.target_z + float(soft_ceiling_margin)
        self.hard_ceiling = self.target_z + float(hard_ceiling_margin)

        # # Rendering --- remove maube?
        # self.render_mode = render_mode
        # self._width, self._height = int(width), int(height)
        # self._camera_name = camera_name
        # self._follow_body_name = follow_body
        # self._follow_bid = None
        # self._cam_distance = float(cam_distance)
        # self._cam_elevation = float(cam_elevation)
        # self._cam_azimuth = float(cam_azimuth)
        # self._renderer: Optional[mj.Renderer] = None
        # if self.render_mode == "rgb_array":
        #     self._renderer = mj.Renderer(self.model, height=self._height, width=self._width)
        # if self._follow_body_name is not None:
        #     bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self._follow_body_name)
        #     if bid < 0:
        #         raise ValueError(f"follow_body='{self._follow_body_name}' not found")
        #     self._follow_bid = int(bid)
        # else:
        #     self._follow_bid = 1 if self.model.nbody > 1 else None

        self.step_idx = 0

        # if print_actuators:
        #     names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.nu)]
        #     print(f"[CFThrustHoverEnv] mode={self.mode}, nu={self.nu}")
        #     print(f"[CFThrustHoverEnv] actuator names: {names}")
        #     print(f"[CFThrustHoverEnv] ctrl ranges:\n{np.c_[self.ctrl_low, self.ctrl_high]}")

    
        ##reset function that is responsible for the start of every episode to reset values and returns the inital observaiton and info if needed
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        ##reset the mujocco data
        mj.mj_resetData(self.model, self.data)

        ##mujocco data has 3 parts
        ##data.qpos = (x,y,z)+(w,x,y,z) position and quaternion pos
        ##data.qpvel = (vx,vy,vz)+(wx,wy,wz) = linear velocity and angular velocity
        ##reset each to base values so 0 for all of them
        ##quaternion position means drone is straight up with no rotation represented by the identity matrix (1,0,0,0)
        #
        self.data.qpos[:] = np.array([0, 0, 0.15, 1, 0, 0, 0], dtype=np.float64)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0#Reset thrust to 0
        self.u_cmd = self.HOVER_THRUST##reset the filtered thrust back to base hover_thrust
        self.last_du = 0.0##reset the last thrust to 0
        ##reset the hover counter back to 0 for the start of a new episode
        self.hover_count = 0
     

        ##clear the history of thrust oscilations and velocity magnitudes.
        self.du_hist.clear()
        self.vz_hist.clear()
        ##reset the step index
        self.step_idx = 0
        return self._obs(), {}

    def _apply_thrust(self, u_scalar: float): ####this applies the filtering to the thrust, u_scalar is the thrust aciton
        ## in this funciton we perform acutator shaping or thrust smoothing
        ## u_cmd:current motor thrust
        ## u_scalar: current requested thrust for the drone
        ##du :limited thrust change for a single step -- (action_thrust_value - base_thrust_command ) clipped between the maximum thrust change
        du = np.clip(u_scalar - self.u_cmd, -self.max_du, self.max_du) ##calculate du which is clip(thrust_action-base_thrust, )
        
        ##u_slewed: new thrust
        ##calcualte new thrust by adding previous thrust to the new requested thrust value's clipped change
        u_slewed = self.u_cmd + du

        ##new_u is the filtered thrust
        ##smooth the new thrust with the low-pass filter = (1-alpha)*current_thrust + alpha*desired_thrust
        new_u = (1.0 - self.alpha) * self.u_cmd + self.alpha * u_slewed

        ##the magnitude of the change in thrust for the current step
        self.last_du = float(abs(new_u - self.u_cmd))##measure how much the thrust has changed and store it in last_du
        self.u_cmd = float(new_u)

        self.data.ctrl[0] = self.u_cmd

        # if self.mode == "per_motor":
        #     self.data.ctrl[:4] = self.u_cmd
        #     if self.nu > 4:
        #         self.data.ctrl[4:] = 0.0
        # else:
        #   ##apply the thrust to the mujoco control
        #     if self.nu > 1:
        #         self.data.ctrl[1:4] = 0.0
        #     if self.nu > 4:
        #         self.data.ctrl[4:] = 0.0

    def step(self, action: np.ndarray): #
        ##reward functionality
        # scalar thrust request
        ##requested thrust is clipped first to be in the action space range
        u_req = float(np.clip(np.asarray(action).squeeze(), self.action_space.low[0], self.action_space.high[0]))
        self._apply_thrust(u_req)##send thrust to be filtered and applied to the data.ctrl

        # advance the physics step by sending in our data 
        
        for _ in range(self.frame_skip):
            mj.mj_step(self.model, self.data)
        self.step_idx += 1

        # state
        obs = self._obs() ##retrieving the current state after performing the action
        ##from the 13d obs array retrieve ny relevant obs variables
        x, y, z = float(obs[0]), float(obs[1]), float(obs[2])
        qw, qx, qy, qz = obs[3:7]
        vz = float(obs[9])
        height_err = abs(z - self.target_z) ##height_err is how far we are from target height
        above = z - self.target_z ##above -- how far above the target are we?If z<target then we will be negative values above target(since above is +)

        # update smoothness trackers
        self.du_hist.append(self.last_du)##add to the histories the vertical velocity and last_du which is difference in thrust between currently applied thrust and last applied thrust
        self.vz_hist.append(abs(vz))

        # ---------- Reward ----------

        ## the final reward is calculated like reward = (closeness to target) + (moving in direction of target) - (go too fast near target) -
        ## (be above target)-(tilt/wander)-(be jerky)-(waste energy)
        # 1) Bounded closeness to target
        reward = 1.4 * (1.0 - np.tanh(6.0 * height_err))

        # 2) Directional progress (only rewards moving TOWARD target)
        #    (target - z) * vz  > 0  iff moving toward target
        toward = (self.target_z - z) * vz ##if the result is positive, that means the velocity is moving towards the target
        reward += 0.9 * np.tanh(toward / 0.25)   # scaled, bounded in [-0.9, 0.9]

        # 3) Near-target damping: penalize vertical speed inside ~two bands
        near = np.exp(- (height_err / max(self.band * 2.0, 1e-6)) ** 2)  # 0..1
        reward -= (0.15 + 0.65 * near) * abs(vz) ## here the stronger the vz is, and the closer the target(near) to the band, the larger the penalty

        # 4) Above-target penalties (anti-overshoot)
        if above > 0.0:
            # mild linear penalty as soon as we're above
            reward -= 1.2 * above
            # quadratic soft-ceiling penalty beyond soft_ceiling
            if z > self.soft_ceiling:
                reward -= 8.0 * (z - self.soft_ceiling) ** 2

        # 5) Gentle attitude/lateral penalties
        reward -= 0.4 * (abs(qx) + abs(qy))
        reward -= 0.012 * (abs(x) + abs(y))

        # 6) Smoothness: penalize thrust jumps
        reward -= 0.5 * self.last_du

        # 7) Energy bias near hover thrust
        reward -= 0.0003 * abs(self.u_cmd - self.HOVER_THRUST)
        # ---------------------------------------

        # Hard crash / NaNs
        if z < 0.0 or np.any(np.isnan(obs)) or np.any(np.isinf(obs)): ## if the drone crashes by going z below 0 then big penalty or if observation hs Nans
            reward -= 50.0
            return obs, reward, True, False, {"crash": True}

        # Hard ceiling termination
        if z > self.hard_ceiling:##if the drone hits the hard ceiling then terminate
            reward -= 50.0
            return obs, reward, True, False, {"ceiling": True}

        # Early success: in band and calm
        in_band = height_err <= self.band and abs(vz) < 0.05
        if in_band: ##if the height_err which is the distance between the drone and the target is within the band, so if the drone is and vertical velocity is less than 0.05
            ##then increment hover_count as a successful hover step
            self.hover_count += 1
        else:##if outside band, then decrement the hover counter by 1 but don't let it go belpw 0
            self.hover_count = max(0, self.hover_count - 1)

        if self.hover_count >= self.hover_required:## if we reach hover counter
            mean_vz = float(np.mean(self.vz_hist)) if len(self.vz_hist) else 999.0 ##get averges over the vertical velocity in the last smoothing_window
            mean_du = float(np.mean(self.du_hist)) if len(self.du_hist) else 999.0## get the averges of thrust change in last smoothing_indow
            if mean_vz < 0.04 and mean_du < 0.010:## so if the average  vertical velocity is that and sam as thrust chnge then that constitutes successful hover
                reward += 50.0
                return obs, reward, True, False, {
                    "success": True,
                    "hover_steps": self.hover_count,
                    "mean_vz": mean_vz,
                    "mean_du": mean_du,
                }

        terminated = False
        truncated  = self.step_idx >= self.max_steps
        return obs, reward, terminated, truncated, {"hover_steps": self.hover_count}

# ---------- Helpers ----------
        
    def _obs(self) -> np.ndarray:##returns the current state of the environment with a 13 dimensional array

        pos = self.data.qpos[0:3]
        quat = self.data.qpos[3:7]
        linv = self.data.qvel[0:3]
        angv = self.data.qvel[3:6]
        return np.concatenate([pos, quat, linv, angv]).astype(np.float32)

    # ---------- Rendering (rgb_array) ----------

    # def _apply_follow_camera(self):
    #     if self._renderer is None or self._follow_bid is None:
    #         return
    #     target = self.data.xpos[self._follow_bid].copy()
    #     cam = self._renderer.cam
    #     cam.type = mj.mjtCamera.mjCAMERA_FREE
    #     cam.lookat[:] = target
    #     cam.distance = self._cam_distance
    #     cam.elevation = self._cam_elevation
    #     cam.azimuth = self._cam_azimuth

    # def render(self) -> Optional[np.ndarray]:
    #     if self.render_mode == "rgb_array":
    #         if self._renderer is None:
    #             self._renderer = mj.Renderer(self.model, height=self._height, width=self._width)
    #         if self._camera_name is not None:
    #             self._renderer.update_scene(self.data, camera=self._camera_name)
    #         else:
    #             self._apply_follow_camera()
    #             self._renderer.update_scene(self.data)
    #         img = self._renderer.render()
    #         return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    #     elif self.render_mode == "human":
    #         return None
    #     return None

    # def close(self):
    #     if self._renderer is not None:
    #         self._renderer.close()
    #         self._renderer = None
