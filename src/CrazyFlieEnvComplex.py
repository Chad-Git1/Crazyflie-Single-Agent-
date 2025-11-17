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
        n_stack: int = 4,##frame stack, we create a stack of observations and n_stack is the number of observations in the stack
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
        ##these are noise paramters to simulate domain randomization
        obs_noise_std: float = 0.0,       # per-step Gaussian noise on obs
        obs_bias_std: float = 0.0,        # per-episode constant bias on obs
        action_noise_std: float = 0.0,    # Gaussian noise on actions
        motor_scale_std: float = 0.0,     # per-episode thrust gain error
        frame_skip: int = 10,             # base frame skip (was 10)
        frame_skip_jitter: int = 0,       # +/- jitter in frame skip per episode
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

        ######Frame Stack#######
        self.n_stack = int(n_stack)   ## how many past frames to stack, frame stacking is for passing an array of observations
        self.obs_dim_single = 13 ##defining a single framed observation as 13 dimensions
        #### Domain randomizations
        self.obs_noise_std = float(obs_noise_std)
        self.obs_bias_std = float(obs_bias_std)
        self.action_noise_std = float(action_noise_std)
        self.motor_scale_std = float(motor_scale_std)

        self.frame_skip_base = int(frame_skip)
        self.frame_skip_jitter = int(frame_skip_jitter)
        self.frame_skip = self.frame_skip_base  # will be randomized in reset()

        # Per-episode random variables (filled in reset)
        self.obs_bias = np.zeros(self.obs_dim_single, dtype=np.float32)
        self.motor_scale = 1.0


        #remove maybe? 
        self.mode = "thrust_plus_moments"## in the mode where the control is the thrust + x,y,z moments


        tmin, tmax = 0.0, 0.35 ##define thrust minimum and maximum values according to mj model (0,0.35)

        
        ##defining the action space as the 4 actuators of the drone thrust, yaw(z-axis rotate in place), pitch(y-axis forward/back), roll(x-axis left/right)
        ##low and high values taken from the cf2.xml actuator ctrl ranges
        ##here we specify the action space using spaces.Box to illustrate a continous range with a range of
        #low values representing each actuator [0,-1,-1,-1] = [thrust,roll,pitch,yaw]
        ##high values are for each actuator too [0.35,1,1,1] = [thrust,roll,pitch,yaw]
        mmax =1
        mmin = -mmax

        # Action: [thrust, mx, my, mz]
        self.action_space = spaces.Box(
            low=np.array([tmin, mmin, mmin, mmin], dtype=np.float32),
            high=np.array([tmax, mmax, mmax, mmax], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space, define an array of 13 dimensions representing our observations and have them be infinity since any of the observations don't have a cap
        
        ##create array of infinities that is of dimension (13*frame_stack_size) 
        hi = np.inf * np.ones(self.obs_dim_single * self.n_stack, dtype=np.float32)
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)
        # observation: 13D [pos3, quat4, linv3, angv3]
        # [x,y,z,  qw,qx,qy,qz,  vx,vy,vz,  wx,wy,wz]

        self.obs_stack = deque(maxlen=self.n_stack)## NEW create datastructure to hold the observations with length of the n_stack
    

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
        ##last_du is the thrust of the last episode, we use this in the reward funciton to discourage jerky changes
        
        self.last_du = 0.0
        ##storing the last moment values(x,y,z) rotations, as array of 3 zeros(0,0,0) since there is no moments yet
        self.last_moments = np.zeros(3,dtype=np.float32)
        self.last_dm = 0


        ## tracking for successful hover
        self.band = float(hover_band)
        self.hover_required = int(hover_required_steps)
        self.hover_count = 0 ##count for hover time

        #  track thrust jerk and vertical speed across the smooth_window(60 steps = 1s)to detect smooth hover with a deque datastructure
        self.du_hist = deque(maxlen=int(smooth_window))##tracking thrust
        self.vz_hist = deque(maxlen=int(smooth_window))##tracking vertical velocity
       
        # Ceiling values
        self.soft_ceiling = self.target_z + float(soft_ceiling_margin)
        self.hard_ceiling = self.target_z + float(hard_ceiling_margin)

        self.step_idx = 0##step counter until reach max_steps

      
        # --- Ground-stall detection ---
        ##ground stall detection is when the drone is at ground level, we use this in reward function to penalize stalling on ground
        self.ground_z_threshold = 0.10    # 10 cm; this is the threshold to detect ground stalling
        self.max_ground_steps = 50        # how many steps allowed near ground before we hit it with a penalty
        self.ground_steps = 0
        self.prev_dz = 0.0



    def _apply_obs_noise(self, single: np.ndarray) -> np.ndarray:
            """
            Take a clean single-frame observation and add:
            - per-episode bias (obs_bias)
            - per-step Gaussian noise
            """
            noisy = single.astype(np.float32).copy()

            # Bias (same every step this episode)
            if self.obs_bias_std > 0.0:
                noisy += self.obs_bias

            # Per-step Gaussian noise
            if self.obs_noise_std > 0.0:
                rng = getattr(self, "np_random", np.random)
                noisy += rng.normal(
                    loc=0.0,
                    scale=self.obs_noise_std,
                    size=single.shape,
                ).astype(np.float32)

            return noisy

    def _sample_episode_randomization(self):
            """
            Sample all per-episode random factors:
            - observation bias
            - motor thrust gain
            - frame_skip jitter
            """
            rng = getattr(self, "np_random", np.random)

            # Observation bias (e.g., altimeter offset)
            if self.obs_bias_std > 0.0:
                self.obs_bias = rng.normal(
                    loc=0.0,
                    scale=self.obs_bias_std,
                    size=self.obs_dim_single,
                ).astype(np.float32)
            else:
                self.obs_bias[:] = 0.0

            # Motor thrust gain error (motors slightly stronger/weaker)
            if self.motor_scale_std > 0.0:
                self.motor_scale = float(1.0 + rng.normal(0.0, self.motor_scale_std))
            else:
                self.motor_scale = 1.0

            # Frame skip jitter (control frequency variation)
            if self.frame_skip_jitter > 0:
                # inclusive low, exclusive high – like randint
                jitter = int(rng.integers(
                    -self.frame_skip_jitter,
                    self.frame_skip_jitter + 1
                ))
                self.frame_skip = max(1, self.frame_skip_base + jitter)
            else:
                self.frame_skip = self.frame_skip_base


    
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
        self.data.qpos[:] = np.array([0, 0, 0.01, 1, 0, 0, 0], dtype=np.float64)##reset qpos to 0,0,0.01 as starting pos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0#Reset thrust to 0
        self.u_cmd = self.HOVER_THRUST##reset the filtered thrust back to base hover_thrust
        self.last_du = 0.0##reset the last thrust to 0
        ##reset the hover counter back to 0 for the start of a new episode
        self.hover_count = 0
     

        self.du_hist.clear()##clear history of thrusts
        self.vz_hist.clear()##clear history of vertical velocity
        self.step_idx = 0
        self.ground_steps = 0

        # --- sample per-episode randomization (bias, motor scale, frame_skip) ---
        self._sample_episode_randomization()

        # --- frame stack reset using NOISY obs (what agent sees) ---
        self.obs_stack.clear()
        single_clean = self._get_single_obs()
        single = self._apply_obs_noise(single_clean)

        for _ in range(self.n_stack):
            self.obs_stack.append(single.copy())

        if self.n_stack == 1:
            obs = single
        else:
            obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)

        return obs, {}


    def _apply_thrust(self, u_scalar: float,m_vec:np.ndarray): ####this applies the filtering to the thrust, u_scalar is the thrust aciton
        ## in this funciton we perform acutator shaping or thrust smoothing and moments
        ## u_cmd:current motor thrust
        ## u_scalar: current requested thrust for the drone
        ##du :limited thrust change for a single step -- (action_thrust_value - base_thrust_command ) clipped between the maximum thrust change
        du = np.clip(
            u_scalar - self.u_cmd,
            -self.max_du,
            self.max_du
        )
        
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

        # ---- Moments: clip & apply ----
        m_vec = np.asarray(m_vec, dtype=np.float32)

        # use the full physical actuator range from MuJoCo
        m_low = -1
        m_high = 1
        m_clipped = np.clip(m_vec, m_low, m_high)

        # apply to actuators 1..3 (assuming 0=thrust,1=mx,2=my,3=mz)
        if self.model.nu >= 4:
            self.data.ctrl[1:4] = m_clipped
            if self.model.nu > 4:
                self.data.ctrl[4:] = 0.0

        # #difference from previous moments
        dm = m_clipped - self.last_moments
        # magnitude of the change for smoothness penalty
        self.last_dm = float(np.linalg.norm(dm, ord=2))
        # store last applied moments for next step
        self.last_moments = m_clipped.copy()

    def step(self, action: np.ndarray): #
        ##reward functionality
        # scalar thrust request
        ##requested thrust is clipped first to be in the action space range
        # u_req = float(np.clip(np.asarray(action).squeeze(), self.action_space.low[0], self.action_space.high[0]))
        # Convert to array and sanity-check shape
        a = np.asarray(action, dtype=np.float32).squeeze()
        if a.shape == ():
            raise ValueError("Action must be 4D: [thrust, mx, my, mz]")

        # --- action randomization / noise ---
        if self.action_noise_std > 0.0:
            rng = getattr(self, "np_random", np.random)
            a = a + rng.normal(0.0, self.action_noise_std, size=a.shape).astype(np.float32)

        # Clip to valid action space
        a_clipped = np.clip(a, self.action_space.low, self.action_space.high)

        # Split into thrust + moments
        u_req = float(a_clipped[0])
        m_req = a_clipped[1:4]

        # Apply per-episode motor scaling on thrust
        u_req = float(u_req * self.motor_scale)
        # Re-clip thrust after scaling
        u_req = float(np.clip(u_req, self.action_space.low[0], self.action_space.high[0]))

        # Send to actuator shaping (low-pass + slew) and MuJoCo
        self._apply_thrust(u_req, m_req)

        # advance the physics step
        for _ in range(self.frame_skip):
            mj.mj_step(self.model, self.data)
        self.step_idx += 1
      
         # Clean single observation from MuJoCo
        single_clean = self._get_single_obs()

        # Noisy obs for the agent
        single = self._apply_obs_noise(single_clean)

        # Frame stack update with noisy obs
        if self.n_stack == 1:
            obs = single
        else:
            if len(self.obs_stack) == 0:
                # If something cleared it, repopulate
                for _ in range(self.n_stack):
                    self.obs_stack.append(single.copy())
            else:
                self.obs_stack.append(single.copy())
            obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)


      
      
        ##from the 13d obs array retrieve any relevant obs variables
        x, y, z = float(single_clean[0]), float(single_clean[1]), float(single_clean[2])
        qw, qx, qy, qz = single_clean[3:7]
        vx, vy, vz = single_clean[7:10]
        wx, wy, wz = single_clean[10:13]
        height_err = abs(z - self.target_z) ##height_err is how far we are from target height
        above = z - self.target_z ##above -- how far above the target are we?If z<target then we will be negative values above target(since above is +)
        
        
        # update smoothness trackers
        self.du_hist.append(self.last_du)##add to the histories the vertical velocity and last_du which is difference in thrust between currently applied thrust and last applied thrust
        self.vz_hist.append(abs(vz))

        

        ## the final reward is calculated like reward = (closeness to target) + (moving in direction of target) - (go too fast near target) -
        ## (be above target)-(tilt/wander)-(be jerky)-(waste energy)
        

               # ---------- Reward ----------
       

        ##ground stall bookkeeping
        # Count how long we stay basically on the floor and not moving up, penalize every step on the ground
        ground_penalty =0
        if (z < 0.015) and (abs(vz) < 0.05):
            
            self.ground_steps += 1
            ground_penalty-=0.5
        else:
            # As soon as we get off the ground or move, reset
            self.ground_steps = 0
          

        ## tilt angle from quaternion (roll/pitch, ignore yaw), how far you are from being upright
        tilt_sin = np.sqrt(qx**2 + qy**2)
        tilt_sin = np.clip(tilt_sin, 0.0, 1.0)
        tilt_angle = 2.0 * np.arcsin(tilt_sin)  # radians, 0 = upright

        ## Height error and relative error w.r.t. target_z
        ##
        dz = z - self.target_z                    # [m]
        h_scale = max(self.target_z, 1e-3)        # avoid divide-by-zero
        dz_rel = dz / h_scale                     # dimensionless

        # 1) Height tracking: quadratic well around target_z (relative)
        k_z = 2.0
        r_z = -k_z * (dz_rel ** 2)

        # 2) Progress shaping in height (relative error improvement)
        prev_err_rel = abs(self.prev_dz) / h_scale
        curr_err_rel = abs(dz) / h_scale
        k_prog = 3.0
        r_progress = k_prog * np.clip(prev_err_rel - curr_err_rel, -0.2, 0.2)
        self.prev_dz = dz  # update for next step

        # 3) Vertical velocity penalty, stronger when near target (relative "near")
        near = np.exp(- (dz_rel / 0.1) ** 2)   # ~10% of target height
        k_vz_far  = 0.1
        k_vz_near = 2.0
        r_vz = -(k_vz_far + k_vz_near * near) * (vz ** 2)

        # 4) Lateral position & velocity: stay near (0,0) and don't drift sideways
        k_xy  = 1.0
        k_vxy = 0.2
        r_xy   = -k_xy  * (x**2 + y**2)
        r_vxy  = -k_vxy * (vx**2 + vy**2)

        # 5) Uprightness & angular rates
        k_tilt      = 2.0
        k_omega_rp  = 0.05   # roll + pitch rates
        k_omega_y   = 0.01   # yaw rate
        r_tilt  = -k_tilt * (tilt_angle ** 2)
        r_omega = -k_omega_rp * (wx**2 + wy**2) - k_omega_y * (wz**2)

        # Upright bonus: extra positive shaping when close to upright
        upright_scale = 0.5                     # max bonus per step
        upright_width = np.deg2rad(8.0)         # around ±8 degrees
        upright_bonus = upright_scale * np.exp(- (tilt_angle / upright_width) ** 2)

        # 6) Control effort & smoothness (torques + thrust changes)
        m_mag2 = float(np.dot(self.last_moments, self.last_moments))  # ||m||^2
        k_moment_abs   = 0.02
        k_moment_jump  = 0.05
        k_thrust_jump  = 0.2

        r_moment_abs    = -k_moment_abs  * m_mag2
        r_moment_jump   = -k_moment_jump * self.last_dm
        r_thrust_smooth = -k_thrust_jump * self.last_du

        # 7) Takeoff shaping (relative climb toward target)
        r_takeoff = 0.0
        if z < 0.02:
            # sitting on the ground is slightly bad
            r_takeoff -= 0.05
        elif z < self.target_z - 0.10:
            # reward being some fraction of the way up to target
            frac = np.clip(z / h_scale, 0.0, 1.0)
            r_takeoff += 0.1 * frac

        # 8) Soft ceiling shaping: discourage overshooting far above target
        if z > self.soft_ceiling:
            k_ceiling = 5.0
            r_ceiling = -k_ceiling * (z - self.soft_ceiling) ** 2
        else:
            r_ceiling = 0.0

        # Combine all reward terms
        reward = (
            r_z + r_progress +
            r_vz +
            r_xy + r_vxy +
            r_tilt + r_omega +
            r_moment_abs + r_moment_jump + r_thrust_smooth +
            r_takeoff + r_ceiling +
            upright_bonus
            +ground_penalty
        )

        # -Hover band bonus (relative to target height) 
        tilt_ok = tilt_angle < np.deg2rad(10.0)
        dz_band_rel = self.band / h_scale   # band in "fraction of target height"
        in_band = (abs(dz_rel) <= dz_band_rel) and (abs(vz) < 0.05) and tilt_ok

        if in_band:
            self.hover_count += 1
            reward += 1.0    # per-step bonus for really good hover
        else:
            self.hover_count = 0

        # Long, smooth hover -> early success
        if self.hover_count >= self.hover_required:
            mean_vz = float(np.mean(self.vz_hist)) if len(self.vz_hist) else 999.0
            mean_du = float(np.mean(self.du_hist)) if len(self.du_hist) else 999.0
            if mean_vz < 0.04 and mean_du < 0.010:
                reward += 50.0
                return obs, reward, True, False, {
                    "success": True,
                    "hover_steps": self.hover_count,
                    "mean_vz": mean_vz,
                    "mean_du": mean_du,
                }



        if self.ground_steps >= self.max_ground_steps:
            reward -= 100  # or 50.0
            return obs, reward, True, False, {
                "crash": True,
                "reason": "stalled_on_ground",
                "ground_steps": self.ground_steps,
            }


        # Ground / NaN crash
        if z < 0.01 or np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            reward -= 100
            return obs, reward, True, False, {"crash": True, "reason": "nan_or_below_ground"}

        # Flip crash
        if tilt_angle > np.deg2rad(120.0):
            reward -= 100
            return obs, reward, True, False, {"crash": True, "reason": "flipped"}

        # Hard ceiling termination
        if z > self.hard_ceiling:
            reward -= 50
            return obs, reward, True, False, {"ceiling": True}

        # Lateral bounds (keep around origin)
        r_xy = np.sqrt(x**2 + y**2)
        if r_xy > 0.8:  # 0.8m radius, not 8m
            reward -= 50
            return obs, reward, True, False, {"crash": True, "reason": "out_of_bounds"}


        terminated = False
        truncated = self.step_idx >= self.max_steps
        return obs, reward, terminated, truncated, {"hover_steps": self.hover_count}

# ---------- Helpers ----------
        
    def _get_single_obs(self) -> np.ndarray:
        """Return the 13D instantaneous state from MuJoCo."""
        pos = self.data.qpos[0:3]
        quat = self.data.qpos[3:7]
        linv = self.data.qvel[0:3]
        angv = self.data.qvel[3:6]
        return np.concatenate([pos, quat, linv, angv]).astype(np.float32)

    def _obs(self) -> np.ndarray:
        """
        Return stacked observations of shape (13 * n_stack,).
        Maintains a deque of the last n_stack single observations.
        """
        single = self._get_single_obs()

        if self.n_stack == 1:
            # No stacking, just return the current frame
            return single

        # Initialize the stack on first call (e.g., right after reset)
        if len(self.obs_stack) == 0:
            for _ in range(self.n_stack):
                self.obs_stack.append(single.copy())
        else:
            self.obs_stack.append(single.copy())

        stacked = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)
        return stacked

