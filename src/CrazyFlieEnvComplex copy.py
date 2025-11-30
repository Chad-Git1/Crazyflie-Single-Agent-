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
        auto_landing: bool = False,
        landing_descent_rate: float = 0.4,   # m/s target descent
        landing_upright_gain: float = 4.0,   # P gain on qx,qy
        landing_rate_gain: float = 0.5,      # D gain on wx,wy
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

        

        # --- Sensor / dynamics randomization state ---
        self.obs_bias = np.zeros(self.obs_dim_single, dtype=np.float32)
        self.motor_scale = 1.0

        # NEW: per-episode sensor gain (e.g. z scale error)
        self.pos_gain = np.ones(3, dtype=np.float32)    # x,y,z gains
        self.vel_gain = np.ones(3, dtype=np.float32)    # vx,vy,vz gains

        # NEW: slowly drifting bias (random walk)
        self.bias_drift = np.zeros(self.obs_dim_single, dtype=np.float32)



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

        # --- Safety thresholds for sim2real guardrails ---
        # If these are exceeded while in HOVER mode, we hand over to the hard-coded landing.
        self.safety_max_tilt_rad = np.deg2rad(35.0)  # ~35 deg from upright
        self.safety_max_abs_vz = 5.0                 # m/s vertical speed limit
        self.safety_radius = 0.8                     # same as lateral bound radius

        # ───────────────────────────────────
        # New landing state machine (env-side)
        # ───────────────────────────────────
        self.phase = "HOVER"  # "HOVER" or "LANDING"
        self.auto_landing = auto_landing

        # Landing controller internal state
        self.landing_step_idx = 0
        self.landing_beta = 0.0          # blend between policy thrust and landing thrust
        self.landing_mode = "DESCEND"    # or "CATCH"
        self.landing_catch_steps = 0

        # Landing parameters (mirroring the external controller logic)
        self.landing_max_radius = 0.8    # must not leave this lateral radius
        self.landing_safe_radius = 0.4   # where we want to stay during landing
        self.landing_tilt_abort_deg = 25.0  # tilt > this -> go to CATCH mode
        self.landing_tilt_ok_deg = 10.0     # considered upright
        self.landing_beta_ramp_steps = 200  # steps to ramp beta 0→1
        self.landing_max_steps = 800        # safety cap on landing duration

        # Vertical profile for landing (m/s)
        self.landing_vz_fast = -0.30   # high up
        self.landing_vz_med  = -0.20
        self.landing_vz_mid  = -0.15
        self.landing_vz_slow = -0.10   # near ground

        # Vertical speed gain
        self.landing_k_vz = 0.4

     




    def _apply_obs_noise(self, single: np.ndarray) -> np.ndarray:
        """
        Apply realistic measurement model to a clean 13D state:
        - true state: [pos(3), quat(4), linv(3), angv(3)]
        - sensor gain + bias on pos & velocities
        - small white noise
        - slowly drifting bias (random walk)
        - keep quaternion unit-normalized
        """
        rng = getattr(self, "np_random", np.random)
        s = single.astype(np.float32).copy()

        # Unpack
        pos = s[0:3]
        quat = s[3:7]
        linv = s[7:10]
        angv = s[10:13]

        # 1) Per-episode gains (scale errors)
        pos_meas = self.pos_gain * pos
        vel_meas = self.vel_gain * linv

        # 2) Per-episode bias
        pos_meas += self.obs_bias[0:3]
        vel_meas += self.obs_bias[7:10]
        angv_meas = angv + self.obs_bias[10:13]

        # 3) Slowly drifting bias (random walk, small step)
        if self.obs_noise_std > 0.0:
            drift_step = rng.normal(
                0.0,
                self.obs_noise_std * 0.01,  # small step
                size=self.obs_dim_single,
            ).astype(np.float32)
            self.bias_drift += drift_step
        else:
            self.bias_drift[:] = 0.0

        pos_meas += self.bias_drift[0:3]
        vel_meas += self.bias_drift[7:10]
        angv_meas += self.bias_drift[10:13]

        # 4) Per-step white noise (channel-specific)
        if self.obs_noise_std > 0.0:
            # positions: bigger noise on z than x,y
            pos_meas[0:2] += rng.normal(
                0.0, self.obs_noise_std * 0.5, size=2
            ).astype(np.float32)
            pos_meas[2] += rng.normal(0.0, self.obs_noise_std * 1.0)  # <--- FIXED

            # linear velocities
            vel_meas += rng.normal(
                0.0, self.obs_noise_std * 0.7, size=3
            ).astype(np.float32)

            # angular velocities: slightly smaller
            angv_meas += rng.normal(
                0.0, self.obs_noise_std * 0.4, size=3
            ).astype(np.float32)

        # 5) Quaternion: keep it clean or add tiny noise + renorm
        quat_meas = quat.copy()
        # optional tiny attitude error:
        # quat_meas += rng.normal(0.0, self.obs_noise_std * 0.1, size=4).astype(np.float32)
        norm_q = np.linalg.norm(quat_meas)
        if norm_q > 1e-6:
            quat_meas /= norm_q

        # 6) Repack
        noisy = np.concatenate([pos_meas, quat_meas, vel_meas, angv_meas]).astype(np.float32)

        # 7) OPTIONAL: rare outlier / glitch
        if self.obs_noise_std > 0.0 and rng.random() < 1e-3:
            glitch = rng.normal(0.0, self.obs_noise_std * 10.0, size=3).astype(np.float32)
            noisy[0:3] += glitch

        return noisy


    def _sample_episode_randomization(self):
        """
        Sample all per-episode random factors:
        - sensor bias (pos + velocities)
        - sensor gain (pos/vel scale factors)
        - motor thrust gain
        - frame_skip jitter
        """
        rng = getattr(self, "np_random", np.random)

        # 1) Observation bias: like altimeter + IMU offset
        if self.obs_bias_std > 0.0:
            # bias only on pos (0:3) and velocities (7:13)
            bias = rng.normal(
                loc=0.0,
                scale=self.obs_bias_std,
                size=self.obs_dim_single,
            ).astype(np.float32)
            # you can zero out quaternion bias explicitly if you want
            bias[3:7] = 0.0
            self.obs_bias = bias
        else:
            self.obs_bias[:] = 0.0

        # 2) Sensor gain: per-axis scale error
        # e.g. ±3% on z, ±2% on x,y, ±5% on velocities
        self.pos_gain[:] = 1.0
        self.vel_gain[:] = 1.0
        if self.motor_scale_std > 0.0:
            self.pos_gain[0:2] = 1.0 + rng.normal(0.0, 0.02, size=2)     # x,y
            self.pos_gain[2]   = 1.0 + rng.normal(0.0, 0.03)             # z
            self.vel_gain[:]   = 1.0 + rng.normal(0.0, 0.05, size=3)     # vx,vy,vz

        # 3) Motor thrust gain error (episode-level)
        if self.motor_scale_std > 0.0:
            self.motor_scale = float(1.0 + rng.normal(0.0, self.motor_scale_std))
        else:
            self.motor_scale = 1.0

        # 4) Frame skip jitter (control frequency variation)
        if self.frame_skip_jitter > 0:
            jitter = int(rng.integers(
                -self.frame_skip_jitter,
                self.frame_skip_jitter + 1
            ))
            self.frame_skip = max(1, self.frame_skip_base + jitter)
        else:
            self.frame_skip = self.frame_skip_base

        # 5) Reset drift state (random walk starts from 0)
        self.bias_drift[:] = 0.0



    
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

                # Reset landing state machine
        self.phase = "HOVER"
        self.landing_step_idx = 0
        self.landing_beta = 0.0
        self.landing_mode = "DESCEND"
        self.landing_catch_steps = 0
        # In case previous episode ended in landing
        self._end_noise_free_landing()


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

    def step(self, action: np.ndarray):
        # Convert action to proper array/shape


        a = np.asarray(action, dtype=np.float32).squeeze()

          # ── Landing phase override ───────────────────────
        if self.auto_landing and (self.phase == "LANDING"):
            # In landing phase: ignore normal hover reward logic and
            # run the landing controller instead.
            return self._step_landing(action)
        
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
                for _ in range(self.n_stack):
                    self.obs_stack.append(single.copy())
            else:
                self.obs_stack.append(single.copy())
            obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)

        # Unpack state from clean obs for reward / safety
        x, y, z = float(single_clean[0]), float(single_clean[1]), float(single_clean[2])
        qw, qx, qy, qz = single_clean[3:7]
        vx, vy, vz = single_clean[7:10]
        wx, wy, wz = single_clean[10:13]

        # update smoothness trackers
        self.du_hist.append(self.last_du)
        self.vz_hist.append(abs(vz))

        # ---------- Reward ----------
        ground_penalty = 0.0
        if (z < 0.015) and (abs(vz) < 0.05):
            self.ground_steps += 1
            ground_penalty -= 0.5
        else:
            self.ground_steps = 0

        # Tilt angle from quaternion (roll/pitch, ignore yaw)
        tilt_sin = np.sqrt(qx**2 + qy**2)
        tilt_sin = np.clip(tilt_sin, 0.0, 1.0)
        tilt_angle = 2.0 * np.arcsin(tilt_sin)

        dz = z - self.target_z
        h_scale = max(self.target_z, 1e-3)
        dz_rel = dz / h_scale

        # 1) Height tracking
        k_z = 2.0
        r_z = -k_z * (dz_rel ** 2)

        # 2) Progress shaping
        prev_err_rel = abs(self.prev_dz) / h_scale
        curr_err_rel = abs(dz) / h_scale
        k_prog = 3.0
        r_progress = k_prog * np.clip(prev_err_rel - curr_err_rel, -0.2, 0.2)
        self.prev_dz = dz

        # 3) Vertical velocity penalty (stronger near target)
        near = np.exp(- (dz_rel / 0.1) ** 2)
        k_vz_far  = 0.1
        k_vz_near = 2.0
        r_vz = -(k_vz_far + k_vz_near * near) * (vz ** 2)

        # 4) Lateral position & velocity
        k_xy  = 1.0
        k_vxy = 0.2
        r_xy   = -k_xy  * (x**2 + y**2)
        r_vxy  = -k_vxy * (vx**2 + vy**2)

        # 5) Uprightness & angular rates
        k_tilt      = 2.0
        k_omega_rp  = 0.05
        k_omega_y   = 0.01
        r_tilt  = -k_tilt * (tilt_angle ** 2)
        r_omega = -k_omega_rp * (wx**2 + wy**2) - k_omega_y * (wz**2)

        upright_scale = 0.5
        upright_width = np.deg2rad(8.0)
        upright_bonus = upright_scale * np.exp(- (tilt_angle / upright_width) ** 2)

        # 6) Control effort & smoothness
        m_mag2 = float(np.dot(self.last_moments, self.last_moments))
        k_moment_abs   = 0.02
        k_moment_jump  = 0.05
        k_thrust_jump  = 0.2
        r_moment_abs    = -k_moment_abs  * m_mag2
        r_moment_jump   = -k_moment_jump * self.last_dm
        r_thrust_smooth = -k_thrust_jump * self.last_du

        # 7) Takeoff shaping
        r_takeoff = 0.0
        if z < 0.02:
            r_takeoff -= 0.05
        elif z < self.target_z - 0.10:
            frac = np.clip(z / h_scale, 0.0, 1.0)
            r_takeoff += 0.1 * frac

        # 8) Soft ceiling shaping
        if z > self.soft_ceiling:
            k_ceiling = 5.0
            r_ceiling = -k_ceiling * (z - self.soft_ceiling) ** 2
        else:
            r_ceiling = 0.0

        reward = (
            r_z + r_progress +
            r_vz +
            r_xy + r_vxy +
            r_tilt + r_omega +
            r_moment_abs + r_moment_jump + r_thrust_smooth +
            r_takeoff + r_ceiling +
            upright_bonus +
            ground_penalty
        )

        # ---------- Hover-band success ----------
        tilt_ok = tilt_angle < np.deg2rad(10.0)
        dz_band_rel = self.band / h_scale
        in_band = (abs(dz_rel) <= dz_band_rel) and (abs(vz) < 0.05) and tilt_ok

        if in_band:
            self.hover_count += 1
            reward += 1.0
            if self.hover_count >= self.hover_required:
                mean_vz = float(np.mean(self.vz_hist)) if len(self.vz_hist) else 999.0
                mean_du = float(np.mean(self.du_hist)) if len(self.du_hist) else 999.0
                if mean_vz < 0.04 and mean_du < 0.010:
                    reward += 50.0
                    info = {
                        "success": True,
                        "hover_steps": self.hover_count,
                        "mean_vz": mean_vz,
                        "mean_du": mean_du,
                    }
                    if self.auto_landing:
                        # Start landing instead of terminating
                        self._start_landing_phase("success")
                        info.update({"phase": "landing_start"})
                        return obs, reward, False, False, info
                else:
                    return obs, reward, True, False, info
        else:
            self.hover_count = 0

        # ---------- Hard crashes / terminations ----------
        if self.ground_steps >= self.max_ground_steps:
            reward -= 100.0
            info = {
                "crash": True,
                "reason": "stalled_on_ground",
                "ground_steps": self.ground_steps,
            }
            return obs, reward, True, False, info

        if z < 0.01 or np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            reward -= 100.0
            return obs, reward, True, False, {"crash": True, "reason": "nan_or_below_ground"}

        if tilt_angle > np.deg2rad(120.0):
            reward -= 100.0
            return obs, reward, True, False, {"crash": True, "reason": "flipped"}

        if z > self.hard_ceiling:
            reward -= 50.0
            if self.auto_landing:
                        # Start landing instead of terminating
                        self._start_landing_phase("hard-ceiling")
                        
                        return obs, reward, False, False, {"ceiling": True}
            else:
                    return obs, reward, True, False, info
          

        r_xy_pos = np.sqrt(x**2 + y**2)
        if r_xy_pos > 0.8:
            reward -= 50.0
            return obs, reward, True, False, {"crash": True, "reason": "out_of_bounds"}

        terminated = False
        timeout = self.step_idx >= self.max_steps

        if timeout:
            info = {
                "hover_steps": self.hover_count,
                "timeout": True,
            }
            if self.auto_landing:
                # Start landing instead of truncating
                self._start_landing_phase("timeout")
                info.update({"phase": "landing_start"})
                return obs, reward, False, False, info
            else:
                truncated = True
                return obs, reward, terminated, truncated, info
        

        # Normal non-timeout return
        truncated = False
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
# ========== Landing helpers for external controller ==========

      # ========== Landing helpers for external controller ==========

    def get_altitude(self) -> float:
        """Current z in world coordinates."""
        return float(self.data.qpos[2])

    def safe_ground_height(self) -> float:
        """
        Height below which we consider the drone 'landed' enough
        to cut motors (tune this if needed).
        """
        return 0.03  # 3 cm

    def landing_action(self, k: int, total_steps: int) -> np.ndarray:
        """
        Thrust-only landing with vertical-speed tracking.

        - No torques (mx=my=mz=0) -> we don't inject tilt.
        - Try to follow a gentle desired descent speed v_des(z).
        - Keep thrust bounded away from 0 so we don't free-fall.
        """
        # --- Read state we care about ---
        z  = float(self.data.qpos[2])
        vz = float(self.data.qvel[2])

        # Height above "ground"
        h = z - self.safe_ground_height()
        if h < 0.0:
            h = 0.0

        # Piecewise desired vertical speed [m/s]
        # Higher up: a bit faster; near ground: very gentle
        if h > 0.6:
            v_des = -0.30
        elif h > 0.3:
            v_des = -0.22
        elif h > 0.1:
            v_des = -0.15
        else:
            v_des = -0.08

        # Vertical speed control: track v_des
        # If vz is more negative than v_des (falling too fast),
        #   err_v = vz - v_des < 0 -> u increases -> slows descent.
        # If vz is too small / positive (not descending enough),
        #   err_v > 0 -> u decreases -> speeds descent.
        k_v = 0.4
        err_v = vz - v_des
        u = self.HOVER_THRUST - k_v * err_v

        # Don't let landing thrust get *too* small or we'll free-fall.
        # This is a "floor" on thrust during landing.
        u_min = 0.12                        # tune if needed
        u_max = self.action_space.high[0]
        u = float(np.clip(u, u_min, u_max))

        # Torques OFF during landing: we rely on the last policy state
        # to already be upright and let dynamics keep it near there.
        return np.array([u, 0.0, 0.0, 0.0], dtype=np.float32)



    def cut_motors(self) -> None:
        """Immediately zero all actuators."""
        self.u_cmd = 0.0
        self.data.ctrl[:] = 0.0
        # ========== Noise control for scripted landing ==========

        # ───────────────────────────────────
    # Noise control for scripted landing
    # ───────────────────────────────────


    def _start_landing_phase(self, reason: str):
        """
        Called once when the hover episode ends (success or timeout or failure)
        and auto_landing=True. Does NOT terminate the Gym episode yet.
        """
        self.phase = "LANDING"
        self.landing_step_idx = 0
        self.landing_beta = 0.0
        self.landing_mode = "DESCEND"
        self.landing_catch_steps = 0
        self._begin_noise_free_landing()
        self.pre_landing_reason = reason

    def _begin_noise_free_landing(self):
        """
        Temporarily disable action noise and motor scaling so the
        landing controller is robust to training-time randomization.
        """
        self._landing_noise_backup = {
            "action_noise_std": self.action_noise_std,
            "motor_scale": self.motor_scale,
        }
        self.action_noise_std = 0.0
        self.motor_scale = 1.0

    def _end_noise_free_landing(self):
        """Restore noise/randomization after landing ends."""
        if hasattr(self, "_landing_noise_backup"):
            self.action_noise_std = self._landing_noise_backup["action_noise_std"]
            self.motor_scale = self._landing_noise_backup["motor_scale"]
            del self._landing_noise_backup

    def safe_ground_height(self) -> float:
        """Height threshold that counts as 'on the ground' for landing logic."""
        return 0.03  # 3 cm

    def _tilt_and_radius(self):
        """
        Compute tilt angle (rad) from quaternion (roll/pitch only)
        and lateral distance r from origin in the x-y plane.
        """
        x, y, z = self.data.qpos[0:3]
        qw, qx, qy, qz = self.data.qpos[3:7]

        tilt_sin = np.sqrt(qx * qx + qy * qy)
        tilt_sin = np.clip(tilt_sin, 0.0, 1.0)
        tilt_angle = 2.0 * np.arcsin(tilt_sin)  # radians

        r = float(np.sqrt(x * x + y * y))
        return tilt_angle, r
    
    def _step_landing(self, action: np.ndarray):
        """
        Landing step:
          - reuses policy torques (mx,my,mz) from the incoming action,
          - overrides thrust by blending policy thrust with a vertical-speed
            landing controller: u = (1-beta)*u_pol + beta*u_land,
          - has DESCEND and CATCH sub-modes.

        Returns: obs, reward, terminated, truncated, info
        """

        # Convert external action -> a_pol
        a_pol = np.asarray(action, dtype=np.float32).squeeze()
        if a_pol.shape != (4,):
            raise ValueError("Landing expects 4D action [thrust, mx, my, mz].")

        # Policy thrust and torques
        u_pol = float(np.clip(
            a_pol[0],
            self.action_space.low[0],
            self.action_space.high[0],
        ))
        m_pol = a_pol[1:4]
        # Clip torques to [-1,1] as usual
        m_pol = np.clip(m_pol, -1.0, 1.0)

        # Clean state from MuJoCo
        x, y, z = self.data.qpos[0:3]
        vx, vy, vz = self.data.qvel[0:3]

        tilt_angle, r = self._tilt_and_radius()
        tilt_deg = float(np.rad2deg(tilt_angle))

        # Initialize landing state on first step
        if self.landing_step_idx == 0:
            self.landing_beta = 0.0
            self.landing_mode = "DESCEND"
            self.landing_catch_steps = 0

        mode = self.landing_mode

        # ───────────── 1) Mode switch: go to CATCH if things get bad ─────────────
        if (tilt_deg > self.landing_tilt_abort_deg
                or r > self.landing_max_radius):
            mode = "CATCH"
            self.landing_mode = "CATCH"
            self.landing_catch_steps = 0

        # ───────────── 2) Decide desired vertical speed and beta ─────────────
        # Default: no descent unless set below
        v_des = 0.0

        if mode == "CATCH":
            # In catch mode: try to hover / hold altitude,
            # and temporarily give more authority back to the policy (beta→0).
            v_des = 0.0
            self.landing_beta = max(0.0, self.landing_beta - 0.05)

            # Once we're upright, inside safe radius, and slow laterally,
            # count some "safe" steps; then return to DESCEND mode.
            if (tilt_deg < self.landing_tilt_ok_deg
                    and r < self.landing_safe_radius
                    and abs(vx) < 0.2 and abs(vy) < 0.2):
                self.landing_catch_steps += 1
            else:
                self.landing_catch_steps = 0

            if self.landing_catch_steps > 50:
                self.landing_mode = "DESCEND"

        else:  # DESCEND
            # Height above ground
            h = z - self.safe_ground_height()
            if h < 0.0:
                h = 0.0

            # Piecewise vertical speed profile (same as external logic)
            if h > 0.8:
                v_des = self.landing_vz_fast
            elif h > 0.4:
                v_des = self.landing_vz_med
            elif h > 0.2:
                v_des = self.landing_vz_mid
            else:
                v_des = self.landing_vz_slow

            # Ramp beta from 0→1 over landing_beta_ramp_steps
            step_idx = self.landing_step_idx
            self.landing_beta = min(
                1.0,
                step_idx / max(1, self.landing_beta_ramp_steps)
            )

        beta = self.landing_beta

        # ───────────── 3) Compute landing thrust u_land ─────────────
        # PD on vertical speed: want vz ≈ v_des
        err_v = vz - v_des      # <0 if falling faster than desired
        u_land = self.HOVER_THRUST - self.landing_k_vz * err_v

        # Clamp landing thrust to avoid free-fall
        u_min = 0.12
        u_max = float(self.action_space.high[0])
        u_land = float(np.clip(u_land, u_min, u_max))

        # ───────────── 4) Blend thrust: u = (1-beta)*u_pol + beta*u_land ─────────────
        u = (1.0 - beta) * u_pol + beta * u_land
        u = float(np.clip(u, u_min, u_max))

        # Apply thrust smoothing & torques using existing machinery
        self._apply_thrust(u, m_pol)

        # Step physics
        for _ in range(self.frame_skip):
            mj.mj_step(self.model, self.data)
        self.step_idx += 1
        self.landing_step_idx += 1

        # Observation for caller (use same noisy pipeline for consistency)
        single_clean = self._get_single_obs()
        single = self._apply_obs_noise(single_clean)

        if self.n_stack == 1:
            obs = single
        else:
            if len(self.obs_stack) == 0:
                for _ in range(self.n_stack):
                    self.obs_stack.append(single.copy())
            else:
                self.obs_stack.append(single.copy())
            obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)

        # ───────────── 5) Check landing completion ─────────────
        low_enough = (z <= self.safe_ground_height())
        slow_enough = (abs(vz) < 0.10)
        upright_enough = (tilt_deg < 10.0)
        inside_radius = (r < self.landing_safe_radius)

        landed = low_enough and slow_enough and upright_enough and inside_radius
        timeout = (self.landing_step_idx >= self.landing_max_steps)

        if landed or timeout:
            self._end_noise_free_landing()
            self.phase = "HOVER"  # or "DONE"; from Gym pov we end episode

        # No learning reward during landing; keep it 0
        reward = 0.0
        terminated = landed or timeout
        truncated = False

        info = {
            "phase": "landing",
            "landing_mode": self.landing_mode,
            "landing_beta": beta,
            "landing_landed": landed,
            "landing_timeout": timeout,
            "tilt_deg": tilt_deg,
            "radius": r,
            "vz": float(vz),
            "pre_landing_reason": self.pre_landing_reason,
        }

        return obs, reward, terminated, truncated, info
    


