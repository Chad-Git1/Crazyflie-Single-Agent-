import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class CrazyflieEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, xml_path, num_drones=1, target_height=0.5):
        """
        Initialize the training environment

        Parameters
        ----------
        xml_path : string 
            Path to xml MuJoCo scene
        num_drones : int
            Number of drones in the MuJoCo scene
        target_height : float
            For now we are doing a basic hover test, this is the target hover height
        """
        super().__init__()

        # Store config
        self.num_drones = num_drones
        self.target_height = target_height

        # MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Drone obervation space
        # [pos(3) + quaternion(4) + velocity(3) + angular_velocity(3)] = 13 per drone
        obs_high = np.inf * np.ones(13 * self.num_drones, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Drone action space (see aicraft axes: https://en.wikipedia.org/wiki/Aircraft_principal_axes)
        # thrust + roll + pitch + yaw = 4 per drone
        act_high = np.tile([0.1, 1, 1, 1], self.num_drones).astype(np.float32)
        act_low = np.tile([-0.1, -1, -1, -1], self.num_drones).astype(np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        # Viewer
        self.viewer = None

        # 7 qpos, 6 qvel, and 4 action controls per drone
        self.qpos_per_drone = 7
        self.qvel_per_drone = 6
        self.ctrl_per_drone = 4


    def reset(self):
        """
        Reset the training environment
        """
        super().reset()

        mujoco.mj_resetData(self.model, self.data)

        # Initialize drones at slightly different z heights with velocities of 0
        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            self.data.qpos[base_qpos + 2] = 0.1 + 0.01 * np.random.rand()

            base_qvel = i * self.qvel_per_drone
            self.data.qvel[base_qvel: base_qvel + self.qvel_per_drone] = 0.0

        return self._get_obs(), {}


    def step(self, action):
        """
        Take a step (action) and track the observation
        """

        # Clip action within action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        BASE_THRUST = 0.26487
        kp = 0.3
        kd = 0.1

        total_reward = 0.0
        done = False

        # Fill control array
        ctrl = np.zeros(self.num_drones * self.ctrl_per_drone, dtype=np.float32)

        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            base_qvel = i * self.qvel_per_drone
            base_ctrl = i * self.ctrl_per_drone

            z_position = self.data.qpos[base_qpos + 2]
            z_velocity = self.data.qvel[base_qvel + 2]

            p_error = self.target_height - z_position
            d_error = -z_velocity

            # Combine RL action (only thrust offset part, no oritentation for now) with PD hover control
            thrust = BASE_THRUST + kp * p_error + kd * d_error + action[base_ctrl]
            ctrl[base_ctrl: base_ctrl + 4] = np.array([thrust, 0.0, 0.0, 0.0])

            # Reward is per drone
            total_reward += -abs(self.target_height - z_position)

            # Terminate if any drone crashes or reaches target
            done = (z_position < 0.0) or (abs(z_position - self.target_height) < 0.01)


        self.data.ctrl[:] = ctrl

        # Take a step and track observation
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        return obs, total_reward, done, False, {}


    def _get_obs(self):
        """
        Retrieve an observation of the current drone state
        """
        obs = []
        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            base_qvel = i * self.qvel_per_drone

            # Observation
            # qpos stores (position[x, y, z], orientation[qz, qy, qz, q2])
            # qvel stores (velocity[vx, vy, vz], angular velocity[wx, wy, wz])

            pos = self.data.qpos[base_qpos: base_qpos + 3]
            quat = self.data.qpos[base_qpos + 3: base_qpos + 7]
            vel = self.data.qvel[base_qvel: base_qvel + 3]
            ang_vel = self.data.qvel[base_qvel + 3: base_qvel + 6]

            obs.extend(np.concatenate([pos, quat, vel, ang_vel]))

        return np.array(obs, dtype=np.float32)


    def render(self):
        """
        Render the model in MuJoCo
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()


    def close(self):
        """
        Close the MuJoCo viewer
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
