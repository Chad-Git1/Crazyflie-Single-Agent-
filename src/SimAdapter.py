import os
import time
import logging
import numpy as np
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from src.CrazyFlieEnvComplex import CrazyFlieEnv
from src.FlightDataLogger import FlightDataLogger
from src.CrazyflieFirmware import CrazyflieFirmware

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SimAdapter:
    """Run trained policy inside CrazyFlieEnvComplex using the CrazyflieFirmware class.

    This adapter validates that the Firmware class (intended for hardware)
    performs correctly within the simulation environment.
    """

    def __init__(self, model_path: str, norm_path: str, xml_path: str, target_z: float = 1.0, max_steps: int = 1500):
        self.xml_path = xml_path
        self.target_z = target_z
        self.max_steps = max_steps

        # Initialize the Firmware controller
        # We pass None for vec_env because we might not have the exact training env structure here
        # but CrazyflieFirmware handles loading VecNormalize without it if needed (or we can pass a dummy).
        # For best results with VecNormalize, we usually need a dummy env to unpickle it.
        
        def _thunk():
             return Monitor(CrazyFlieEnv(xml_path=xml_path, target_z=target_z, max_steps=max_steps, hover_required_steps=600))
        dummy_env = DummyVecEnv([_thunk])

        self.firmware = CrazyflieFirmware(
            model_path=model_path,
            norm_path=norm_path,
            vec_env=dummy_env,
            hover_thrust=0.27
        )
        
    def run_episode(self, render: bool = False, log_id: str = "sim_001"):
        # Create a fresh single env for rendering and stepping
        env = CrazyFlieEnv(xml_path=self.xml_path, target_z=self.target_z, max_steps=self.max_steps, n_stack=4, hover_required_steps=600)
        obs_raw, _ = env.reset()

        logger.info("Starting simulation episode with CrazyflieFirmware")
        dt_sim = env.model.opt.timestep
        dt_step = dt_sim * env.frame_skip

        logger_adapter = FlightDataLogger(log_id, output_dir=os.path.join(os.path.dirname(__file__), '..', 'flight_logs'))

        total_reward = 0.0
        terminated = False
        truncated = False
        step = 0

        t0 = time.time()
        while not (terminated or truncated) and step < self.max_steps:
            # Use the Firmware to predict action
            # The firmware handles normalization, clamping, and safety checks internally
            action = self.firmware.predict_action(obs_raw)

            # Step environment
            obs_raw, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            # Extract useful telemetry from raw obs
            # obs_raw is stacked frames: [frame0, frame1, frame2, frame3]
            # single frame structure: [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
            # we take the newest frame (last 13 values)
            single = obs_raw[-13:]
            x, y, z = float(single[0]), float(single[1]), float(single[2])
            # Get linear velocities
            vx, vy, vz = float(single[7]), float(single[8]), float(single[9])
            # Angular velocities
            p, q, r = float(single[10]), float(single[11]), float(single[12])

            # For orientation, we have quaternion; convert to approximated small euler angles if needed
            # But for logging we keep quaternion
            qw, qx, qy, qz = single[3], single[4], single[5], single[6]
            # Approximate roll/pitch via quaternion (small-angle approx)
            # This isn't exact but good enough for basic checks
            # roll = atan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy)) etc.
            # We'll compute exact euler for logging
            try:
                # Quaternion to euler (ZYX)
                t0_q = 2.0 * (qw * qx + qy * qz)
                t1_q = 1.0 - 2.0 * (qx * qx + qy * qy)
                roll = float(np.arctan2(t0_q, t1_q))
                t2_q = 2.0 * (qw * qy - qz * qx)
                t2_q = np.clip(t2_q, -1.0, 1.0)
                pitch = float(np.arcsin(t2_q))
                t3_q = 2.0 * (qw * qz + qx * qy)
                t4_q = 1.0 - 2.0 * (qy * qy + qz * qz)
                yaw = float(np.arctan2(t3_q, t4_q))
            except Exception:
                roll = pitch = yaw = 0.0

            # Log step
            logger_adapter.log_step(
                step=step,
                x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz,
                roll=roll, pitch=pitch, yaw=yaw,
                p=p, q=q, r=r,
                thrust=float(action[0]), mx=float(action[1]), my=float(action[2]), mz=float(action[3]),
                battery=3.8,
                reward=float(reward)
            )

            if render:
                # Basic render by syncing MuJoCo viewer if desired
                pass

            step += 1

        duration = time.time() - t0
        logger.info(f"Episode finished: steps={step}, total_reward={total_reward:.3f}, duration={duration:.2f}s")
        logger_adapter.save()
        return {"steps": step, "total_reward": total_reward, "duration": duration}
