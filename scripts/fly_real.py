"""Script to fly the real Crazyflie using the trained PPO policy.

Usage:
    python fly_real.py --uri radio://0/80/2M/E7E7E7E7E7 --model models/PPO/best_model.zip --norm models/PPO/vecnormalize.pkl
"""
import argparse
import time
import logging
import numpy as np
from collections import deque

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.CrazyflieDriver import CrazyflieDriver
from src.CrazyflieFirmware import CrazyflieFirmware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uri', type=str, default='radio://0/80/2M/E7E7E7E7E7', help='Crazyflie URI')
    parser.add_argument('--model', type=str, required=True, help='Path to PPO model zip')
    parser.add_argument('--norm', type=str, required=True, help='Path to VecNormalize pkl')
    parser.add_argument('--duration', type=float, default=5.0, help='Flight duration in seconds')
    args = parser.parse_args()

    # 1. Initialize Driver
    driver = CrazyflieDriver(uri=args.uri)
    
    # 2. Initialize Firmware (Controller)
    # We pass the driver so the firmware can send actions
    firmware = CrazyflieFirmware(
        model_path=args.model,
        norm_path=args.norm,
        driver=driver,
        hover_thrust=0.27
    )

    try:
        # 3. Connect to Drone
        driver.connect()
        
        # 4. Start Logging (to get sensor data)
        driver.start_logging()
        
        # Wait for Kalman filter to settle? (Optional but good practice)
        logger.info("Waiting for estimator to settle...")
        time.sleep(2.0)

        logger.info("Starting Flight Control Loop!")
        
        # 5. Control Loop
        # Run at 50Hz (0.02s) to match simulation
        rate = 50.0
        dt = 1.0 / rate
        
        start_time = time.time()
        
        # Frame stacking buffer (size 4)
        obs_stack = deque(maxlen=4)
        
        # Fill stack with initial observation
        initial_obs = driver.get_latest_obs()
        for _ in range(4):
            obs_stack.append(initial_obs)

        while (time.time() - start_time) < args.duration:
            loop_start = time.time()

            # A. Get latest observation from driver
            single_obs = driver.get_latest_obs()
            
            # B. Update stack
            obs_stack.append(single_obs)
            stacked_obs = np.concatenate(list(obs_stack))

            # C. Predict & Send Action
            # The firmware handles normalization and safety checks
            firmware.step_and_send(stacked_obs)

            # D. Sleep to maintain rate
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, dt - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Flight aborted by user!")
    except Exception as e:
        logger.error(f"Flight error: {e}")
    finally:
        logger.info("Landing/Stopping...")
        driver.stop()
        driver.disconnect()

if __name__ == '__main__':
    main()
