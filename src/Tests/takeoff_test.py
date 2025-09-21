import mujoco
import mujoco.viewer
import numpy as np
import os

scenePath = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "Assets", 
    "bitcraze_crazyflie_2", 
    "scene.xml"
)

def takeoff_test():
    '''
        I am trying to make the drone hover in a stable manner as a test to play around with mujoco and understand what our AI
        needs to do in order to succeed.  
    '''
    path = os.path.abspath(scenePath)

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    # Hover thrust 
    THRUST = 0.264875 ## weight = mass (0.027 kg) x gravity (9.81 m/s^2) + 0.000005

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Movement
            x = 0.0
            y = 0.0
            z = 0.005

            # Clip values to min and max of the actuators' ranges
            x = np.clip(x, -1, 1)
            y = np.clip(y, -1, 1)

            # Apply control
            data.ctrl[:] = np.array([THRUST, x, y, z])

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    takeoff_test()