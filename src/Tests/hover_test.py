import time
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

def hover_test():
    '''
        I am trying to make the drone hover towards a target height 
    '''
    path = os.path.abspath(scenePath)

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    # Base hover thrust (keeps the drone at a stationary hover in this case, counteracts gravity)
    BASE_THRUST = 0.26487 ## weight = mass (0.027 kg) x gravity (9.81 m/s^2)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Thrust towards a target height
        target_height = 0.5   # target height in meters
        kp = 0.3              # proportional gain --> Scales thrust based on drone distance from target
        kd = 0.1              # derivative gain --> Scales thrust based on drone velocity

        while viewer.is_running():
            # Add delay so it doesn't run super fast (we want to see the sim)
            time.sleep(0.005)

            z_position = data.qpos[2]
            z_velocity = data.qvel[2]
            
            p_error = target_height - z_position
            d_error = -z_velocity
            
            # PD control law
            # kp * p_error part --> Reacts to position error e.g. drone below target --> p_error > 0 --> thrust increases
            # kd * d_error part --> Reacts to velocity e.g. drone falling --> d_error = positive velocity --> Add to thrust
            u = BASE_THRUST + kp * p_error + kd * d_error
            print(f"Thrust: {u}")
            
            data.ctrl[:] = np.array([u, 0.0, 0.0, 0.0])
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    hover_test()