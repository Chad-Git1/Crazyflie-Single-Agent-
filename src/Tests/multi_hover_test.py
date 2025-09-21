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
    "scene_multi.xml"
)

def multi_hover_test():
    """
    Initialize two drones side by side, then takeoff and hover both
    """
    path = os.path.abspath(scenePath)

    model = mujoco.MjModel.from_xml_path(path)
    
    data = mujoco.MjData(model)

    BASE_THRUST = 0.26487  # hover thrust

    with mujoco.viewer.launch_passive(model, data) as viewer:
        target_height = 0.5
        kp = 0.3
        kd = 0.1
        print("\n\nVerifying actuator names")
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(i, name)
        print("\n\n")


        while viewer.is_running():
            time.sleep(0.005)

            # --- Drone 1 ---

            z_position_1 = data.qpos[2]
            z_velocity_1 = data.qvel[2]
            
            p_error_1 = target_height - z_position_1
            d_error_1 = -z_velocity_1

            u1 = BASE_THRUST + kp * p_error_1 + kd * d_error_1


            # --- Drone 2 ---

            # Flatten drone 1 & 2 pos/vel into qpos/qvel in scene_multi.xml, so offset index by 7/6
            z_position_2 = data.qpos[2 + 7]
            z_velocity_2 = data.qvel[2 + 6]
            
            p_error_2 = target_height - z_position_2
            d_error_2 = -z_velocity_2

            u2 = BASE_THRUST + kp * p_error_2 + kd * d_error_2

            # Control
            data.ctrl[:] = np.array([
                u1, 0.0, 0.0, 0.0,
                u2, 0.0, 0.0, 0.0
            ])

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    multi_hover_test()
