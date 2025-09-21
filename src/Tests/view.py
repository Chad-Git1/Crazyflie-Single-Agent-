import mujoco
import mujoco.viewer

def main():
    '''
    Loads and visualizes the Crazyflie 2 drone model in MuJoCo.

    This creates a Passive simulation : No active controls or motor forces are applied.
    '''
    # Path to Crazyflie model
    path = r".\Assets\bitcraze_crazyflie_2\scene.xml"

    # Load the model
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()