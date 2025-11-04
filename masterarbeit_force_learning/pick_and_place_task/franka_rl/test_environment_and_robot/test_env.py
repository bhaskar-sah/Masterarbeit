import mujoco
import mujoco.viewer
import time
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FRANKA_RL_DIR = os.path.dirname(CURRENT_DIR)
SCENE_XML_PATH = os.path.join(FRANKA_RL_DIR, 'franka', 'scene.xml')
print(f"Path to scene file: {SCENE_XML_PATH}")


try:
    # noinspection PyArgumentList
    model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
    data = mujoco.MjData(model)

    print("Model loaded successfully!")
    print(f"Robot joints: {model.njnt}")
    print(f"Actuators: {model.nu}")

    """
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched. Running for 10 seconds...")
        start = time.time()

        while viewer.is_running() and time.time() - start < 10:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
    """

    viewer = mujoco.viewer.launch_passive(model, data)
    while viewer.is_running():
        time.sleep(0.01)

    print("Viewer closed")

except Exception as e:
    print(f"Error: Could not load or view the model.\n{e}")