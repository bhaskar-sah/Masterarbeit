import os
import mujoco
import mujoco.viewer
import time

current_dir = os.path.dirname(os.path.realpath(__file__))
SCENE_XML_PATH = os.path.join(current_dir, "panda_robot.xml")

print(f"Path to scene file: {SCENE_XML_PATH}")

try:
    model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
    data = mujoco.MjData(model)

    print("Model loaded successfully!")
    print(f"Robot joints: {model.njnt}")
    print(f"Actuators: {model.nu}")

    home_key_id = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home_key_id)

    # self.home_key_id = self.model.key("home").id
    # mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id


    # mujoco.mj_forward(model, data)
    # time.sleep(0.1)
    print("Robot set to 'home' keyframe position")

    mujoco.viewer.launch(model, data)

except Exception as e:
    print(f"Error loading model: {e}")