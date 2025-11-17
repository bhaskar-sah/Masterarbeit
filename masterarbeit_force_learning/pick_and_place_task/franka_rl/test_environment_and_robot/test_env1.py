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


    mujoco.mj_forward(model, data)
    # time.sleep(0.1)
    print("Robot set to 'home' keyframe position")

    body_names = ["robot_pedestal", "task_table", "bottle", "link0"]
    print("\n--- Component Origins (xpos) ---")

    for name in body_names:
        # Get the ID of the body by its name
        body_id = model.body(name).id

        # Access the position (x, y, z) and orientation (rotation matrix)
        position = data.xpos[body_id]
        orientation_matrix = data.xmat[body_id].reshape(3, 3)

        print(f"Body: '{name}' (ID: {body_id})")
        print(f"  Position (x, y, z): {position}")
        print(f"  Orientation Matrix (3x3):\n{orientation_matrix}\n")

    mujoco.viewer.launch(model, data)

except Exception as e:
    print(f"Error loading model: {e}")