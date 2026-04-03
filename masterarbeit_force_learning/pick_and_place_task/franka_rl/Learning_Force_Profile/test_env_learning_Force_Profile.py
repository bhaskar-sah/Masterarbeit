import os
import mujoco
import mujoco.viewer
import time
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
SCENE_XML_PATH = os.path.join(current_dir, "robot_panda_push_force.xml")

print(f"Path to scene file: {SCENE_XML_PATH}")

#### use for simple test ######
# try:
#     model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
#     data = mujoco.MjData(model)
#
#     print("Model loaded successfully!")
#     print(f"Robot joints: {model.njnt}")
#     print(f"Actuators: {model.nu}")
#
#     home_key_id = model.key("home").id
#     mujoco.mj_resetDataKeyframe(model, data, home_key_id)
#
#     # self.home_key_id = self.model.key("home").id
#     # mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id
#
#
#     mujoco.mj_forward(model, data)
#     # time.sleep(0.1)
#     print("Robot set to 'home' keyframe position")
#
#     body_names = ["robot_pedestal", "task_table", "bottle", "link0"]
#     print("\n--- Component Origins (xpos) ---")
#
#     for name in body_names:
#         # Get the ID of the body by its name
#         body_id = model.body(name).id
#
#         # Access the position (x, y, z) and orientation (rotation matrix)
#         position = data.xpos[body_id]
#         orientation_matrix = data.xmat[body_id].reshape(3, 3)
#
#         print(f"Body: '{name}' (ID: {body_id})")
#         print(f"  Position (x, y, z): {position}")
#         print(f"  Orientation Matrix (3x3):\n{orientation_matrix}\n")
#
#     mujoco.viewer.launch(model, data)
#
# except Exception as e:
#     print(f"Error loading model: {e}")
#### use for simple test ######

#### for complete test use this ######
try:
    model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
    data = mujoco.MjData(model)
    print("Model loaded successfully!")

    # Reset to home position
    home_key_id = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    mujoco.mj_forward(model, data)
    print("Robot set to 'home' keyframe position\n")

    # ==================== JOINT 7 (WRIST) ====================
    print("=" * 60)
    print("JOINT 7 (WRIST) ANALYSIS")
    print("=" * 60)

    joint7_id = model.joint("joint7").id
    joint7_qpos_idx = model.jnt_qposadr[joint7_id]
    joint7_value = data.qpos[joint7_qpos_idx]
    print(f"Joint 7 current value: {joint7_value:.4f} rad ({np.degrees(joint7_value):.2f}°)")

    # ==================== LINK 7 FRAME ====================
    print("\n" + "=" * 60)
    print("LINK 7 FRAME")
    print("=" * 60)

    link7_id = model.body("link7").id
    link7_pos = data.xpos[link7_id]
    link7_mat = data.xmat[link7_id].reshape(3, 3)

    print(f"Link7 Position: {link7_pos}")
    print(f"Link7 Rotation Matrix:\n{link7_mat}")
    print(f"\nLink7 X-axis (world frame): {link7_mat[:, 0]}")
    print(f"Link7 Y-axis (world frame): {link7_mat[:, 1]}")
    print(f"Link7 Z-axis (world frame): {link7_mat[:, 2]}")

    # ==================== HAND (END-EFFECTOR) FRAME ====================
    print("\n" + "=" * 60)
    print("HAND (END-EFFECTOR) FRAME")
    print("=" * 60)

    hand_id = model.body("hand").id
    hand_pos = data.xpos[hand_id]
    hand_mat = data.xmat[hand_id].reshape(3, 3)

    print(f"Hand Position: {hand_pos}")
    print(f"Hand Rotation Matrix:\n{hand_mat}")
    print(f"\nHand X-axis (world frame): {hand_mat[:, 0]}")
    print(f"Hand Y-axis (world frame): {hand_mat[:, 1]}")
    print(f"Hand Z-axis (world frame): {hand_mat[:, 2]}")

    # ==================== GRIPPER CENTER SITE ====================
    print("\n" + "=" * 60)
    print("GRIPPER CENTER SITE")
    print("=" * 60)

    gripper_site_id = model.site("gripper_center").id
    gripper_pos = data.site_xpos[gripper_site_id]
    gripper_mat = data.site_xmat[gripper_site_id].reshape(3, 3)

    print(f"Gripper Center Position: {gripper_pos}")
    print(f"Gripper Center Rotation Matrix:\n{gripper_mat}")
    print(f"\nGripper +X axis (world frame): {gripper_mat[:, 0]}")
    print(f"Gripper +Y axis (world frame): {gripper_mat[:, 1]}")
    print(f"Gripper +Z axis (world frame): {gripper_mat[:, 2]}")

    # ==================== PUSH DIRECTION ANALYSIS ====================
    print("\n" + "=" * 60)
    print("PUSH DIRECTION ANALYSIS")
    print("=" * 60)

    # Gripper +X axis in world frame (this is the push direction)
    gripper_x_world = gripper_mat[:, 0]

    # Calculate angle of gripper +X in XY plane
    angle_rad = np.arctan2(gripper_x_world[1], gripper_x_world[0])
    angle_deg = np.degrees(angle_rad)

    print(
        f"Gripper +X axis (push direction): [{gripper_x_world[0]:.4f}, {gripper_x_world[1]:.4f}, {gripper_x_world[2]:.4f}]")
    print(f"Angle in XY plane: {angle_rad:.4f} rad ({angle_deg:.2f}°)")

    # What this means for trajectory following
    print("\n" + "-" * 40)
    print("INTERPRETATION:")
    print("-" * 40)

    if abs(gripper_x_world[0]) < 0.1 and gripper_x_world[1] < -0.5:
        print("Gripper +X points in -Y direction (towards goal for straight trajectory)")
        print(f"default_angle should be: -π/2 = -90°")
    elif gripper_x_world[0] > 0.5 and abs(gripper_x_world[1]) < 0.1:
        print("Gripper +X points in +X direction")
        print(f"default_angle should be: 0°")
    elif gripper_x_world[0] < -0.5 and abs(gripper_x_world[1]) < 0.1:
        print("Gripper +X points in -X direction")
        print(f"default_angle should be: π = 180°")
    elif abs(gripper_x_world[0]) < 0.1 and gripper_x_world[1] > 0.5:
        print("Gripper +X points in +Y direction")
        print(f"default_angle should be: π/2 = 90°")
    else:
        print(f"Gripper +X points at angle: {angle_deg:.2f}°")
        print(f"default_angle should be: {angle_rad:.4f} rad")

    # ==================== BOTTLE POSITION ====================
    print("\n" + "=" * 60)
    print("BOTTLE POSITION")
    print("=" * 60)

    bottle_id = model.body("bottle").id
    bottle_pos = data.xpos[bottle_id]
    print(f"Bottle Position: {bottle_pos}")

    # Vector from gripper to bottle
    gripper_to_bottle = bottle_pos[:2] - gripper_pos[:2]
    dist = np.linalg.norm(gripper_to_bottle)
    direction = gripper_to_bottle / (dist + 1e-6)
    angle_to_bottle = np.arctan2(direction[1], direction[0])

    print(f"Gripper to Bottle vector (XY): {gripper_to_bottle}")
    print(f"Distance to bottle: {dist:.4f}m")
    print(f"Angle to bottle: {np.degrees(angle_to_bottle):.2f}°")

    # ==================== GOAL POSITION ====================
    print("\n" + "=" * 60)
    print("GOAL POSITION")
    print("=" * 60)

    goal_site_id = model.site("goal").id
    goal_pos = data.site_xpos[goal_site_id]
    print(f"Goal Position: {goal_pos}")

    # Launch viewer
    print("\n" + "=" * 60)
    print("Launching viewer...")
    print("=" * 60)
    mujoco.viewer.launch(model, data)

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
#### for complete test use this ######