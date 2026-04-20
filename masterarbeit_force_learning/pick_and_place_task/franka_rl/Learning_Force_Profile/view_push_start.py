# import os
# import mujoco
# import mujoco.viewer

# current_dir = os.path.dirname(os.path.realpath(__file__))
# SCENE_XML_PATH = os.path.join(current_dir, "robot_panda_push_force.xml")

# model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
# data = mujoco.MjData(model)

# key_id = model.key("push_start").id
# mujoco.mj_resetDataKeyframe(model, data, key_id)
# mujoco.mj_forward(model, data)

# # Static visualization - no physics, just display
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         # Don't call mj_step() - robot stays frozen
#         viewer.sync()


import os
import mujoco
import mujoco.viewer
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
SCENE_XML_PATH = os.path.join(current_dir, "robot_panda_push_force.xml")

model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
data = mujoco.MjData(model)

key_id = model.key("push_start").id
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

# ============ DEBUG: Check distances ============
hand_site_id = model.site("hand_center").id
hand_site_pos = data.site_xpos[hand_site_id]
bottle_pos = data.xpos[model.body("bottle").id]

print("=" * 50)
print(f"Hand site position:   ({hand_site_pos[0]:.4f}, {hand_site_pos[1]:.4f}, {hand_site_pos[2]:.4f})")
print(f"Bottle position:      ({bottle_pos[0]:.4f}, {bottle_pos[1]:.4f}, {bottle_pos[2]:.4f})")

xy_dist = np.linalg.norm(hand_site_pos[:2] - bottle_pos[:2])
print(f"XY distance (hand to bottle center): {xy_dist*100:.2f} cm")
#print(f"Gap to bottle surface: {(xy_dist - 0.039)*100:.2f} cm")
print("=" * 50)
# ================================================

# Static visualization - no physics, just display
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()