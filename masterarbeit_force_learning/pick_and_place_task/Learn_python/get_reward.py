import numpy as np

def _get_reward(self):
    gripper_pos = self.data.site(self.gripper_site_id).xpos
    bottle_pos = self.data.body(self.bottle_body_id).xpos
    dist_gripper_to_bottle = np.linalg.norm(gripper_pos - bottle_pos)

    reward = -dist_gripper_to_bottle
    if dist_gripper_to_bottle < 0.03:  # within 3cm
        reward += 100
    return reward

