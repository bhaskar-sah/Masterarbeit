import numpy as np


class ObservationBuilder:
    """
    Assembles the observation vector for the Panda push environment.

    Observation (39-dim):
        qpos (7), qvel (7), hand_pos (3), bottle_pos (3),
        dir_to_bottle (2), dist_to_bottle (1),
        push_dir (2), dist_to_target (1),
        deviation_vec (2), deviation_mag (1),
        angle_error (1), progress (1),
        contact_force (3), is_touching (1),
        K_normalized (2), settling_flag (1), wrist_normalized (1)
    """

    def __init__(self, data, traj_manager, push_ctrl, contact_manager,
                 hand_body_id, bottle_body_id,
                 K_min, K_max, max_wrist_rotation):
        self.data = data
        self.traj_manager = traj_manager
        self.push_ctrl = push_ctrl
        self.contact_manager = contact_manager

        self.hand_body_id = hand_body_id
        self.bottle_body_id = bottle_body_id

        self.K_min = K_min
        self.K_max = K_max
        self.max_wrist_rotation = max_wrist_rotation

    def get_obs(self, current_K, wrist_offset):
        qpos = self.data.qpos[7:14].astype(np.float32)
        qvel = self.data.qvel[6:13].astype(np.float32)
        hand_pos = self.data.xpos[self.hand_body_id].astype(np.float32)
        bottle_pos = self.data.xpos[self.bottle_body_id].astype(np.float32)
        bottle_xy = bottle_pos[:2]

        # Direction to bottle
        hand_to_bottle = bottle_pos[:2] - hand_pos[:2]
        dist_to_bottle = np.linalg.norm(hand_to_bottle)
        dir_to_bottle = hand_to_bottle / (dist_to_bottle + 1e-6)

        # Push direction (to lookahead target)
        push_dir, dist_to_target, _ = self.push_ctrl.get_push_direction(bottle_xy)

        # Deviation from trajectory
        deviation_vec, deviation_mag = self.traj_manager._get_path_deviation(bottle_xy)

        # Angle error between force and push direction
        angle_error = self.push_ctrl.get_angle_error(bottle_xy)

        # Progress
        progress = self.traj_manager._get_progress(bottle_xy)

        # Contact
        contact_force = self.contact_manager.get_contact_force()
        is_touching = np.array([1.0 if self.contact_manager.is_touching() else 0.0], dtype=np.float32)

        # Stiffness / wrist
        K_normalized = (current_K - self.K_min) / (self.K_max - self.K_min)
        settling_flag = np.array([1.0 if self.push_ctrl.is_settling else 0.0], dtype=np.float32)
        wrist_normalized = np.array([wrist_offset / self.max_wrist_rotation], dtype=np.float32)

        obs = np.concatenate([
            qpos,             # 7
            qvel,             # 7
            hand_pos,         # 3
            bottle_pos,       # 3
            dir_to_bottle,    # 2
            [dist_to_bottle], # 1
            push_dir,         # 2
            [dist_to_target], # 1
            deviation_vec,    # 2
            [deviation_mag],  # 1
            [angle_error],    # 1
            [progress],       # 1
            contact_force,    # 3
            is_touching,      # 1
            K_normalized,     # 2
            settling_flag,    # 1
            wrist_normalized, # 1
        ])  # Total: 39

        return obs.astype(np.float32)
