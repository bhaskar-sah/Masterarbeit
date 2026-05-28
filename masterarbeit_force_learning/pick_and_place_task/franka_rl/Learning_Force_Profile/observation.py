"""
observation.py

CHANGES from previous version:
  1. contact_force now stored as f_robot_on_bottle (sign-flipped from raw
     mj_contactForce). This matches the convention used in reward.py and
     gives the agent a cleaner physical interpretation: positive components
     mean "I'm pushing the bottle in that direction."

  2. Added cos_force_alignment (1D): cos(angle) between measured horizontal
     force-on-bottle and push_dir. Direct signal for spec point 7
     (the angle the agent must reduce when correcting deviation).

  3. Added future_push_dir (2D): push direction at a point 8 indices ahead
     of the lookahead point. Lets the agent anticipate upcoming tangent
     changes for curved/s-curve trajectories. For straight trajectories
     this equals current push_dir (no information lost).

obs_dim: 45 -> 48
"""

import numpy as np


# How many additional indices ahead to peek for future_push_dir
FUTURE_LOOKAHEAD = 8


class ObservationBuilder:
    def __init__(self, model, data, config,
                 hand_body_id, bottle_body_id,
                 traj_manager, contact_manager, push_controller):
        self.model = model
        self.data = data
        self.config = config
        self.hand_body_id = hand_body_id
        self.bottle_body_id = bottle_body_id
        self.traj_manager = traj_manager
        self.contact_manager = contact_manager
        self.push_controller = push_controller

        self.qpos_start = 7
        self.qvel_start = 6

    def _get_future_push_dir(self, bottle_xy):
        """
        Get push direction at a point further ahead on the trajectory.
        Used so the agent can anticipate upcoming tangent changes.
        Falls back to current push_dir near the end of trajectory.
        """
        traj = self.traj_manager.trajectory
        if traj is None or len(traj) < 2:
            push_dir, _, _ = self.traj_manager.get_push_direction(bottle_xy)
            return push_dir

        # Find current closest point and look further ahead
        idx, _, _ = self.traj_manager.get_closest_point_on_trajectory(bottle_xy)
        future_idx = min(
            idx + self.traj_manager.lookahead_points + FUTURE_LOOKAHEAD,
            len(traj) - 1,
        )
        # Use the future trajectory point as a "virtual bottle position"
        # to query push_dir from there
        future_pos = traj[future_idx]
        push_dir, _, _ = self.traj_manager.get_push_direction(future_pos)
        return push_dir

    def get_observation(self):
        # Robot state
        qpos = self.data.qpos[self.qpos_start:self.qpos_start + 7].astype(np.float32)
        qvel = self.data.qvel[self.qvel_start:self.qvel_start + 7].astype(np.float32)

        # End-effector state
        hand_pos = self.data.xpos[self.hand_body_id].astype(np.float32)
        hand_vel = self.push_controller.get_ee_velocity().astype(np.float32)

        # Bottle state
        bottle_pos = self.data.xpos[self.bottle_body_id].astype(np.float32)
        bottle_xy = bottle_pos[:2]

        # Hand-to-bottle
        hand_to_bottle = bottle_pos[:2] - hand_pos[:2]
        dist_to_bottle = np.linalg.norm(hand_to_bottle)
        dir_to_bottle = hand_to_bottle / (dist_to_bottle + 1e-6)

        # Trajectory info
        push_dir, dist_to_target, _ = self.traj_manager.get_push_direction(bottle_xy)
        deviation_vec, deviation_mag = self.traj_manager.get_path_deviation(bottle_xy)
        progress = self.traj_manager.get_progress(bottle_xy)

        # ---- contact force, sign-corrected ----
        # MuJoCo gives bottle-on-robot reaction; flip to robot-on-bottle
        # so positive components mean "robot is pushing bottle in that direction"
        # raw_force = self.contact_manager.get_contact_force()
        # f_robot_on_bottle = -raw_force
        f_robot_on_bottle = self.contact_manager.get_contact_force()
        is_touching_bool = self.contact_manager.is_touching()
        is_touching = np.array(
            [1.0 if is_touching_bool else 0.0],
            dtype=np.float32,
        )

        # Bottle tilt
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        bottle_tilt = np.array([bottle_mat[2, 2]], dtype=np.float32)

        # End-effector x-axis
        ee_mat = self.data.xmat[self.hand_body_id].reshape(3, 3)
        ee_x_axis = ee_mat[:, 0].astype(np.float32)

        # Flange pushing axis in world XY
        flange_axis_3d = self.push_controller.get_flange_axis_world()
        flange_axis_xy = flange_axis_3d[:2].astype(np.float32)

        # cos(angle) between flange axis and push direction
        cos_alignment = float(np.dot(flange_axis_xy, push_dir))
        cos_alignment_arr = np.array([cos_alignment], dtype=np.float32)

        # side_dot: dot(bottle->flange, push_dir); -1 = perfectly behind bottle
        b2f = hand_pos[:2] - bottle_xy
        b2f_norm = np.linalg.norm(b2f)
        if b2f_norm > 1e-6:
            b2f_unit = b2f / b2f_norm
            side_dot = float(np.dot(b2f_unit, push_dir))
        else:
            side_dot = 0.0
        side_dot_arr = np.array([side_dot], dtype=np.float32)

        # ---- NEW: cos(angle) between measured force and push_dir ----
        # Direct signal for the angle θ in your spec point 7.
        # 1 = force perfectly along push_dir, 0 = perpendicular, -1 = opposite.
        f_xy = f_robot_on_bottle[:2]
        f_xy_mag = np.linalg.norm(f_xy)
        if f_xy_mag > 0.1:
            cos_force_alignment = float(np.dot(f_xy / f_xy_mag, push_dir))
        else:
            cos_force_alignment = 0.0
        cos_force_alignment_arr = np.array([cos_force_alignment], dtype=np.float32)

        # ---- NEW: future push direction (anticipatory) ----
        future_push_dir = self._get_future_push_dir(bottle_xy).astype(np.float32)

        # =====================================================================
        # FRAME TRANSFORMATION FOR RL AGENT (EGO-CENTRIC)
        # =====================================================================
        # The agent acts in the local End-Effector frame. We must rotate the 
        # world-frame trajectory vectors into the EE frame so the agent knows
        # where the trajectory is relative to its current gripper orientation.
        
        R_world_to_ee = ee_mat.T 

        # 1. Transform Push Direction to EE Frame
        push_dir_3d = np.array([push_dir[0], push_dir[1], 0.0])
        push_dir_ee = (R_world_to_ee @ push_dir_3d)[:2].astype(np.float32)

        # 2. Transform Deviation Vector to EE Frame
        dev_vec_3d = np.array([deviation_vec[0], deviation_vec[1], 0.0])
        deviation_vec_ee = (R_world_to_ee @ dev_vec_3d)[:2].astype(np.float32)

        # 3. Transform Future Push Direction to EE Frame
        future_push_dir_3d = np.array([future_push_dir[0], future_push_dir[1], 0.0])
        future_push_dir_ee = (R_world_to_ee @ future_push_dir_3d)[:2].astype(np.float32)

        # Build observation
        obs = np.concatenate([
            qpos,                                            # 7
            qvel,                                            # 7
            hand_pos,                                        # 3
            hand_vel,                                        # 3
            bottle_pos,                                      # 3
            dir_to_bottle.astype(np.float32),                # 2
            np.array([dist_to_bottle], dtype=np.float32),    # 1
            push_dir_ee,                                     # 2
            np.array([dist_to_target], dtype=np.float32),    # 1
            deviation_vec_ee,                                # 2
            np.array([deviation_mag], dtype=np.float32),     # 1
            np.array([progress], dtype=np.float32),          # 1
            f_robot_on_bottle.astype(np.float32),            # 3  (sign-corrected)
            is_touching,                                     # 1
            bottle_tilt,                                     # 1
            ee_x_axis,                                       # 3
            flange_axis_xy,                                  # 2
            cos_alignment_arr,                               # 1
            side_dot_arr,                                    # 1
            cos_force_alignment_arr,                         # 1  (NEW)
            future_push_dir_ee,                              # 2  (NEW)
        ])  # Total: 48

        return obs.astype(np.float32)


# ==================== OBSERVATION INDEX REFERENCE ====================
"""
Index   | Dim | Name                  | Description
--------|-----|-----------------------|----------------------------------
0-6     | 7   | qpos                  | Robot joint positions
7-13    | 7   | qvel                  | Robot joint velocities
14-16   | 3   | hand_pos              | EE position [x,y,z]
17-19   | 3   | hand_vel              | EE velocity [vx,vy,vz]
20-22   | 3   | bottle_pos            | Bottle position [x,y,z]
23-24   | 2   | dir_to_bottle         | Unit vector to bottle [x,y]
25      | 1   | dist_to_bottle        | Distance to bottle
26-27   | 2   | push_dir              | Desired push direction [x,y]
28      | 1   | dist_to_target        | Distance to lookahead target
29-30   | 2   | deviation_vec         | Bottle->trajectory vector [x,y]
31      | 1   | deviation_mag         | |deviation|
32      | 1   | progress              | Progress in [0,1]
33-35   | 3   | f_robot_on_bottle     | Force ROBOT applies to bottle (sign-flipped)
36      | 1   | is_touching           | Binary contact flag
37      | 1   | bottle_tilt           | Bottle z-component (1=upright)
38-40   | 3   | ee_x_axis             | EE local x-axis in world
41-42   | 2   | flange_axis_xy        | Flange push axis in world XY
43      | 1   | cos_alignment         | cos(flange_axis, push_dir)
44      | 1   | side_dot              | dot(bottle->flange, push_dir); -1=behind
45      | 1   | cos_force_alignment   | cos(force, push_dir)        [NEW]
46-47   | 2   | future_push_dir_ee       | Push dir 8 points further   [NEW]
TOTAL: 48
"""