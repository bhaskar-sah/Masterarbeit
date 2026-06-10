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
 
        idx, _, _ = self.traj_manager.get_closest_point_on_trajectory(bottle_xy)
        future_idx = min(
            idx + self.traj_manager.lookahead_points + FUTURE_LOOKAHEAD,
            len(traj) - 1,
        )
        future_pos = traj[future_idx]
        push_dir, _, _ = self.traj_manager.get_push_direction(future_pos)
        return push_dir

    def get_observation(self):
        pc = self.push_controller
        w_R_b = pc.get_base_orientation()
        base_pos_world = pc.get_base_pos()

        # Robot state
        qpos = self.data.qpos[self.qpos_start:self.qpos_start + 7].astype(np.float32)
        qvel = self.data.qvel[self.qvel_start:self.qvel_start + 7].astype(np.float32)

        # End-effector in B
        ee_pos = pc.get_ee_position().astype(np.float32)        # {B}
        ee_vel = pc.get_ee_velocity().astype(np.float32)        # {B}

        w_R_ee = self.data.site_xmat[pc.gripper_site_id].reshape(3, 3)
        b_R_ee = w_R_b.T @ w_R_ee
        ee_x_axis = b_R_ee[:, 0].astype(np.float32)             # {B}

        # ---------- bottle in {B} ----------
        bottle_pos_world = self.data.xpos[self.bottle_body_id].copy()
        bottle_xy_world = bottle_pos_world[:2]
        bottle_pos = (w_R_b.T @ (bottle_pos_world - base_pos_world)).astype(np.float32)
        bottle_xy = bottle_pos[:2]
 
        # EE -> bottle (both in {B}, both the SITE point)
        ee_to_bottle = bottle_xy - ee_pos[:2]
        dist_to_bottle = float(np.linalg.norm(ee_to_bottle))
        dir_to_bottle = (ee_to_bottle / (dist_to_bottle + 1e-6)).astype(np.float32)
 
        # # ---------- path frame {P} (shared with the controller's action) ----------
        # R_path, t_hat, b_hat, n_hat, _ = pc.compute_path_frame_base(bottle_xy_world)

 
        # frame-invariant scalar trajectory quantities
        push_dir_world, dist_to_target, _ = self.traj_manager.get_push_direction(bottle_xy_world)
        deviation_vec_world, deviation_mag = self.traj_manager.get_path_deviation(bottle_xy_world)
        progress = self.traj_manager.get_progress(bottle_xy_world)

        # ---------- path frame {P}, built inline ----------
        # ... reconstructs the SAME frame compute_torque builds (same b_hat = n_hat x t_hat)
        t_hat = w_R_b.T @ np.array([push_dir_world[0], push_dir_world[1], 0.0])
        n_hat = w_R_b.T @ np.array([0.0, 0.0, 1.0])
        nt = np.linalg.norm(t_hat)
        t_hat = t_hat / nt if nt > 1e-6 else np.array([1.0, 0.0, 0.0])
        n_hat = n_hat / np.linalg.norm(n_hat)
        b_hat = np.cross(n_hat, t_hat)
        b_hat /= np.linalg.norm(b_hat)
        R_path = np.column_stack((t_hat, b_hat, n_hat))
 
        # ---- push direction as a HEADING, in {B} ----
        push_dir_b = (w_R_b.T @ np.array([push_dir_world[0], push_dir_world[1], 0.0]))[:2].astype(np.float32)
 
        # ---- future push direction as a HEADING, in {B} ----
        fut_world = self._get_future_push_dir(bottle_xy_world)
        future_push_dir_b = (w_R_b.T @ np.array([fut_world[0], fut_world[1], 0.0]))[:2].astype(np.float32)
 
        # ---- deviation as TRACKING ERROR, in {P} = [along-track, cross-track] ----
        dev_b = w_R_b.T @ np.array([deviation_vec_world[0], deviation_vec_world[1], 0.0])
        deviation_vec_path = (R_path.T @ dev_b)[:2].astype(np.float32)
 
        # ---------- contact force, robot->bottle, rotated into {B} ----------
        f_world = self.contact_manager.get_contact_force()
        f_robot_on_bottle = (w_R_b.T @ f_world).astype(np.float32)
        is_touching_bool = self.contact_manager.is_touching()
        is_touching = np.array([1.0 if is_touching_bool else 0.0], dtype=np.float32)
 
        # ---------- bottle tilt (frame-invariant scalar) ----------
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        bottle_tilt = np.array([bottle_mat[2, 2]], dtype=np.float32)
 
        # ---------- gripper push axis in {B} XY ----------
        flange_axis_b = pc.get_flange_axis_world()
        flange_axis_xy = flange_axis_b[:2].astype(np.float32)
 
        # ---------- direction cosines (frame-invariant; computed in {B}) ----------
        # cos(flange axis, push heading): yaw alignment cue
        cos_alignment = float(np.dot(flange_axis_xy, push_dir_b))
        cos_alignment_arr = np.array([cos_alignment], dtype=np.float32)
 
        # side_dot: dot(bottle->flange, push heading); -1 == perfectly behind bottle
        b2f = ee_pos[:2] - bottle_xy
        b2f_norm = np.linalg.norm(b2f)
        if b2f_norm > 1e-6:
            side_dot = float(np.dot(b2f / b2f_norm, push_dir_b))
        else:
            side_dot = 0.0
        side_dot_arr = np.array([side_dot], dtype=np.float32)
 
        # cos(measured force, push heading): the angle the agent reduces when
        # correcting deviation (1 = force along push, 0 = perpendicular)
        f_xy = f_robot_on_bottle[:2]
        f_xy_mag = np.linalg.norm(f_xy)
        if f_xy_mag > 0.1:
            cos_force_alignment = float(np.dot(f_xy / f_xy_mag, push_dir_b))
        else:
            cos_force_alignment = 0.0
        cos_force_alignment_arr = np.array([cos_force_alignment], dtype=np.float32)
 
        # ---------- assemble (order/dims identical to previous 48-D layout) ----------
        obs = np.concatenate([
            qpos,                                            # 7
            qvel,                                            # 7
            ee_pos,                                          # 3   {B}, site
            ee_vel,                                          # 3   {B}, site
            bottle_pos,                                      # 3   {B}
            dir_to_bottle,                                   # 2   {B}
            np.array([dist_to_bottle], dtype=np.float32),    # 1
            push_dir_b,                                      # 2   {B} heading
            np.array([dist_to_target], dtype=np.float32),    # 1
            deviation_vec_path,                              # 2   {P} [along, cross]
            np.array([deviation_mag], dtype=np.float32),     # 1
            np.array([progress], dtype=np.float32),          # 1
            f_robot_on_bottle,                               # 3   {B}
            is_touching,                                     # 1
            bottle_tilt,                                     # 1
            ee_x_axis,                                       # 3   {B}
            flange_axis_xy,                                  # 2   {B}
            cos_alignment_arr,                               # 1
            side_dot_arr,                                    # 1
            cos_force_alignment_arr,                         # 1
            future_push_dir_b,                               # 2   {B} heading
        ])  # Total: 48
 
        return obs.astype(np.float32)
 
 
# ==================== OBSERVATION INDEX REFERENCE ====================
"""
Index   | Dim | Name                | Frame | Description
--------|-----|---------------------|-------|----------------------------------
0-6     | 7   | qpos                |  -    | Robot joint positions
7-13    | 7   | qvel                |  -    | Robot joint velocities
14-16   | 3   | ee_pos              | {B}   | EE (gripper_center site) position
17-19   | 3   | ee_vel              | {B}   | EE (site) linear velocity
20-22   | 3   | bottle_pos          | {B}   | Bottle position
23-24   | 2   | dir_to_bottle       | {B}   | Unit vector EE->bottle (xy)
25      | 1   | dist_to_bottle      |  -    | Distance EE->bottle (invariant)
26-27   | 2   | push_dir            | {B}   | Desired push HEADING (xy)
28      | 1   | dist_to_target      |  -    | Distance to lookahead target
29-30   | 2   | deviation_vec       | {P}   | Tracking error [along t_hat, cross b_hat]
31      | 1   | deviation_mag       |  -    | |deviation|
32      | 1   | progress            |  -    | Progress in [0,1]
33-35   | 3   | f_robot_on_bottle   | {B}   | Force ROBOT applies to bottle
36      | 1   | is_touching         |  -    | Binary contact flag
37      | 1   | bottle_tilt         |  -    | Bottle z-component (1=upright)
38-40   | 3   | ee_x_axis           | {B}   | EE local x-axis
41-42   | 2   | flange_axis_xy      | {B}   | Flange push axis (xy)
43      | 1   | cos_alignment       |  -    | cos(flange_axis, push_dir)
44      | 1   | side_dot            |  -    | dot(bottle->flange, push_dir); -1=behind
45      | 1   | cos_force_alignment |  -    | cos(force, push_dir)
46-47   | 2   | future_push_dir     | {B}   | Push HEADING ~8 points further ahead
TOTAL: 48
 
Headings (push_dir, future_push_dir, gripper axes) share {B} so the agent
reasons about orientation/yaw in one frame; the tracking error is in {P} so it
lines up with the force action subspaces (action[0]~t_hat, action[1]~b_hat).
"""