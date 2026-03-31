"""
Panda Push Environment - More RL-Based Version with Learned Wrist Rotation

Main idea:
    - Keep a simple nominal pushing prior
    - RL learns:
        1) forward modulation
        2) lateral correction
        3) wrist rotation
        4) Kx
        5) Ky
    - lookahead information is added for better curved-trajectory behavior
    - reward is smooth and interpretable

Action:
    [forward_mod, lateral_mod, wrist_mod, Kx, Ky]

Compared to the previous version:
    - wrist is now learned by RL
    - automatic wrist alignment during push is removed
    - lookahead direction is added to observation
    - desired push direction is added to observation
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os


class PandaPushTrajectoryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_type="straight"):
        super().__init__()

        # ==================== Load model ====================
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "robot_panda_push_force.xml")
        xml_path = os.path.abspath(xml_path)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find XML file at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ==================== IDs ====================
        self.home_key_id = self.model.key("home").id
        self.bottle_body_id = self.model.body("bottle").id
        self.hand_body_id = self.model.body("hand").id
        self.goal_site_id = self.model.site("goal").id
        self.left_finger_body_id = self.model.body("left_finger").id
        self.right_finger_body_id = self.model.body("right_finger").id

        self.robot_contact_bodies = {
            self.hand_body_id,
            self.left_finger_body_id,
            self.right_finger_body_id
        }

        # ==================== Control ====================
        self.actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = self.actuator_ranges[:, 0]
        self.act_high = self.actuator_ranges[:, 1]
        self.current_qpos_target = np.zeros(7)

        # ==================== Push parameters ====================
        self.base_forward_speed = 0.015
        self.behind_distance = 0.04

        # RL-controlled correction scales
        self.lateral_push_scale = 0.45
        self.lateral_hand_offset_scale = 0.04

        # ==================== Wrist rotation ====================
        self.gripper_push_angle_at_home = -np.pi / 2
        self.max_wrist_rotation = 2.5
        self.wrist_offset = 0.0
        self.base_wrist_pos = 0.0

        # ==================== Stiffness ====================
        self.K_min = 100.0
        self.K_max = 500.0
        self.current_K = np.array([300.0, 300.0], dtype=np.float32)

        # ==================== Tilt safety ====================
        self.tilt_ok = 0.99
        self.tilt_slow = 0.98
        self.tilt_stop = 0.96

        self.is_settling = False
        self.settle_counter = 0
        self.settle_required = 25

        # ==================== Cartesian control ====================
        self.target_z = 0.92
        self.z_gain = 10.0
        self.damping = 0.01

        # ==================== Trajectory ====================
        self.trajectory_type = trajectory_type
        self.trajectory = None
        self.total_arc_length = 0.0
        self.path_tolerance = 0.05
        self.goal_position = np.array([0.4, -0.2], dtype=np.float32)

        # Phase
        self.in_approach = True

        # ==================== Action space ====================
        # [forward_mod, lateral_mod, wrist_mod, Kx, Ky]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ==================== Observation space ====================
        # qpos(7), qvel(7), hand_pos(3), bottle_pos(3), dir_to_bottle(2), dist_to_bottle(1),
        # tangent_dir(2), deviation_vec(2), deviation_mag(1), progress(1),
        # contact_force(3), is_touching(1), K_norm(2), settling(1), wrist_norm(1),
        # tilt(1), angle_error(1), dist_to_goal(1), lookahead_dir(2), lookahead_dist(1),
        # desired_push_dir(2)
        obs_dim = 45
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ==================== State ====================
        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0
        self.max_episode_length = 2500
        self.prev_progress = 0.0

        self.last_desired_push_dir = np.array([0.0, -1.0], dtype=np.float32)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # ==================== Logging ====================
        self.force_profile_log = []
        self.stiffness_profile_log = []
        self.deviation_log = []
        self.angle_error_log = []

        print(f"\n{'=' * 60}")
        print("MORE RL-BASED PUSHING ENVIRONMENT")
        print(f"{'=' * 60}")
        print("Nominal prior: push along path tangent + lookahead")
        print("RL controls: forward speed, lateral correction, wrist, Kx, Ky")
        print("Reward is smooth and interpretable")
        print(f"Goal position: {self.goal_position}")
        print(f"{'=' * 60}")

    # ==================================================================
    #                      TRAJECTORY HELPERS
    # ==================================================================

    def _generate_trajectory(self, bottle_start_xy, traj_type):
        """Generate trajectory from bottle start position to fixed goal."""
        start = bottle_start_xy.copy()
        n_points = 50
        goal = self.goal_position.copy()

        if traj_type == "straight":
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, goal)
            return traj.astype(np.float32)

        elif traj_type == "curved":
            t = np.linspace(0, 1, n_points)
            mid_point = np.array([0.55, 0.0], dtype=np.float32)
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid_point[0] + t ** 2 * goal[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid_point[1] + t ** 2 * goal[1]
            return np.stack([x, y], axis=1).astype(np.float32)

        elif traj_type == "s_curve":
            t = np.linspace(0, 1, n_points)
            wiggle_strength = np.sin(np.pi * t)
            x = start[0] + 0.08 * np.sin(2 * np.pi * t) * wiggle_strength
            y = start[1] + t * (goal[1] - start[1])
            return np.stack([x, y], axis=1).astype(np.float32)

        else:
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, goal)
            return traj.astype(np.float32)

    def _compute_arc_length(self):
        if self.trajectory is None or len(self.trajectory) < 2:
            self.total_arc_length = 0.0
            return
        diffs = np.diff(self.trajectory, axis=0)
        self.total_arc_length = np.sum(np.linalg.norm(diffs, axis=1))

    def _get_closest_point_on_trajectory(self, pos_xy):
        if self.trajectory is None:
            return 0, pos_xy.copy(), 0.0

        distances = np.linalg.norm(self.trajectory - pos_xy, axis=1)
        idx = np.argmin(distances)
        closest_pt = self.trajectory[idx].copy()
        arc_len = (
            np.sum(np.linalg.norm(np.diff(self.trajectory[:idx + 1], axis=0), axis=1))
            if idx > 0 else 0.0
        )
        return idx, closest_pt, arc_len

    def _get_path_deviation(self, bottle_xy):
        _, closest_pt, _ = self._get_closest_point_on_trajectory(bottle_xy)
        deviation_vec = closest_pt - bottle_xy
        return deviation_vec.astype(np.float32), float(np.linalg.norm(deviation_vec))

    def _get_path_tangent(self, bottle_xy):
        if self.trajectory is None or len(self.trajectory) < 2:
            return np.array([0.0, -1.0], dtype=np.float32)

        idx, _, _ = self._get_closest_point_on_trajectory(bottle_xy)
        if idx < len(self.trajectory) - 1:
            tangent = self.trajectory[idx + 1] - self.trajectory[idx]
        else:
            tangent = self.trajectory[idx] - self.trajectory[idx - 1]

        norm = np.linalg.norm(tangent)
        if norm > 1e-6:
            return (tangent / norm).astype(np.float32)
        return np.array([0.0, -1.0], dtype=np.float32)

    def _get_progress(self, bottle_xy):
        if self.total_arc_length < 1e-6:
            return 0.0
        _, _, arc_len = self._get_closest_point_on_trajectory(bottle_xy)
        return float(np.clip(arc_len / self.total_arc_length, 0.0, 1.0))

    def _get_lookahead_target(self, bottle_xy):
        """
        Returns a point ahead on trajectory.
        """
        if self.trajectory is None or len(self.trajectory) < 2:
            return self.goal_position.copy()

        idx, _, _ = self._get_closest_point_on_trajectory(bottle_xy)
        lookahead_steps = 5
        target_idx = min(idx + lookahead_steps, len(self.trajectory) - 1)
        return self.trajectory[target_idx].copy()

    def _get_lookahead_direction(self, bottle_xy):
        target = self._get_lookahead_target(bottle_xy)
        vec = target - bottle_xy
        dist = np.linalg.norm(vec)

        if dist > 1e-6:
            return (vec / dist).astype(np.float32), float(dist)
        return np.array([0.0, 0.0], dtype=np.float32), 0.0

    # ==================================================================
    #                      RL-DRIVEN GEOMETRY
    # ==================================================================

    def _get_rl_push_direction(self, bottle_xy, lateral_mod):
        """
        Improved nominal direction:
            - combines local tangent + lookahead direction
            - RL adds lateral correction around that nominal direction
        """
        tangent = self._get_path_tangent(bottle_xy)
        lookahead_dir, _ = self._get_lookahead_direction(bottle_xy)

        blend = 0.5
        nominal_dir = (1.0 - blend) * tangent + blend * lookahead_dir
        norm = np.linalg.norm(nominal_dir)

        if norm > 1e-6:
            nominal_dir = nominal_dir / norm
        else:
            nominal_dir = tangent

        lateral_dir = np.array([-nominal_dir[1], nominal_dir[0]], dtype=np.float32)

        push_dir = nominal_dir + lateral_mod * lateral_dir * 0.5

        norm = np.linalg.norm(push_dir)
        if norm > 1e-6:
            push_dir = push_dir / norm
        else:
            push_dir = nominal_dir

        return push_dir.astype(np.float32)

    def _get_hand_target_position(self, bottle_xy, lateral_mod):
        """
        Hand target:
            - behind bottle along path tangent
            - RL chooses lateral offset toward correction side
        """
        tangent_dir = self._get_path_tangent(bottle_xy)
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)

        hand_xy = bottle_xy - tangent_dir * self.behind_distance

        if deviation_mag > 1e-6:
            correction_dir = deviation_vec / deviation_mag
        else:
            correction_dir = np.zeros(2, dtype=np.float32)

        hand_xy = hand_xy + lateral_mod * self.lateral_hand_offset_scale * correction_dir

        return np.array([hand_xy[0], hand_xy[1], self.target_z], dtype=np.float32)

    def _get_angle_error(self):
        """
        Compute angle between measured contact force and desired push direction.
        """
        contact_force = self._get_contact_force()
        force_mag = np.linalg.norm(contact_force[:2])

        if force_mag > 0.5:
            force_dir = -contact_force[:2] / force_mag
            dot_product = np.clip(np.dot(force_dir, self.last_desired_push_dir), -1.0, 1.0)
            angle_error = np.arccos(dot_product)
        else:
            angle_error = 0.0

        return float(angle_error)

    # ==================================================================
    #                      CONTACT / ROBOT HELPERS
    # ==================================================================

    def _get_contact_force(self):
        total_force = np.zeros(3, dtype=np.float32)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            robot_touch = body1 in self.robot_contact_bodies or body2 in self.robot_contact_bodies
            bottle_touch = body1 == self.bottle_body_id or body2 == self.bottle_body_id

            if robot_touch and bottle_touch:
                c_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_force)
                frame = contact.frame.reshape(3, 3)
                force_world = frame.T @ c_force[:3]
                total_force += force_world.astype(np.float32)
        return total_force

    def _is_touching(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            robot_touch = body1 in self.robot_contact_bodies or body2 in self.robot_contact_bodies
            bottle_touch = body1 == self.bottle_body_id or body2 == self.bottle_body_id
            if robot_touch and bottle_touch:
                return True
        return False

    def _get_bottle_tilt(self):
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        return bottle_mat[2, 2]

    def _check_stable(self):
        tilt = self._get_bottle_tilt()
        angvel = np.linalg.norm(self.data.qvel[3:6])
        return tilt > 0.99 and angvel < 0.1

    def _cartesian_to_joint_velocity(self, cart_vel):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)

        J = jacp[:, 6:13]
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.damping ** 2 * np.eye(3))
        return J_pinv @ cart_vel

    # ==================================================================
    #                      VELOCITY COMPUTATION
    # ==================================================================

    def _compute_approach_velocity(self):
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        target_pos = self._get_hand_target_position(bottle_xy, lateral_mod=0.0)
        error = target_pos - hand_pos
        dist = np.linalg.norm(error[:2])

        v_desired = error * 2.0
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.04:
            v_desired[:2] = v_desired[:2] / v_mag * 0.04

        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])
        return v_desired.astype(np.float32), dist

    def _compute_push_velocity(self, action):
        """
        RL-based push controller:
            - RL controls forward speed, lateral correction, wrist, Kx, Ky
            - nominal prior uses tangent + lookahead
            - minimal safety / recovery logic remains
        """
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        tilt = self._get_bottle_tilt()
        is_touching = self._is_touching()

        forward_mod = float(action[0])
        lateral_mod = float(action[1])

        self.current_K[0] = self.K_min + action[3] * (self.K_max - self.K_min)
        self.current_K[1] = self.K_min + action[4] * (self.K_max - self.K_min)
        K_avg = np.mean(self.current_K)

        # ==================== Minimal tilt safety ====================
        if self.is_settling:
            if self._check_stable():
                self.settle_counter += 1
                if self.settle_counter >= self.settle_required:
                    self.is_settling = False
                    self.settle_counter = 0
            else:
                self.settle_counter = 0

            v_desired = np.zeros(3, dtype=np.float32)
            v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])
            return v_desired

        if tilt < self.tilt_stop:
            self.is_settling = True
            self.settle_counter = 0

        if tilt > self.tilt_ok:
            speed_mult = 1.0
        elif tilt > self.tilt_slow:
            speed_mult = 0.5
        else:
            speed_mult = 0.2

        v_desired = np.zeros(3, dtype=np.float32)

        # ==================== Simple contact recovery ====================
        if not is_touching:
            bottle_direction = bottle_xy - hand_pos[:2]
            bottle_dist = np.linalg.norm(bottle_direction)

            if bottle_dist > 0.01:
                bottle_dir_normalized = bottle_direction / bottle_dist
                recovery_speed = min(0.08, bottle_dist * 3.0)

                v_desired[0] = bottle_dir_normalized[0] * recovery_speed
                v_desired[1] = bottle_dir_normalized[1] * recovery_speed
                v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

                if self.episode_length % 50 == 0:
                    print(f"    CONTACT LOST! Recovery: dist={bottle_dist * 100:.1f}cm")

                return v_desired

        # ==================== RL-driven push ====================
        push_dir = self._get_rl_push_direction(bottle_xy, lateral_mod)
        self.last_desired_push_dir = push_dir.copy()

        base_speed = self.base_forward_speed * (1.0 + forward_mod * 0.3) * speed_mult
        k_factor = 0.8 + 0.4 * (K_avg / self.K_max)
        push_speed = base_speed * k_factor

        v_push = push_dir * push_speed

        # ==================== Simple tracking ====================
        target_hand_pos = self._get_hand_target_position(bottle_xy, lateral_mod)
        pos_error = target_hand_pos[:2] - hand_pos[:2]
        v_tracking = pos_error * 3.0
        v_tracking = np.clip(v_tracking, -0.03, 0.03)

        v_desired[0] = v_push[0] + v_tracking[0]
        v_desired[1] = v_push[1] + v_tracking[1]

        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.05:
            v_desired[:2] = v_desired[:2] / v_mag * 0.05

        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        angle_error = self._get_angle_error()
        self.angle_error_log.append(angle_error)

        if self.episode_length % 100 == 0:
            _, dev_mag = self._get_path_deviation(bottle_xy)
            print(
                f"    RL push: dir=[{push_dir[0]:.2f}, {push_dir[1]:.2f}], "
                f"lat_mod={lateral_mod:.2f}, dev={dev_mag * 100:.1f}cm, "
                f"θ={np.degrees(angle_error):.1f}°"
            )

        return v_desired.astype(np.float32)

    # ==================================================================
    #                      OBSERVATION
    # ==================================================================

    def _get_obs(self):
        qpos = self.data.qpos[7:14].astype(np.float32)
        qvel = self.data.qvel[6:13].astype(np.float32)
        hand_pos = self.data.xpos[self.hand_body_id].astype(np.float32)
        bottle_pos = self.data.xpos[self.bottle_body_id].astype(np.float32)
        bottle_xy = bottle_pos[:2]

        hand_to_bottle = bottle_pos[:2] - hand_pos[:2]
        dist_to_bottle = np.linalg.norm(hand_to_bottle)
        dir_to_bottle = hand_to_bottle / (dist_to_bottle + 1e-6)

        tangent_dir = self._get_path_tangent(bottle_xy)
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        progress = self._get_progress(bottle_xy)

        contact_force = self._get_contact_force()
        is_touching = np.array([1.0 if self._is_touching() else 0.0], dtype=np.float32)

        K_normalized = (self.current_K - self.K_min) / (self.K_max - self.K_min)
        settling_flag = np.array([1.0 if self.is_settling else 0.0], dtype=np.float32)
        wrist_normalized = np.array([self.wrist_offset / self.max_wrist_rotation], dtype=np.float32)
        tilt = np.array([self._get_bottle_tilt()], dtype=np.float32)
        angle_error = np.array([self._get_angle_error()], dtype=np.float32)
        dist_to_goal = np.array([np.linalg.norm(self.goal_position - bottle_xy)], dtype=np.float32)

        lookahead_dir, lookahead_dist = self._get_lookahead_direction(bottle_xy)
        desired_push_dir = self.last_desired_push_dir.astype(np.float32)

        obs = np.concatenate([
            qpos,                  # 7
            qvel,                  # 7
            hand_pos,              # 3
            bottle_pos,            # 3
            dir_to_bottle,         # 2
            [dist_to_bottle],      # 1
            tangent_dir,           # 2
            deviation_vec,         # 2
            [deviation_mag],       # 1
            [progress],            # 1
            contact_force,         # 3
            is_touching,           # 1
            K_normalized,          # 2
            settling_flag,         # 1
            wrist_normalized,      # 1
            tilt,                  # 1
            angle_error,           # 1
            dist_to_goal,          # 1
            lookahead_dir,         # 2
            [lookahead_dist],      # 1
            desired_push_dir       # 2
        ])

        return obs.astype(np.float32)

    # ==================================================================
    #                      REWARD
    # ==================================================================

    def _get_reward(self, action=None):
        """
        Smooth interpretable reward.

        During push:
            progress reward
            deviation penalty
            force-angle penalty
            tilt penalty
            contact reward
            wrist alignment reward
            wrist smoothness penalty
            time penalty

        Terminal success / failure remain discrete.
        """
        info = {"is_success": False}

        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        hand_pos = self.data.xpos[self.hand_body_id]

        _, deviation_mag = self._get_path_deviation(bottle_xy)
        progress = self._get_progress(bottle_xy)
        tilt = self._get_bottle_tilt()
        is_touching = self._is_touching()
        force = self._get_contact_force()
        force_mag = np.linalg.norm(force)
        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)
        angle_error = self._get_angle_error()

        if self.in_approach:
            total_reward = -2.0 * dist_to_bottle - 5.0 * abs(hand_pos[2] - self.target_z)

            if is_touching or dist_to_bottle < 0.06:
                total_reward += 10.0
                self.in_approach = False

            if self.episode_length % 50 == 0:
                print(f"Step {self.episode_length} [APPROACH]: dist={dist_to_bottle:.3f}m")

        else:
            progress_delta = max(progress - self.prev_progress, 0.0)

            # Smooth interpretable weights
            w_progress = 100.0
            w_deviation = 40.0
            w_angle = 2.0
            w_tilt = 50.0
            w_wrist = 0.5
            contact_bonus = 2.0
            no_contact_penalty = 2.0
            time_penalty = 0.005

            r_progress = w_progress * progress_delta
            r_deviation = -w_deviation * deviation_mag
            r_angle = -w_angle * angle_error
            r_tilt = -w_tilt * (1.0 - tilt)
            r_contact = contact_bonus if is_touching else -no_contact_penalty

            # ================= Wrist alignment reward =================
            desired_push_angle = np.arctan2(
                self.last_desired_push_dir[1],
                self.last_desired_push_dir[0]
            )

            current_wrist_angle = self.wrist_offset + self.gripper_push_angle_at_home

            wrist_error = desired_push_angle - current_wrist_angle
            while wrist_error > np.pi:
                wrist_error -= 2 * np.pi
            while wrist_error < -np.pi:
                wrist_error += 2 * np.pi

            r_wrist = -w_wrist * abs(wrist_error)

            # ================= Wrist smoothness penalty =================
            if action is not None and self.prev_action is not None:
                wrist_delta = abs(self.prev_action[2] - action[2])
                r_wrist_smooth = -0.05 * wrist_delta
            else:
                r_wrist_smooth = 0.0

            total_reward = (
                r_progress +
                r_deviation +
                r_angle +
                r_tilt +
                r_contact +
                r_wrist +
                r_wrist_smooth -
                time_penalty
            )

            if self.episode_length % 50 == 0:
                K_avg = np.mean(self.current_K)
                mode = "SETTLE" if self.is_settling else "PUSH"
                contact_status = "CONTACT" if is_touching else "NO CONTACT!"
                print(
                    f"Step {self.episode_length} [{mode}]: "
                    f"prog={progress:.1%}, dev={deviation_mag * 100:.1f}cm, "
                    f"θ={np.degrees(angle_error):.1f}°, K={K_avg:.0f}, "
                    f"F={force_mag:.1f}N, tilt={tilt:.4f}, {contact_status}"
                )

        # Terminal terms
        if progress > 0.95 and deviation_mag < self.path_tolerance:
            total_reward += 100.0
            info["is_success"] = True
            print(f"SUCCESS at step {self.episode_length}!")

        if tilt < 0.5:
            total_reward -= 100.0
            info["bottle_fallen"] = True

        if deviation_mag > 0.15:
            total_reward -= 50.0
            info["off_path"] = True

        self.prev_progress = progress

        info["progress"] = progress
        info["deviation"] = deviation_mag
        info["angle_error"] = angle_error
        info["force_magnitude"] = force_mag
        info["bottle_tilt"] = tilt
        info["stiffness"] = np.mean(self.current_K)

        return total_reward, info

    # ==================================================================
    #                      RESET / STEP
    # ==================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

        self.current_qpos_target = self.data.qpos[7:14].copy()
        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(100):
            self.data.ctrl[:7] = self.current_qpos_target
            mujoco.mj_step(self.model, self.data)

        bottle_start = self.data.xpos[self.bottle_body_id].copy()

        traj_type = self.trajectory_type
        if options and "trajectory_type" in options:
            traj_type = options["trajectory_type"]

        self.trajectory = self._generate_trajectory(bottle_start[:2], traj_type)
        self._compute_arc_length()

        self.prev_progress = 0.0
        self.in_approach = True
        self.episode_length = 0
        self.current_K = np.array([200.0, 200.0], dtype=np.float32)
        self.is_settling = False
        self.settle_counter = 0
        self.last_desired_push_dir = np.array([0.0, -1.0], dtype=np.float32)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        self.base_wrist_pos = self.current_qpos_target[6]
        self.wrist_offset = 0.0

        self.force_profile_log = []
        self.stiffness_profile_log = []
        self.deviation_log = []
        self.angle_error_log = []

        print(f"\n{'=' * 50}")
        print(f"EPISODE: {traj_type}")
        print(f"Bottle start: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"Goal: ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})")
        print(f"Trajectory arc length: {self.total_arc_length:.3f}m")
        print(f"{'=' * 50}")

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        if self.in_approach:
            cart_vel, dist = self._compute_approach_velocity()
            target_wrist_rotation = 0.0

            if dist < 0.06 or self._is_touching():
                self.in_approach = False
                print(f"Step {self.episode_length}: → PUSH phase")
        else:
            cart_vel = self._compute_push_velocity(action)

            # RL wrist control
            wrist_mod = float(action[2])
            target_wrist_rotation = wrist_mod * self.max_wrist_rotation

        q_dot = self._cartesian_to_joint_velocity(cart_vel)
        dt = 0.02

        self.current_qpos_target[:6] = np.clip(
            self.current_qpos_target[:6] + q_dot[:6] * dt,
            self.act_low[:6], self.act_high[:6]
        )

        wrist_error = target_wrist_rotation - self.wrist_offset
        wrist_speed = 5.0 * wrist_error
        wrist_speed = np.clip(wrist_speed, -2.0, 2.0)

        self.wrist_offset += wrist_speed * dt
        self.wrist_offset = np.clip(
            self.wrist_offset,
            -self.max_wrist_rotation,
            self.max_wrist_rotation
        )

        self.current_qpos_target[6] = np.clip(
            self.base_wrist_pos + self.wrist_offset,
            self.act_low[6], self.act_high[6]
        )

        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Logging
        force = self._get_contact_force()
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        _, dev = self._get_path_deviation(bottle_xy)

        self.force_profile_log.append(force.copy())
        self.stiffness_profile_log.append(self.current_K.copy())
        self.deviation_log.append(dev)

        obs = self._get_obs()
        reward, info = self._get_reward(action)
        self.episode_length += 1

        terminated = bool(info.get("is_success") or info.get("bottle_fallen") or info.get("off_path"))
        truncated = bool(self.episode_length >= self.max_episode_length)

        self.prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    # ==================================================================
    #                      RENDER / CLOSE
    # ==================================================================

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.user_scn.ngeom = 0

            if self.trajectory is not None:
                self.viewer.user_scn.ngeom = 0
                for i in range(len(self.trajectory) - 1):
                    if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                        break

                    p1 = np.array([self.trajectory[i][0], self.trajectory[i][1], 0.801])
                    p2 = np.array([self.trajectory[i + 1][0], self.trajectory[i + 1][1], 0.801])

                    mujoco.mjv_initGeom(
                        self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=[0.003, 0, 0],
                        pos=(p1 + p2) / 2,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32)
                    )
                    mujoco.mjv_connector(
                        self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        0.003,
                        p1, p2
                    )
                    self.viewer.user_scn.ngeom += 1

            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None