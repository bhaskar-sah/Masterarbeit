"""
Panda Push Environment - Universal Policy Version

Key Concept:
    - Push direction = Vector from bottle to LOOKAHEAD TARGET
    - ONE set of parameters for ALL trajectories
    - RL learns to adapt via actions (K, speed, wrist)
    - No force penalty - let RL discover optimal force profile

The agent learns to handle different curvatures through its actions,
not through trajectory-specific parameter tuning.
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os


class PandaPushTrajectoryEnv(gym.Env):
    """
    Universal policy environment - same parameters for all trajectory types.
    RL agent learns to adapt through actions.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_type="straight"):
        super().__init__()

        # Load model
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

        self.link6_body_id = self.model.body("link6").id
        self.link7_body_id = self.model.body("link7").id

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

        # ==================== PUSH PARAMETERS (UNIVERSAL) ====================
        self.base_forward_speed = 0.012  # Reduced from 0.015 to prevent tipping
        self.behind_distance = 0.03

        # ==================== LOOKAHEAD PARAMETERS (UNIVERSAL) ====================
        # Conservative middle-ground values that work for all trajectories
        # RL adapts to curvature via its actions, not via these parameters
        self.lookahead_points = 4
        self.blend_threshold_low = 0.02  # 1cm - start blending earlier
        self.blend_threshold_high = 0.08  # 4cm - full correction earlier

        # ==================== WRIST ROTATION ====================
        self.gripper_push_angle_at_home = -np.pi / 2
        self.max_wrist_rotation = 2.5
        self.wrist_offset = 0.0
        self.base_wrist_pos = 0.0

        # ==================== STIFFNESS (RL learns this!) ====================
        self.K_min = 100.0
        self.K_max = 500.0
        self.current_K = np.array([300.0, 300.0])

        # ==================== TILT SAFETY ====================
        self.tilt_ok = 0.99
        self.tilt_slow = 0.98
        self.tilt_stop = 0.96

        self.is_settling = False
        self.settle_counter = 0
        self.settle_required = 25

        # ==================== CARTESIAN CONTROL ====================
        # Hand body is at wrist; fingertips are ~10cm below
        # target_z=0.92 means fingertips at ~0.82 (just above table at 0.80)
        self.target_z = 0.92
        self.z_gain = 10.0
        self.damping = 0.01

        # ==================== TRAJECTORY ====================
        self.trajectory_type = trajectory_type
        self.trajectory = None
        self.total_arc_length = 0.0
        self.path_tolerance = 0.05
        # DOWN-LEFT direction - matches robot's natural push capability
        self.goal_position = np.array([0.4, -0.4])  # Back to this (worked better)

        # Phase
        self.in_approach = True

        # Action space: [forward_mod, wrist_mod, Kx, Ky]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space
        obs_dim = 40  # Added correction_flag
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Adaptive Z reference
        self.bottle_start_pos = np.zeros(2)

        # State
        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0
        self.max_episode_length = 2500
        self.prev_progress = 0.0

        self.prev_action = None

        # Logging
        self.force_profile_log = []
        self.stiffness_profile_log = []
        self.deviation_log = []
        self.angle_error_log = []

        print(f"\n{'=' * 60}")
        print("UNIVERSAL POLICY - Same Parameters for All Trajectories")
        print(f"{'=' * 60}")
        print("RL controls: forward_speed, wrist_rotation, Kx, Ky")
        print("Agent learns to adapt to curvature via actions")
        print(f"Lookahead: {self.lookahead_points} points")
        print(f"Blend thresholds: [{self.blend_threshold_low * 100:.1f}, {self.blend_threshold_high * 100:.1f}] cm")
        print(f"K range: [{self.K_min}, {self.K_max}] N/m")
        print(f"No force penalty - RL discovers optimal force profile")
        print(f"{'=' * 60}")

    # ==================================================================
    #           CORE CONCEPT: Lookahead Target + Blended Direction
    # ==================================================================

    def _get_lookahead_target(self, bottle_xy):
        """Get target point ahead of bottle's closest point on trajectory."""
        if self.trajectory is None or len(self.trajectory) < 2:
            return self.goal_position.copy(), 0

        idx, _, _ = self._get_closest_point_on_trajectory(bottle_xy)
        target_idx = min(idx + self.lookahead_points, len(self.trajectory) - 1)
        target_point = self.trajectory[target_idx].copy()

        if target_idx >= len(self.trajectory) - 2:
            target_point = self.goal_position.copy()

        return target_point, target_idx

    def _get_push_direction(self, bottle_xy):
        """
        BLENDED push direction:
        - When ON trajectory: push along TANGENT
        - When OFF trajectory: push toward LOOKAHEAD TARGET
        """
        target_point, _ = self._get_lookahead_target(bottle_xy)
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)

        to_target = target_point - bottle_xy
        distance = np.linalg.norm(to_target)

        if distance > 1e-6:
            correction_dir = to_target / distance
        else:
            correction_dir = self._get_path_tangent(bottle_xy)

        tangent_dir = self._get_path_tangent(bottle_xy)

        # Blend: 0 = pure tangent, 1 = full correction
        blend_range = self.blend_threshold_high - self.blend_threshold_low
        if blend_range < 1e-6:
            blend = 0.0
        else:
            blend = np.clip(
                (deviation_mag - self.blend_threshold_low) / blend_range,
                0.0,
                1.0
            )

        push_direction = (1.0 - blend) * tangent_dir + blend * correction_dir

        norm = np.linalg.norm(push_direction)
        if norm > 1e-6:
            push_direction = push_direction / norm
        else:
            push_direction = tangent_dir

        return push_direction.astype(np.float32), float(distance), target_point.astype(np.float32), float(blend)

    def _get_angle_error(self, bottle_xy):
        """Compute angle θ between current force and desired push direction."""
        contact_force = self._get_contact_force()
        force_mag = np.linalg.norm(contact_force[:2])
        push_dir, _, _, _ = self._get_push_direction(bottle_xy)

        if force_mag > 0.5:
            P_normalized = -contact_force[:2] / force_mag
            dot_product = np.clip(np.dot(P_normalized, push_dir), -1.0, 1.0)
            angle_error = np.arccos(dot_product)
        else:
            angle_error = 0.0

        return float(angle_error)

    def _get_hand_target_position(self, bottle_xy):
        """
        Hand position: simply behind bottle along push direction.

        The push_dir already blends tangent + correction, so positioning
        the hand behind the bottle along push_dir naturally creates the
        corrective force. No additional side offset needed!
        """
        push_dir, _, _, _ = self._get_push_direction(bottle_xy)

        # Hand is behind bottle, opposite to push direction
        hand_xy = bottle_xy - push_dir * self.behind_distance

        return np.array([hand_xy[0], hand_xy[1], self.target_z])

    # ==================================================================
    #           VELOCITY COMPUTATION
    # ==================================================================

    def _compute_approach_velocity(self):
        """Approach: Move hand behind bottle."""
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        target_pos = self._get_hand_target_position(bottle_xy)
        error = target_pos - hand_pos
        dist = np.linalg.norm(error[:2])

        v_desired = error * 2.0
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.04:
            v_desired[:2] = v_desired[:2] / v_mag * 0.04

        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        return v_desired, dist

    def _compute_adaptive_z(self, bottle_pos):
        """Compute adaptive Z based on total distance traveled from start."""
        distance_from_start = np.linalg.norm(bottle_pos[:2] - self.bottle_start_pos)
        current_target_z = self.target_z + 0.02 * distance_from_start
        return max(current_target_z, 0.88)

    # @staticmethod
    def _get_hand_alignment(self, bottle_xy, hand_xy, push_dir):
        """
        Returns alignment score:
        +1 -> hand is well behind bottle relative to push direction
         0 -> hand is on the side
        -1 -> hand is in front of bottle (bad for pushing)
        """
        hand_to_bottle = bottle_xy - hand_xy
        norm = np.linalg.norm(hand_to_bottle)

        if norm < 1e-6:
            return 1.0

        return float(np.dot(hand_to_bottle / norm, push_dir))

    def _compute_reposition_velocity(self, hand_pos, target_hand_pos, lift_height):
        v_desired = np.zeros(3, dtype=np.float32)

        pos_error_xy = target_hand_pos[:2] - hand_pos[:2]
        # v_desired[:2] = 6.0 * pos_error_xy
        v_desired[:2] = 8.0 * pos_error_xy
        # v_desired[:2] = np.clip(v_desired[:2], -0.08, 0.08)
        v_desired[:2] = np.clip(v_desired[:2], -0.12, 0.12)

        # z_target = lift_height
        v_desired[2] = self.z_gain * (lift_height - hand_pos[2])

        return v_desired

    def _compute_recovery_velocity(self, hand_pos, target_hand_pos):
        v_desired = np.zeros(3, dtype=np.float32)

        recovery_vec = target_hand_pos[:2] - hand_pos[:2]
        recovery_dist = np.linalg.norm(recovery_vec)

        if recovery_dist > 1e-6:
            recovery_dir = recovery_vec / recovery_dist
            recovery_speed = min(0.05, 2.0 * recovery_dist)
            v_desired[:2] = recovery_dir * recovery_speed

        # v_desired[2] = self.z_gain * (target_hand_pos[2] - hand_pos[2])
        recovery_z = target_hand_pos[2] + 0.02
        v_desired[2] = self.z_gain * (recovery_z - hand_pos[2])
        return v_desired

    def _compute_push_velocity(self, action):
        """Push velocity with continuous contact method."""
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]
        tilt = self._get_bottle_tilt()
        is_touching = self._is_touching()

        current_target_z = self._compute_adaptive_z(bottle_pos)

        # Parse RL actions
        forward_mod = float(action[0])
        self.current_K[0] = self.K_min + action[2] * (self.K_max - self.K_min)
        self.current_K[1] = self.K_min + action[3] * (self.K_max - self.K_min)
        K_avg = np.mean(self.current_K)

        # ==================== tilt and settling condition ====================
        if self.is_settling:
            self.settle_counter += 1

            if self._check_stable():
                self.is_settling = False
                self.settle_counter = 0
            elif self.settle_counter >= 200:
                self.is_settling = False
                self.settle_counter = 0

            if self.is_settling:
                v_desired = np.zeros(3, dtype=np.float32)
                v_desired[2] = self.z_gain * (current_target_z - hand_pos[2])
                return v_desired

        if tilt < self.tilt_stop:
            self.is_settling = True
            self.settle_counter = 0
            v_desired = np.zeros(3, dtype=np.float32)
            v_desired[2] = self.z_gain * (current_target_z - hand_pos[2])
            return v_desired

        # ===================== tilt-based speed =======================
        if tilt > self.tilt_ok:
            speed_mult = 1.0
        elif tilt > self.tilt_slow:
            speed_mult = 0.5
        else:
            speed_mult = 0.2

        # ==================== Geometry ====================
        push_dir, dist_to_target, target_point, blend = self._get_push_direction(bottle_xy)
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        target_hand_pos = self._get_hand_target_position(bottle_xy)

        pos_error = target_hand_pos[:2] - hand_pos[:2]
        pos_error_mag = np.linalg.norm(pos_error)

        hand_alignment = self._get_hand_alignment(bottle_xy, hand_pos[:2], push_dir)

        # ==================== Mode decision ======================
        # Reposition if hand is clearly on wrong side or very far from target behind-bottle pose
        # need_reposition = (hand_alignment < 0.2 and deviation_mag > 0.02) or (pos_error_mag > 0.06 and deviation_mag > 0.03)
        need_reposition = (
                (hand_alignment < 0.4 and deviation_mag > 0.015) or
                (pos_error_mag > 0.04 and deviation_mag > 0.02)
        )

        if need_reposition:
            self.in_correction_mode = True
            # lift_height = current_target_z + 0.03
            lift_height = current_target_z + 0.05
            v_desired = self._compute_reposition_velocity(hand_pos, target_hand_pos, lift_height)

            if self.episode_length % 50 == 0:
                print(f" [REPOSITION] align={hand_alignment:.2f}, dev={deviation_mag*100:.1f}cm, pos_error={pos_error_mag*100:.1f}cm")

            return v_desired

        # Recover contact toward desired contact pose, not bottle center
        if not is_touching:
            self.in_correction_mode = True
            v_desired = self._compute_recovery_velocity(hand_pos, target_hand_pos)

            if self.episode_length % 50 == 0:
                print(f" [RECOVERY] dev={deviation_mag*100:.1f}cm, pos_error={pos_error_mag*100:.}cm")

            return v_desired

        # ==================== NORMAL PUSH ====================
        self.in_correction_mode = blend > 0.2

        v_desired = np.zeros(3, dtype=np.float32)

        base_speed = self.base_forward_speed * (1.0 + 0.3 * forward_mod) * speed_mult
        k_factor = 0.85 + 0.30 * (K_avg / self.K_max)
        raw_push_speed = base_speed * k_factor

        # Keep push term alive even if hand target error is not perfect
        # alignment_factor = np.clip(1.0 - (pos_error_mag / 0.10), 0.5, 1.0)
        alignment_factor = np.clip(1.0 - (pos_error_mag / 0.10), 0.7, 1.0)
        push_speed = raw_push_speed * alignment_factor

        v_push = push_dir * push_speed

        # Tracking only helps maintain behind-bottle placement
        # Parallel component stronger, perpendicular weaker
        # tracking_gain_parallel = 6.0
        tracking_gain_parallel = 4.0
        # tracking_gain_perp = 1.5 * (1.0 - blend)
        tracking_gain_perp = 0.8 * (1.0 - blend)

        error_parallel = np.dot(pos_error, push_dir) * push_dir
        error_perpendicular = pos_error - error_parallel

        v_tracking_parallel = tracking_gain_parallel * error_parallel
        v_tracking_perpendicular = tracking_gain_perp * error_perpendicular
        v_tracking = v_tracking_parallel + v_tracking_perpendicular
        # v_tracking = np.clip(v_tracking, -0.06, 0.06)
        v_tracking = np.clip(v_tracking, -0.04, 0.04)

        v_desired[0] = v_push[0] + v_tracking[0]
        v_desired[1] = v_push[1] + v_tracking[1]

        max_vel = 0.08
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > max_vel:
            v_desired[:2] = v_desired[:2] / v_mag * max_vel

        v_desired[2] = self.z_gain * (current_target_z - hand_pos[2])

        angle_error = self._get_angle_error(bottle_xy)
        self.angle_error_log.append(angle_error)

        if self.episode_length % 50 == 0:
            idx, closest_pt, _ = self._get_closest_point_on_trajectory(bottle_xy)
            angle_deg = np.degrees(angle_error)
            hand_error = target_hand_pos[:2] - hand_pos[:2]
            mode_str = "CORR" if self.in_correction_mode else "PUSH"

            print(f"    Bottle: [{bottle_xy[0]:.3f}, {bottle_xy[1]:.3f}], "
                  f"Traj: [{closest_pt[0]:.3f}, {closest_pt[1]:.3f}]")
            print(f"    Push: [{push_dir[0]:.2f}, {push_dir[1]:.2f}], "
                  f"dev={deviation_mag * 100:.1f}cm, blend={blend:.2f}, mode={mode_str}")
            print(f"    Hand: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}], "
                  f"Target: [{target_hand_pos[0]:.3f}, {target_hand_pos[1]:.3f}], "
                  f"Error: [{hand_error[0] * 100:.1f}, {hand_error[1] * 100:.1f}]cm")
            print(f"    v_push: [{v_push[0]:.4f}, {v_push[1]:.4f}], "
                  f"v_track: [{v_tracking[0]:.4f}, {v_tracking[1]:.4f}], "
                  f"v_total: [{v_desired[0]:.4f}, {v_desired[1]:.4f}]")

        return v_desired


    def _get_bottle_tilt(self):
        """Get bottle tilt (1.0 = upright)."""
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        return bottle_mat[2, 2]

    def _check_stable(self):
        """Check if bottle is stable."""
        tilt = self._get_bottle_tilt()
        angvel = np.linalg.norm(self.data.qvel[3:6])
        return tilt > 0.97 and angvel < 0.1

    def _cartesian_to_joint_velocity(self, cart_vel):
        """Convert Cartesian velocity to joint velocities."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)

        J = jacp[:, 6:13]
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.damping ** 2 * np.eye(3))

        return J_pinv @ cart_vel

    # ==================================================================
    #                      TRAJECTORY
    # ==================================================================

    def _generate_trajectory(self, bottle_start_xy, traj_type):
        """Generate trajectory from bottle start to goal."""
        start = bottle_start_xy.copy()
        n_points = 50
        goal = self.goal_position.copy()

        if traj_type == "straight":
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, goal)
            return traj.astype(np.float32)

        elif traj_type == "curved":
            t = np.linspace(0, 1, n_points)
            # FIX: Control point relative to start/goal, not hardcoded
            mid_point = np.array([
                (start[0] + goal[0]) / 2 + 0.15,
                (start[1] + goal[1]) / 2
            ])
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid_point[0] + t ** 2 * goal[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid_point[1] + t ** 2 * goal[1]
            return np.stack([x, y], axis=1).astype(np.float32)

        elif traj_type == "s_curve":
            t = np.linspace(0, 1, n_points)
            wiggle_strength = np.sin(np.pi * t)
            # FIX: X also progresses toward goal, not just wiggling around start
            x_center = start[0] + t * (goal[0] - start[0])
            x = x_center + 0.08 * np.sin(2 * np.pi * t) * wiggle_strength
            y = start[1] + t * (goal[1] - start[1])
            return np.stack([x, y], axis=1).astype(np.float32)

        else:
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, goal)
            return traj.astype(np.float32)

    def _compute_arc_length(self):
        """Compute total arc length of trajectory."""
        if self.trajectory is None or len(self.trajectory) < 2:
            self.total_arc_length = 0.0
            return
        diffs = np.diff(self.trajectory, axis=0)
        self.total_arc_length = np.sum(np.linalg.norm(diffs, axis=1))

    def _get_closest_point_on_trajectory(self, pos_xy):
        """Find closest point on trajectory."""
        if self.trajectory is None:
            return 0, pos_xy.copy(), 0.0
        distances = np.linalg.norm(self.trajectory - pos_xy, axis=1)
        idx = np.argmin(distances)
        closest_pt = self.trajectory[idx].copy()
        arc_len = np.sum(np.linalg.norm(np.diff(self.trajectory[:idx + 1], axis=0), axis=1)) if idx > 0 else 0.0
        return idx, closest_pt, arc_len

    def _get_path_deviation(self, bottle_xy):
        """Get deviation from trajectory."""
        _, closest_pt, _ = self._get_closest_point_on_trajectory(bottle_xy)
        deviation_vec = closest_pt - bottle_xy
        return deviation_vec.astype(np.float32), float(np.linalg.norm(deviation_vec))

    def _get_path_tangent(self, bottle_xy):
        """Get tangent at closest point."""
        if self.trajectory is None or len(self.trajectory) < 2:
            return np.array([0.0, -1.0], dtype=np.float32)
        idx, _, _ = self._get_closest_point_on_trajectory(bottle_xy)
        if idx < len(self.trajectory) - 1:
            tangent = self.trajectory[idx + 1] - self.trajectory[idx]
        else:
            tangent = self.trajectory[idx] - self.trajectory[idx - 1]
        norm = np.linalg.norm(tangent)
        return (tangent / norm).astype(np.float32) if norm > 1e-6 else np.array([0.0, -1.0], dtype=np.float32)

    def _get_progress(self, bottle_xy):
        """Get progress along trajectory (0 to 1)."""
        if self.total_arc_length < 1e-6:
            return 0.0
        _, _, arc_len = self._get_closest_point_on_trajectory(bottle_xy)
        return float(np.clip(arc_len / self.total_arc_length, 0.0, 1.0))

    # ==================================================================
    #                      CONTACT
    # ==================================================================

    def _get_contact_force(self):
        """Get contact force between robot and bottle."""
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
        """Check if robot is touching bottle."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            robot_touch = body1 in self.robot_contact_bodies or body2 in self.robot_contact_bodies
            bottle_touch = body1 == self.bottle_body_id or body2 == self.bottle_body_id
            if robot_touch and bottle_touch:
                return True
        return False

    def _is_wrong_body_touching(self):
        """Check if link6/7 is touching bottle."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            bottle_touch = body1 == self.bottle_body_id or body2 == self.bottle_body_id
            if bottle_touch:
                other_body = body1 if body2 == self.bottle_body_id else body2
                if other_body in {self.link6_body_id, self.link7_body_id}:
                    return True
        return False

    # ==================================================================
    #                      OBSERVATION
    # ==================================================================

    def _get_obs(self):
        """Build observation vector."""
        qpos = self.data.qpos[7:14].astype(np.float32)
        qvel = self.data.qvel[6:13].astype(np.float32)
        hand_pos = self.data.xpos[self.hand_body_id].astype(np.float32)
        bottle_pos = self.data.xpos[self.bottle_body_id].astype(np.float32)
        bottle_xy = bottle_pos[:2]

        hand_to_bottle = bottle_pos[:2] - hand_pos[:2]
        dist_to_bottle = np.linalg.norm(hand_to_bottle)
        dir_to_bottle = hand_to_bottle / (dist_to_bottle + 1e-6)

        push_dir, dist_to_target, _, _ = self._get_push_direction(bottle_xy)
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        angle_error = self._get_angle_error(bottle_xy)
        progress = self._get_progress(bottle_xy)
        contact_force = self._get_contact_force()
        is_touching = np.array([1.0 if self._is_touching() else 0.0], dtype=np.float32)

        K_normalized = (self.current_K - self.K_min) / (self.K_max - self.K_min)
        settling_flag = np.array([1.0 if self.is_settling else 0.0], dtype=np.float32)
        wrist_normalized = np.array([self.wrist_offset / self.max_wrist_rotation], dtype=np.float32)
        correction_flag = np.array([1.0 if getattr(self, 'in_correction_mode', False) else 0.0], dtype=np.float32)

        obs = np.concatenate([
            qpos,  # 7
            qvel,  # 7
            hand_pos,  # 3
            bottle_pos,  # 3
            dir_to_bottle,  # 2
            [dist_to_bottle],  # 1
            push_dir,  # 2
            [dist_to_target],  # 1
            deviation_vec,  # 2
            [deviation_mag],  # 1
            [angle_error],  # 1
            [progress],  # 1
            contact_force,  # 3
            is_touching,  # 1
            K_normalized,  # 2
            settling_flag,  # 1
            wrist_normalized,  # 1
            correction_flag,  # 1
        ])  # Total: 40

        return obs.astype(np.float32)

    # ==================================================================
    #                      REWARD
    # ==================================================================

    def _get_reward(self, action=None):
        """
        Reward function - NO FORCE PENALTY.
        RL discovers optimal force profile through outcome-based rewards.
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
        angle_error = self._get_angle_error(bottle_xy)

        if self.in_approach:
            total_reward = -2.0 * dist_to_bottle
            total_reward += -5.0 * abs(hand_pos[2] - self.target_z)
            if is_touching or dist_to_bottle < 0.06:
                total_reward += 10.0
                self.in_approach = False

            if self.episode_length % 50 == 0:
                print(f"Step {self.episode_length} [APPROACH]: dist={dist_to_bottle:.3f}m")
        else:
            # 1. Progress reward
            progress_delta = progress - self.prev_progress
            r_progress = 50.0 * max(progress_delta, 0)

            # 2. Deviation penalty (quadratic)
            r_deviation = 5.0 - 2000.0 * deviation_mag * deviation_mag
            r_deviation = max(r_deviation, -20.0)

            # 3. Angle error reward - CRITICAL for correction!
            # When deviated, reducing angle error (aligning force with correction) is priority
            angle_deg = np.degrees(angle_error)
            if not is_touching or force_mag < 0.5:
                r_angle = 0.0
            else:
                angle_deg_clamped = min(angle_deg, 90.0)
                # Base angle reward: good alignment = +5, perpendicular = -10
                r_angle_base = 5.0 - (angle_deg_clamped / 6.0)

                # Scale importance by deviation: when off-track, alignment matters MORE
                # At 5cm deviation, angle reward is 3x more important
                deviation_scale = 1.0 + 20.0 * deviation_mag  # 1.0 at 0cm, 2.0 at 5cm, 3.0 at 10cm
                r_angle = r_angle_base * deviation_scale

            # 4. Stability reward
            tilt_clamped = max(tilt, 0.96)
            r_stability = 150.0 * (tilt_clamped - 0.98)

            # 5. Contact reward
            wrong_body = self._is_wrong_body_touching()
            if wrong_body:
                r_contact = -5.0
            elif is_touching:
                r_contact = 2.0
            else:
                r_contact = -10.0 * dist_to_bottle
                if dist_to_bottle > 0.08:
                    r_contact -= 5.0

            # let RL discover optimal force profile!
            total_reward = r_progress + r_deviation + r_angle + r_stability + r_contact

            # 6. Action smoothing (stiffness + wrist)
            if action is not None and self.prev_action is not None:
                k_delta = np.linalg.norm(action[2:4] - self.prev_action[2:4])
                wrist_delta = abs(action[1] - self.prev_action[1])
                r_smooth = (-0.1 * k_delta) + (-2.0 * wrist_delta)
                total_reward += r_smooth

            if self.episode_length % 50 == 0:
                K_avg = np.mean(self.current_K)
                mode = "SETTLE" if self.is_settling else "PUSH"
                contact_status = "CONTACT" if is_touching else "NO CONTACT!"
                print(f"Step {self.episode_length} [{mode}]: prog={progress:.1%}, "
                      f"dev={deviation_mag * 100:.1f}cm, θ={angle_deg:.1f}°, K={K_avg:.0f}, "
                      f"F={force_mag:.1f}N, tilt={tilt:.4f}, {contact_status}")

        # Time penalty
        total_reward -= 0.01

        # Success
        if progress > 0.95 and deviation_mag < self.path_tolerance:
            total_reward += 100.0
            info["is_success"] = True
            print(f"SUCCESS at step {self.episode_length}!")

        # Failures
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
        self.bottle_start_pos = bottle_start[:2].copy()

        traj_type = self.trajectory_type
        if options and "trajectory_type" in options:
            traj_type = options["trajectory_type"]

        self.trajectory = self._generate_trajectory(bottle_start[:2], traj_type)
        self._compute_arc_length()

        self.prev_progress = 0.0
        self.in_approach = True
        self.episode_length = 0
        self.current_K = np.array([200.0, 200.0])
        self.is_settling = False
        self.settle_counter = 0

        self.base_wrist_pos = self.current_qpos_target[6]
        self.wrist_offset = 0.0

        # Reset correction mode state
        self.in_correction_mode = False
        if hasattr(self, 'correction_wrist_target'):
            delattr(self, 'correction_wrist_target')

        self.force_profile_log = []
        self.stiffness_profile_log = []
        self.deviation_log = []
        self.angle_error_log = []

        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        print(f"\n{'=' * 50}")
        print(f"EPISODE: {traj_type}")
        print(f"Bottle start: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"Goal: ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})")
        print(f"Arc length: {self.total_arc_length:.3f}m")
        print(f"{'=' * 50}")

        return self._get_obs(), {}

    def step(self, action):
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        if self.in_approach:
            cart_vel, dist = self._compute_approach_velocity()

            # Pre-align wrist to trajectory tangent (straight down -Y for start)
            tangent = self._get_path_tangent(bottle_xy)
            push_angle = np.arctan2(tangent[1], tangent[0])
            alignment_rotation = push_angle - self.gripper_push_angle_at_home

            while alignment_rotation > np.pi: alignment_rotation -= 2 * np.pi
            while alignment_rotation < -np.pi: alignment_rotation += 2 * np.pi

            target_wrist_rotation = np.clip(alignment_rotation, -self.max_wrist_rotation, self.max_wrist_rotation)

            # Check if hand is properly positioned behind bottle
            hand_pos = self.data.xpos[self.hand_body_id][:2]
            hand_to_bottle = bottle_xy - hand_pos
            hand_to_bottle_dist = np.linalg.norm(hand_to_bottle)

            # For straight trajectory, hand should be ABOVE bottle (positive Y relative to bottle)
            # because we push in -Y direction
            hand_behind_bottle = hand_to_bottle[1] < 0  # Hand Y < Bottle Y means hand is "above" in +Y

            # Only exit approach when:
            # 1. Hand is close to target position (dist < 0.04)
            # 2. OR touching AND hand is behind bottle
            proper_position = dist < 0.04
            touching_and_aligned = self._is_touching() and hand_behind_bottle and hand_to_bottle_dist < 0.08

            if proper_position or touching_and_aligned:
                self.in_approach = False
                print(f"Step {self.episode_length}: → PUSH phase (dist={dist:.3f}, behind={hand_behind_bottle})")
        else:
            cart_vel = self._compute_push_velocity(action)

            # For debugging: Use PURE geometric wrist alignment (no RL tweak)
            push_dir, _, _, _ = self._get_push_direction(bottle_xy)
            ideal_push_angle = np.arctan2(push_dir[1], push_dir[0])
            base_alignment = ideal_push_angle - self.gripper_push_angle_at_home

            while base_alignment > np.pi: base_alignment -= 2 * np.pi
            while base_alignment < -np.pi: base_alignment += 2 * np.pi

            # Geometric base + RL residual for wrist
            # RL gets ±0.3 radians (~17°) to fine-tune alignment
            rl_wrist_tweak = action[1] * 0.15 # 0.3
            target_wrist_rotation = base_alignment + rl_wrist_tweak
            target_wrist_rotation = np.clip(target_wrist_rotation, -self.max_wrist_rotation, self.max_wrist_rotation)

        q_dot = self._cartesian_to_joint_velocity(cart_vel)
        dt = 0.02

        self.current_qpos_target[:6] = np.clip(
            self.current_qpos_target[:6] + q_dot[:6] * dt,
            self.act_low[:6], self.act_high[:6]
        )

        wrist_error = target_wrist_rotation - self.wrist_offset
        wrist_speed = 5.0 * wrist_error  # Increased from 3.0
        wrist_speed = np.clip(wrist_speed, -1.0, 1.0)  # Increased from 0.5

        self.wrist_offset += wrist_speed * dt
        self.wrist_offset = np.clip(self.wrist_offset, -self.max_wrist_rotation, self.max_wrist_rotation)

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