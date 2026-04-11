import numpy as np
import mujoco


class PushController:
    """
    Handles push direction, hand positioning, and velocity computation
    using the Lookahead Target concept (Supervisor's Method).
    """

    def __init__(self, model, data, traj_manager, contact_manager,
                 hand_body_id, bottle_body_id,
                 goal_position, lookahead_points,
                 base_forward_speed, behind_distance,
                 target_z, z_gain, damping,
                 K_min, K_max,
                 tilt_ok, tilt_slow, tilt_stop,
                 settle_required):

        self.model = model
        self.data = data
        self.traj_manager = traj_manager
        self.contact_manager = contact_manager

        self.hand_body_id = hand_body_id
        self.bottle_body_id = bottle_body_id
        self.goal_position = goal_position

        self.lookahead_points = lookahead_points
        self.base_forward_speed = base_forward_speed
        self.behind_distance = behind_distance
        self.target_z = target_z
        self.z_gain = z_gain
        self.damping = damping

        self.K_min = K_min
        self.K_max = K_max

        self.tilt_ok = tilt_ok
        self.tilt_slow = tilt_slow
        self.tilt_stop = tilt_stop

        self.is_settling = False
        self.settle_counter = 0
        self.settle_required = settle_required

        self.current_K = np.array([300.0, 300.0])
        self.wrist_offset = 0.0
        self.logger = None  # set by env after construction

    # ==================================================================
    #           LOOKAHEAD TARGET + BLENDED DIRECTION
    # ==================================================================

    def get_lookahead_target(self, bottle_xy):
        """
        Get target point 3-4 points AHEAD of bottle's closest point on trajectory.
        This gives the bottle time to return to trajectory smoothly.

        Returns: target_point (2D), target_index
        """
        if self.traj_manager.trajectory is None or len(self.traj_manager.trajectory) < 2:
            return self.goal_position.copy(), 0

        idx, _, _ = self.traj_manager._get_closest_point_on_trajectory(bottle_xy)

        target_idx = min(idx + self.lookahead_points, len(self.traj_manager.trajectory) - 1)
        target_point = self.traj_manager.trajectory[target_idx].copy()

        if target_idx >= len(self.traj_manager.trajectory) - 2:
            target_point = self.goal_position.copy()

        return target_point, target_idx

    def get_push_direction(self, bottle_xy):
        """
        BLENDED push direction:
        - When ON trajectory (small deviation): push along TANGENT
        - When OFF trajectory (large deviation): push toward LOOKAHEAD TARGET
        - Smooth blend prevents overcorrection oscillation

        Returns: (push_direction, distance_to_target, target_point)
        """
        target_point, _ = self.get_lookahead_target(bottle_xy)
        deviation_vec, deviation_mag = self.traj_manager._get_path_deviation(bottle_xy)

        to_target = target_point - bottle_xy
        distance = np.linalg.norm(to_target)

        if distance > 1e-6:
            correction_dir = to_target / distance
        else:
            correction_dir = self.traj_manager._get_path_tangent(bottle_xy)

        tangent_dir = self.traj_manager._get_path_tangent(bottle_xy)

        blend = np.clip((deviation_mag - 0.005) / 0.02, 0.0, 1.0)

        push_direction = (1.0 - blend) * tangent_dir + blend * correction_dir

        norm = np.linalg.norm(push_direction)
        if norm > 1e-6:
            push_direction = push_direction / norm
        else:
            push_direction = tangent_dir

        return push_direction.astype(np.float32), float(distance), target_point

    def get_angle_error(self, bottle_xy):
        """
        Compute angle θ between current force direction P and desired push direction.

        Goal: Minimize this angle!

        Returns: angle_error in radians (0 = perfect alignment)
        """
        contact_force = self.contact_manager.get_contact_force()
        force_mag = np.linalg.norm(contact_force[:2])

        push_dir, _, _ = self.get_push_direction(bottle_xy)

        if force_mag > 0.5:
            P_normalized = -contact_force[:2] / force_mag
            dot_product = np.clip(np.dot(P_normalized, push_dir), -1.0, 1.0)
            angle_error = np.arccos(dot_product)
        else:
            angle_error = 0.0

        return float(angle_error)

    def get_hand_target_position(self, bottle_xy):
        """
        Hand position strategy:
        - Small deviation: behind bottle along push direction (normal pushing)
        - Large deviation: shift toward the side AWAY from trajectory
            so the hand can push bottle BACK toward trajectory

        Returns: 3D target position [x, y, z] for hand
        """
        push_dir, _, _ = self.get_push_direction(bottle_xy)
        hand_xy = bottle_xy - push_dir * self.behind_distance
        return np.array([hand_xy[0], hand_xy[1], self.target_z])

    # ==================================================================
    #           VELOCITY COMPUTATION
    # ==================================================================

    def compute_approach_velocity(self):
        """Approach: Move hand behind bottle."""
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        target_pos = self.get_hand_target_position(bottle_xy)
        error = target_pos - hand_pos
        dist = np.linalg.norm(error[:2])

        v_desired = error * 2.0
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.04:
            v_desired[:2] = v_desired[:2] / v_mag * 0.04

        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        return v_desired, dist

    def compute_push_velocity(self, action, episode_length, bottle_start_y):
        """
        SUPERVISOR'S METHOD:
        - Push direction = direction to lookahead target
        - Automatically blends forward + correction based on geometry
        - Hand tracks behind bottle relative to push direction
        """
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        tilt = self.get_bottle_tilt()
        is_touching = self.contact_manager.is_touching()

        # Adaptive height - compensate for arm droop at full extension
        bottle_y = self.data.xpos[self.bottle_body_id][1]
        distance_from_start = abs(bottle_y - bottle_start_y)
        current_target_z = self.target_z + 0.02 * distance_from_start
        current_target_z = max(current_target_z, 0.88)

        # Parse action
        forward_mod = action[0]
        self.current_K[0] = self.K_min + action[2] * (self.K_max - self.K_min)
        self.current_K[1] = self.K_min + action[3] * (self.K_max - self.K_min)
        K_avg = np.mean(self.current_K)

        # ==================== TILT SAFETY ====================
        if self.is_settling:
            self.settle_counter += 1

            if self._check_stable():
                self.is_settling = False
                self.settle_counter = 0
            elif self.settle_counter >= 200:
                self.is_settling = False
                self.settle_counter = 0

            if self.is_settling:
                v_desired = np.zeros(3)
                v_desired[2] = self.z_gain * (current_target_z - hand_pos[2])
                return v_desired

        if tilt < self.tilt_stop:
            self.is_settling = True
            self.settle_counter = 0

        # Speed multiplier based on tilt
        if tilt > self.tilt_ok:
            speed_mult = 1.0
        elif tilt > self.tilt_slow:
            speed_mult = 0.5
        else:
            speed_mult = 0.2

        v_desired = np.zeros(3)

        # ==================== CONTACT RECOVERY ====================
        if not is_touching:
            bottle_direction = bottle_xy - hand_pos[:2]
            bottle_dist = np.linalg.norm(bottle_direction)

            if bottle_dist > 0.01:
                bottle_dir_normalized = bottle_direction / bottle_dist
                recovery_speed = min(0.08, bottle_dist * 3.0)

                v_desired[0] = bottle_dir_normalized[0] * recovery_speed
                v_desired[1] = bottle_dir_normalized[1] * recovery_speed
                v_desired[2] = self.z_gain * (current_target_z - hand_pos[2])

                if episode_length % 50 == 0:
                    print(f"    CONTACT LOST! Recovery: dist={bottle_dist * 100:.1f}cm")

                return v_desired

        # ==================== PUSH USING LOOKAHEAD TARGET ====================
        push_dir, dist_to_target, target_point = self.get_push_direction(bottle_xy)

        # base_speed = self.base_forward_speed * (1.0 + forward_mod * 0.3) * speed_mult
        # k_factor = 0.8 + 0.4 * (K_avg / self.K_max)
        # push_speed = base_speed * k_factor

        # Agent can control speed from 10% to 100% (not 70% to 130%)
        speed_factor = 0.1 + 0.9 * ((forward_mod + 1.0) / 2.0)  # Maps [-1,1] to [0.1, 1.0]
        base_speed = self.base_forward_speed * speed_factor * speed_mult
        k_factor = 0.8 + 0.4 * (K_avg / self.K_max)
        push_speed = base_speed * k_factor

        v_push = push_dir * push_speed

        # ==================== TRACKING (stay behind bottle) ====================
        target_hand_pos = self.get_hand_target_position(bottle_xy)
        pos_error = target_hand_pos[:2] - hand_pos[:2]
        v_tracking = pos_error * 3.0
        v_tracking = np.clip(v_tracking, -0.03, 0.03)

        v_desired[0] = v_push[0] + v_tracking[0]
        v_desired[1] = v_push[1] + v_tracking[1]

        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.05:
            v_desired[:2] = v_desired[:2] / v_mag * 0.05

        v_desired[2] = self.z_gain * (current_target_z - hand_pos[2])

        # ==================== LOGGING ====================
        angle_error = self.get_angle_error(bottle_xy)
        if self.logger is not None:
            self.logger.log_angle_error(angle_error)

        if episode_length % 100 == 0:
            _, dev_mag = self.traj_manager._get_path_deviation(bottle_xy)
            _, target_idx = self.get_lookahead_target(bottle_xy)
            angle_deg = np.degrees(angle_error)
            _, deviation_mag = self.traj_manager._get_path_deviation(bottle_xy)
            blend = np.clip((deviation_mag - 0.005) / 0.02, 0.0, 1.0)
            print(f"    Lookahead: idx={target_idx}, push_dir=[{push_dir[0]:.2f}, {push_dir[1]:.2f}], "
                  f"θ={angle_deg:.1f}°, dev={dev_mag * 100:.1f}cm, blend={blend:.2f}")

        return v_desired

    # ==================================================================
    #           ROBOT KINEMATICS / BOTTLE STATE
    # ==================================================================

    def get_bottle_tilt(self):
        """Get bottle tilt (1.0 = upright, <1.0 = tilting)"""
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        return bottle_mat[2, 2]

    def _check_stable(self):
        """Check if bottle is stable (upright and not moving)"""
        tilt = self.get_bottle_tilt()
        angvel = np.linalg.norm(self.data.qvel[3:6])
        return tilt > 0.97 and angvel < 0.1

    def compute_wrist_joint_target(self, target_wrist_rotation, base_wrist_pos,
                                   max_wrist_rotation, act_low_6, act_high_6, dt):
        """
        Update wrist_offset toward target_wrist_rotation and return the
        clamped joint position target for the wrist (joint 6).
        """
        wrist_error = target_wrist_rotation - self.wrist_offset
        wrist_speed = np.clip(5.0 * wrist_error, -2.0, 2.0)
        self.wrist_offset += wrist_speed * dt
        self.wrist_offset = np.clip(self.wrist_offset, -max_wrist_rotation, max_wrist_rotation)
        return float(np.clip(base_wrist_pos + self.wrist_offset, act_low_6, act_high_6))

    def cartesian_to_joint_velocity(self, cart_vel):
        """Convert Cartesian velocity to joint velocities using Jacobian."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)

        J = jacp[:, 6:13]
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.damping ** 2 * np.eye(3))

        return J_pinv @ cart_vel
