"""
Panda Push Environment - Lookahead Target Concept (Supervisor's Method)

Key Concept:
    - Push direction = Vector from bottle to LOOKAHEAD TARGET (3-4 points ahead)
    - When ON trajectory: lookahead direction ≈ tangent (forward motion)
    - When OFF trajectory: lookahead direction = correction + forward (automatic blend!)
    - Minimize angle θ between Force P and correction vector
    - No separate forward/correction weighting needed - geometry handles it!

Updates:
    - Blended push direction (tangent when on-track, correction when off-track)
    - Hand repositions to SIDE of bottle when deviation is large
    - Wrist aligns with push direction (hand position handles correction geometry)
    - Faster wrist response
    - 200 trajectory points for finer resolution
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os


class PandaPushTrajectoryEnv(gym.Env):
    """
    Lookahead Target Concept (Supervisor's Method):
        - Push direction = direction to lookahead target (blended with tangent)
        - Hand repositions to side of bottle for correction
        - Wrist aligns with push direction
        - Automatically blends forward + correction based on geometry
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

        # IDs
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

        # Control
        self.actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = self.actuator_ranges[:, 0]
        self.act_high = self.actuator_ranges[:, 1]
        self.current_qpos_target = np.zeros(7)

        # ==================== PUSH PARAMETERS ====================
        self.base_forward_speed = 0.015  # Base push speed
        self.behind_distance = 0.04  # Distance hand stays behind bottle

        # ==================== LOOKAHEAD PARAMETERS ====================
        self.lookahead_points = 4  # Look 4 points ahead on trajectory

        # ==================== WRIST ROTATION ====================
        self.gripper_push_angle_at_home = -np.pi / 2  # -90 degrees
        self.max_wrist_rotation = 2.5 # Allow more rotation
        self.wrist_offset = 0.0
        self.base_wrist_pos = 0.0

        # ==================== STIFFNESS (What RL learns!) ====================
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
        self.target_z = 0.92
        self.z_gain = 10.0
        self.damping = 0.01

        # ==================== TRAJECTORY ====================
        self.trajectory_type = trajectory_type
        self.trajectory = None
        self.total_arc_length = 0.0
        self.path_tolerance = 0.05
        self.goal_position = np.array([0.4, -0.2])
        # self.goal_position = np.array([0.4, -0.4])

        # Phase
        self.in_approach = True

        # Action space: [forward_mod, lateral_mod, Kx, Ky]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space
        obs_dim = 39  # Added: lookahead_direction (2) + angle_error (1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State
        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0
        self.max_episode_length = 2500
        self.prev_progress = 0.0

        # Logging
        self.force_profile_log = []
        self.stiffness_profile_log = []
        self.deviation_log = []
        self.angle_error_log = []

        print(f"\n{'=' * 60}")
        print("LOOKAHEAD TARGET CONCEPT (SUPERVISOR'S METHOD)")
        print(f"{'=' * 60}")
        print("Push direction = Blended tangent + correction to LOOKAHEAD target")
        print("Hand repositions to SIDE of bottle for correction")
        print(f"Lookahead points: {self.lookahead_points}")
        print(f"K range: [{self.K_min}, {self.K_max}] N/m")
        print(f"Goal position: {self.goal_position}")
        print(f"{'=' * 60}")

    # ==================================================================
    #           CORE CONCEPT: Lookahead Target + Blended Direction
    # ==================================================================

    def _get_lookahead_target(self, bottle_xy):
        """
        Get target point 3-4 points AHEAD of bottle's closest point on trajectory.
        This gives the bottle time to return to trajectory smoothly.

        Returns: target_point (2D), target_index
        """
        if self.trajectory is None or len(self.trajectory) < 2:
            return self.goal_position.copy(), 0

        # Find bottle's closest point on trajectory
        idx, _, _ = self._get_closest_point_on_trajectory(bottle_xy)

        # Look ahead by lookahead_points
        target_idx = min(idx + self.lookahead_points, len(self.trajectory) - 1)
        target_point = self.trajectory[target_idx].copy()

        # Near end of trajectory: use goal directly as target
        # this ensures correction still works even at the end
        if target_idx >= len(self.trajectory) - 2:
            target_point = self.goal_position.copy()

        return target_point, target_idx

    def _get_push_direction(self, bottle_xy):
        """
        BLENDED push direction:
        - When ON trajectory (small deviation): push along TANGENT
        - When OFF trajectory (large deviation): push toward LOOKAHEAD TARGET
        - Smooth blend prevents overcorrection oscillation

        Returns: (push_direction, distance_to_target, target_point)
        """
        target_point, _ = self._get_lookahead_target(bottle_xy)
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)

        # Vector from bottle to target
        to_target = target_point - bottle_xy
        distance = np.linalg.norm(to_target)

        if distance > 1e-6:
            correction_dir = to_target / distance
        else:
            # Already at target, use tangent
            correction_dir = self._get_path_tangent(bottle_xy)

        # pure tangent direction
        tangent_dir = self._get_path_tangent(bottle_xy)

        # Blend factor: 0 = pure tangent, 1 = full correction
        # Below 0.5cm: mostly tangent (just push forward)
        # Above 2.5cm: mostly correction ( push back to trajectory)
        blend = np.clip((deviation_mag - 0.005) / 0.02, 0.0, 1.0)

        push_direction = (1.0 - blend) * tangent_dir + blend * correction_dir

        # Normalize
        norm = np.linalg.norm(push_direction)
        if norm > 1e-6:
            push_direction = push_direction/norm
        else:
            push_direction = tangent_dir

        return push_direction.astype(np.float32), float(distance), target_point

    def _get_angle_error(self, bottle_xy):
        """
        Compute angle θ between current force direction P and desired push direction.

        Goal: Minimize this angle!

        Returns: angle_error in radians (0 = perfect alignment)
        """
        # Get current force direction
        contact_force = self._get_contact_force()
        force_mag = np.linalg.norm(contact_force[:2])

        # Get desired push direction
        push_dir, _, _ = self._get_push_direction(bottle_xy)

        if force_mag > 0.5:  # Only compute if significant force
            P_normalized = -contact_force[:2] / force_mag
            dot_product = np.clip(np.dot(P_normalized, push_dir), -1.0, 1.0)
            angle_error = np.arccos(dot_product)
        else:
            angle_error = 0.0

        return float(angle_error)

    def _get_hand_target_position(self, bottle_xy):
        """
        Hand position strategy:
        - Small deviation: behind bottle along push direction (normal pushing)
        - Large deviation: shift toward the side AWAY from trajectory
            so the hand can push bottle BACK toward trajectory

        The hand repositioning is what creates the corrective force, NOT the wrist rotation.

        Returns: 3D target position [x, y, z] for hand
        """
        push_dir, _, _ = self._get_push_direction(bottle_xy)
        # deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)

        # Base position: behind bottle along push direction
        hand_xy = bottle_xy - push_dir * self.behind_distance

        # # When deviation is significant, offset hand toward the outside
        # # deviation_vec points FROM bottle TO trajectory
        # # We want the hand on the opposite side (away from trajectory)
        # if deviation_mag > 0.005:
        #     away_from_traj = -deviation_vec / (deviation_mag + 1e-6)
        #
        #     # How much to offset: grows with deviation
        #     side_blend = np.clip(deviation_mag / 0.04, 0.0, 1.0)
        #     side_offset = away_from_traj * self.behind_distance * side_blend
        #
        #     hand_xy += side_offset

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

    def _compute_push_velocity(self, action):
        """
        SUPERVISOR'S METHOD:
        - Push direction = direction to lookahead target
        - Automatically blends forward + correction based on geometry
        - Hand tracks behind bottle relative to push direction
        """
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        tilt = self._get_bottle_tilt()
        is_touching = self._is_touching()

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
                v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])
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
                v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

                if self.episode_length % 50 == 0:
                    print(f"    CONTACT LOST! Recovery: dist={bottle_dist * 100:.1f}cm")

                return v_desired

        # ==================== PUSH USING LOOKAHEAD TARGET ====================
        # Get push direction (automatically blends forward + correction!)
        push_dir, dist_to_target, target_point = self._get_push_direction(bottle_xy)

        # Calculate push speed (RL can modulate)
        base_speed = self.base_forward_speed * (1.0 + forward_mod * 0.3) * speed_mult

        # K can also modulate speed slightly
        k_factor = 0.8 + 0.4 * (K_avg / self.K_max)  # 0.8 to 1.2
        push_speed = base_speed * k_factor

        # Push velocity in direction of lookahead target
        v_push = push_dir * push_speed

        # ==================== TRACKING (stay behind bottle) ====================
        target_hand_pos = self._get_hand_target_position(bottle_xy)
        pos_error = target_hand_pos[:2] - hand_pos[:2]
        v_tracking = pos_error * 3.0
        v_tracking = np.clip(v_tracking, -0.03, 0.03)

        # Combine push + tracking
        v_desired[0] = v_push[0] + v_tracking[0]
        v_desired[1] = v_push[1] + v_tracking[1]

        # Limit total velocity
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.05:
            v_desired[:2] = v_desired[:2] / v_mag * 0.05

        # Height control
        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        # ==================== LOGGING ====================
        angle_error = self._get_angle_error(bottle_xy)
        self.angle_error_log.append(angle_error)

        # Debug output
        if self.episode_length % 100 == 0:
            _, dev_mag = self._get_path_deviation(bottle_xy)
            _, target_idx = self._get_lookahead_target(bottle_xy)
            angle_deg = np.degrees(angle_error)
            _, deviation_mag = self._get_path_deviation(bottle_xy)
            blend = np.clip((deviation_mag - 0.005) / 0.02, 0.0, 1.0)
            print(f"    Lookahead: idx={target_idx}, push_dir=[{push_dir[0]:.2f}, {push_dir[1]:.2f}], "
                  f"θ={angle_deg:.1f}°, dev={dev_mag * 100:.1f}cm, blend={blend:.2f}")

        return v_desired

    def _get_bottle_tilt(self):
        """Get bottle tilt (1.0 = upright, <1.0 = tilting)"""
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        return bottle_mat[2, 2]

    def _check_stable(self):
        """Check if bottle is stable (upright and not moving)"""
        tilt = self._get_bottle_tilt()
        angvel = np.linalg.norm(self.data.qvel[3:6])
        return tilt > 0.97 and angvel < 0.1

    def _cartesian_to_joint_velocity(self, cart_vel):
        """Convert Cartesian velocity to joint velocities using Jacobian."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)

        J = jacp[:, 6:13]
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.damping ** 2 * np.eye(3))

        return J_pinv @ cart_vel

    # def _compute_wrist_rotation(self, bottle_xy):
    #     """
    #     Align gripper's -X axis with PUSH DIRECTION (to lookahead target).
    #     This minimizes angle θ between force P and desired direction!
    #     """
    #     # Get push direction (to lookahead target)
    #     push_dir, _, _ = self._get_push_direction(bottle_xy)
    #
    #     # Angle of push direction in world frame
    #     push_angle = np.arctan2(push_dir[1], push_dir[0])
    #
    #     # Rotation needed to align gripper -X with push direction
    #     alignment_rotation = push_angle - self.gripper_push_angle_at_home
    #
    #     # Normalize to [-π, π]
    #     while alignment_rotation > np.pi:
    #         alignment_rotation -= 2 * np.pi
    #     while alignment_rotation < -np.pi:
    #         alignment_rotation += 2 * np.pi
    #
    #     # Clamp to joint limits
    #     target_rotation = np.clip(alignment_rotation, -self.max_wrist_rotation, self.max_wrist_rotation)
    #
    #     # Debug output
    #     if self.episode_length % 100 == 0:
    #         print(f"    Wrist: push_angle={np.degrees(push_angle):.1f}°, "
    #               f"rotation={np.degrees(target_rotation):.1f}°")
    #
    #     return target_rotation

    # ==================================================================
    #                      TRAJECTORY
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
            mid_point = np.array([0.55, 0.0])
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid_point[0] + t ** 2 * goal[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid_point[1] + t ** 2 * goal[1]
            return np.stack([x, y], axis=1).astype(np.float32)

        elif traj_type == "s_curve":
            t = np.linspace(0, 1, n_points)
            wiggle_strength = np.sin(np.pi * t)  # Peaks at middle, zero at ends
            # x = start[0] + 0.08 * np.sin(2 * np.pi * t) * wiggle_strength
            x = start[0] + 0.04 * np.sin(2 * np.pi * t) * wiggle_strength
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
        """Find closest point on trajectory to given position."""
        if self.trajectory is None:
            return 0, pos_xy.copy(), 0.0
        distances = np.linalg.norm(self.trajectory - pos_xy, axis=1)
        idx = np.argmin(distances)
        closest_pt = self.trajectory[idx].copy()
        arc_len = np.sum(np.linalg.norm(np.diff(self.trajectory[:idx + 1], axis=0), axis=1)) if idx > 0 else 0.0
        return idx, closest_pt, arc_len

    def _get_path_deviation(self, bottle_xy):
        """Get deviation vector from bottle to trajectory."""
        _, closest_pt, _ = self._get_closest_point_on_trajectory(bottle_xy)
        deviation_vec = closest_pt - bottle_xy
        return deviation_vec.astype(np.float32), float(np.linalg.norm(deviation_vec))

    def _get_path_tangent(self, bottle_xy):
        """Get tangent vector at bottle's closest point on trajectory."""
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
        """Get progress along trajectory (0.0 = start, 1.0 = goal)."""
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

    # ==================================================================
    #                      OBSERVATION
    # ==================================================================

    def _get_obs(self):
        """
        Observation includes:
        - Robot state (qpos, qvel)
        - Positions (hand, bottle)
        - Push direction (to lookahead target)
        - Deviation from trajectory
        - Angle error θ between force and push direction
        - Contact info
        """
        qpos = self.data.qpos[7:14].astype(np.float32)
        qvel = self.data.qvel[6:13].astype(np.float32)
        hand_pos = self.data.xpos[self.hand_body_id].astype(np.float32)
        bottle_pos = self.data.xpos[self.bottle_body_id].astype(np.float32)
        bottle_xy = bottle_pos[:2]

        # Direction to bottle
        hand_to_bottle = bottle_pos[:2] - hand_pos[:2]
        dist_to_bottle = np.linalg.norm(hand_to_bottle)
        dir_to_bottle = hand_to_bottle / (dist_to_bottle + 1e-6)

        # Push direction (to lookahead target) - THE KEY OBSERVATION!
        push_dir, dist_to_target, _ = self._get_push_direction(bottle_xy)

        # Deviation from trajectory
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)

        # Angle error between force and push direction
        angle_error = self._get_angle_error(bottle_xy)

        # Progress
        progress = self._get_progress(bottle_xy)

        # Contact
        contact_force = self._get_contact_force()
        is_touching = np.array([1.0 if self._is_touching() else 0.0], dtype=np.float32)

        # Stiffness
        K_normalized = (self.current_K - self.K_min) / (self.K_max - self.K_min)
        settling_flag = np.array([1.0 if self.is_settling else 0.0], dtype=np.float32)

        wrist_normalized = np.array([self.wrist_offset / self.max_wrist_rotation], dtype=np.float32)

        obs = np.concatenate([
            qpos,  # 7
            qvel,  # 7
            hand_pos,  # 3
            bottle_pos,  # 3
            dir_to_bottle,  # 2
            [dist_to_bottle],  # 1
            push_dir,  # 2  (NEW: direction to lookahead target)
            [dist_to_target],  # 1  (NEW: distance to lookahead target)
            deviation_vec,  # 2
            [deviation_mag],  # 1
            [angle_error],  # 1  (NEW: angle θ between force and push dir)
            [progress],  # 1
            contact_force,  # 3
            is_touching,  # 1
            K_normalized,  # 2
            settling_flag,  # 1
            wrist_normalized, #1
        ])  # Total: 39

        return obs.astype(np.float32)

    # ==================================================================
    #                      REWARD
    # ==================================================================

    def _get_reward(self):
        """
        Reward function:
        - Progress along trajectory
        - Low deviation from trajectory
        - Small angle error θ (force aligned with push direction)
        - Maintain contact
        - Keep bottle upright
        """
        info = {"is_success": False}

        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        hand_pos = self.data.xpos[self.hand_body_id]

        # Get metrics
        _, deviation_mag = self._get_path_deviation(bottle_xy)
        progress = self._get_progress(bottle_xy)
        tilt = self._get_bottle_tilt()
        is_touching = self._is_touching()
        force = self._get_contact_force()
        force_mag = np.linalg.norm(force)
        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)
        angle_error = self._get_angle_error(bottle_xy)

        if self.in_approach:
            # Approach phase
            total_reward = -2.0 * dist_to_bottle
            total_reward += -5.0 * abs(hand_pos[2] - self.target_z)
            if is_touching or dist_to_bottle < 0.06:
                total_reward += 10.0
                self.in_approach = False

            if self.episode_length % 50 == 0:
                print(f"Step {self.episode_length} [APPROACH]: dist={dist_to_bottle:.3f}m")
        else:
            # Push phase

            # 1. Progress reward
            progress_delta = progress - self.prev_progress
            r_progress = 100.0 * max(progress_delta, 0)

            # 2. Deviation reward (stay on trajectory)
            max_dev_reward = 5.0
            dev_slope = 100.0
            r_deviation = max_dev_reward - dev_slope * deviation_mag
            r_deviation = max(r_deviation, -15.0)

            # 3. Angle error reward (NEW: minimize θ)
            # Small angle = good, large angle = bad
            angle_deg = np.degrees(angle_error)
            if angle_deg < 10:
                r_angle = 2.0  # Well aligned
            elif angle_deg < 30:
                r_angle = 1.0  # Acceptable
            elif angle_deg < 60:
                r_angle = 0.0  # Needs improvement
            else:
                r_angle = -2.0  # Poorly aligned

            # 4. Stability reward
            if tilt > 0.995:
                r_stability = 3.0
            elif tilt > 0.99:
                r_stability = 1.0
            elif tilt > 0.98:
                r_stability = 0.0
            else:
                r_stability = -15.0 * (1 - tilt)

            # 5. Contact reward (STRONGER)
            if is_touching:
                r_contact = 2.0
            else:
                r_contact = -10.0 * dist_to_bottle
                if dist_to_bottle > 0.08:
                    r_contact -= 5.0

            total_reward = r_progress + r_deviation + r_angle + r_stability + r_contact

            # Print status
            if self.episode_length % 50 == 0:
                K_avg = np.mean(self.current_K)
                mode = "SETTLE" if self.is_settling else "PUSH"
                contact_status = "CONTACT" if is_touching else "NO CONTACT!"
                print(f"Step {self.episode_length} [{mode}]: "
                      f"prog={progress:.1%}, dev={deviation_mag * 100:.1f}cm, "
                      f"θ={angle_deg:.1f}°, K={K_avg:.0f}, F={force_mag:.1f}N, "
                      f"tilt={tilt:.4f}, {contact_status}")

        # Time penalty
        total_reward -= 0.005

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

        self.force_profile_log = []
        self.stiffness_profile_log = []
        self.deviation_log = []
        self.angle_error_log = []

        print(f"\n{'=' * 50}")
        print(f"EPISODE: {traj_type}")
        print(f"Bottle start: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"Goal: ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})")
        print(f"Trajectory arc length: {self.total_arc_length:.3f}m")
        print(f"Lookahead points: {self.lookahead_points}")
        print(f"{'=' * 50}")

        return self._get_obs(), {}

    def step(self, action):
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        if self.in_approach:
            cart_vel, dist = self._compute_approach_velocity()
            target_wrist_rotation = 0.0
            if dist < 0.06 or self._is_touching():
                self.in_approach = False
                print(f"Step {self.episode_length}: → PUSH phase")
        else:
            cart_vel = self._compute_push_velocity(action)
            # target_wrist_rotation = self._compute_wrist_rotaiton()
            target_wrist_rotation = action[1] * self.max_wrist_rotation # RL controls wrist

        q_dot = self._cartesian_to_joint_velocity(cart_vel)

        dt = 0.02

        self.current_qpos_target[:6] = np.clip(
            self.current_qpos_target[:6] + q_dot[:6] * dt,
            self.act_low[:6], self.act_high[:6]
        )

        wrist_error = target_wrist_rotation - self.wrist_offset
        wrist_speed = 5.0 * wrist_error # before 2.0
        # wrist_speed = np.clip(wrist_speed, -0.5, 0.5)
        wrist_speed = np.clip(wrist_speed, -2.0, 2.0)

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
        reward, info = self._get_reward()
        self.episode_length += 1

        terminated = bool(info.get("is_success") or info.get("bottle_fallen") or info.get("off_path"))
        truncated = bool(self.episode_length >= self.max_episode_length)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # Install the custom drawing callback
                self.viewer.user_scn.ngeom = 0  # reset custom geoms

            # Draw trajectory
            if self.trajectory is not None:
                self.viewer.user_scn.ngeom = 0  # clear previous frame's geoms
                for i in range(len(self.trajectory) - 1):
                    if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                        break

                    p1 = np.array([self.trajectory[i][0], self.trajectory[i][1], 0.801])
                    p2 = np.array([self.trajectory[i + 1][0], self.trajectory[i + 1][1], 0.801])

                    mujoco.mjv_initGeom(
                        self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=[0.003, 0, 0],  # radius
                        pos=(p1 + p2) / 2,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32)
                    )
                    # Orient capsule from p1 to p2
                    mujoco.mjv_connector(
                        self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        0.003,  # width
                        p1, p2
                    )
                    self.viewer.user_scn.ngeom += 1

            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None