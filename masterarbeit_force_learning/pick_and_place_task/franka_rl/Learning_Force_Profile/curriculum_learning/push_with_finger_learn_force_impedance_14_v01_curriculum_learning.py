"""
Panda Push Environment - Direction Vector Concept

Works for ALL trajectories: straight, curved, diagonal, s_curve

Key changes from previous version:
1. Increased correction gains to handle curved trajectories
2. Full 2D forward + correction vectors (not just Y component)
3. Hand follows bottle along ANY trajectory shape
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os

from matplotlib.style.core import available


class PandaPushTrajectoryEnv(gym.Env):
    """
    Direction Vector concept:
    - Hand stays behind bottle
    - Push = Forward + K * Correction
    - K (stiffness) is learned by RL
    - Works for ALL trajectory types!
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_type="straight"):
        super().__init__()

        # Load model
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "../robot_panda_push_force.xml")
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
        self.base_forward_speed = 0.012  # Slightly reduced for better control
        self.behind_distance = 0.04  # Distance behind bottle

        # ==================== WRIST ROTATION ====================
        # robot's initial position behind the bottle
        # at this position (wrist = 0), gripper -X points in -Y direction
        # This is the default push direction for straight trajectory
        self.gripper_push_angle_at_home = -np.pi/2 # -90 degrees

        self.wrist_rotation_gain = 5.0
        # self.max_wrist_rotation = 0.4  # ~23 degrees
        self.max_wrist_rotation = 1.57  # ~90 degrees
        self.wrist_offset = 0.0
        self.base_wrist_pos = 0.0

        # ==================== STIFFNESS (What RL learns!) ====================
        self.K_min = 100.0
        self.K_max = 500.0
        self.current_K = np.array([300.0, 300.0])

        # ==================== CORRECTION GAINS ====================
        # These need to be higher for curved trajectories
        self.base_correction_gain = 0.8# 0.5  # Increased from 0.25 # Base correction strength
        self.k_correction_gain = 0.8# 0.6  # Increased from 0.35 # Additional correction from k
        self.max_correction_speed = 0.05# 0.035  # Increased from 0.020 # max correction velocity

        # ==================== TILT SAFETY ====================
        self.tilt_ok = 0.99 # Full speed
        self.tilt_slow = 0.98 # half speed
        self.tilt_stop = 0.96 # stop and settle

        self.is_settling = False
        self.settle_counter = 0
        self.settle_required = 25

        # ==================== Cartesian control ==============
        self.target_z = 0.92 # Hand Height (gripper contacts bottle ~0.1m below)
        self.z_gain = 10.0
        self.damping = 0.01

        # Trajectory
        self.trajectory_type = trajectory_type
        self.trajectory = None
        self.total_arc_length = 0.0
        self.path_tolerance = 0.05
        self.goal_position = np.array([0.4, -0.2]) # fixed goal for all trajectories

        # Phase
        self.in_approach = True

        # Action space: [forward_mod, lateral_mod, Kx, Ky]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space
        obs_dim = 36
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ================= Curriculum Learning ================
        self.episode_count = 0
        self.current_traj_type = "straight"

        # curriculum schedule (episode_threshold: available_trajectories)
        self.curriculum_schedule = {
            0: ["straight"], # Episodes 0 - 999
            1000: ["straight", "curved"], # Episodes 1000 - 2999
            3000: ["straight", "curved", "s_curve"], # Episodes 3000+
        }

        # Statistics tracking
        self.trajectory_stats = {
            "straight": {"attempts": 0, "successes":0},
            "curved": {"attempts": 0, "successes": 0},
            "s_curve": {"attempts": 0, "successes": 0},
        }

        # ================= State ==============================
        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0
        self.max_episode_length = 2500
        self.prev_progress = 0.0

        # --------------- Logging -----------------------------------
        self.force_profile_log = []
        self.stiffness_profile_log = []
        self.deviation_log = []
        self.correction_log = []

        print(f"\n{'=' * 60}")
        print("DIRECTION VECTOR CONCEPT - ALL TRAJECTORIES")
        print(f"{'=' * 60}")
        print("Push = Forward_vector + K * Correction_vector")
        print(f"K range: [{self.K_min}, {self.K_max}] N/m")
        print(f"Base correction gain: {self.base_correction_gain}")
        print(f"K correction gain: {self.k_correction_gain}")
        print(f"Max correction speed: {self.max_correction_speed * 1000:.1f} mm/s")
        print(f"Wrist rotation: max {np.degrees(self.max_wrist_rotation):.1f}°")
        print(f"Gripper -X at home points: {np.degrees(self.gripper_push_angle_at_home):.1f}°")
        print(f"Goal position: {self.goal_position}")
        print(f"{'=' * 60}")

    # ==================================================================
    #           CORE CONCEPT: Direction Vectors
    # ==================================================================

    def _get_forward_vector(self, bottle_xy):
        """
        Forward vector: Direction along trajectory towards goal.
        This is the tangent to the trajectory at bottle's closest point.
        returns 2D unit vector pointing along trajectory
        """
        return self._get_path_tangent(bottle_xy)

    def _get_correction_vector(self, bottle_xy):
        """
        Correction vector: Points from bottle TOWARDS trajectory.
        Magnitude = deviation distance (how far bottle is from path)
        reutrns (correction_vec, correction_magnitude)
        """
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        return deviation_vec, deviation_mag

    def _get_hand_target_position(self, bottle_xy):
        """
        Hand stays BEHIND the bottle relative to the push direction.
        Push direction = Forward + Correction (weighted by k)
        returns: 3D target position [x, y, z] for hand
        """
        forward_vec = self._get_forward_vector(bottle_xy)
        correction_vec, correction_mag = self._get_correction_vector(bottle_xy)

        K_normalized = np.mean(self.current_K) / self.K_max

        if correction_mag > 0.001:
            correction_dir = correction_vec / correction_mag
            # Higher weight for correction when deviation is large
            correction_weight = min(correction_mag * 15 * K_normalized, 0.7)  # Increased from 10, 0.6
            combined_dir = (1 - correction_weight) * forward_vec + correction_weight * correction_dir
            combined_dir = combined_dir / (np.linalg.norm(combined_dir) + 1e-6)
        else:
            combined_dir = forward_vec

        hand_xy = bottle_xy - combined_dir * self.behind_distance

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
        MAIN CONCEPT: Push = Forward + K * Correction

        Forward: Along trajectory tangent (follows the small line segments) - Both forward and correction are FULL 2D vectors!
        Correction: Towards trajectory (brings bottle back when deviated) - This allows following ANY trajectory shape.

        Both are FULL 2D vectors - works for ANY trajectory shape!
        """
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        tilt = self._get_bottle_tilt()
        is_touching = self._is_touching()

        # Get direction vectors
        forward_vec = self._get_forward_vector(bottle_xy)
        correction_vec, correction_mag = self._get_correction_vector(bottle_xy)

        # Parse action
        forward_mod = action[0]

        # STIFFNESS from action
        self.current_K[0] = self.K_min + action[2] * (self.K_max - self.K_min)
        self.current_K[1] = self.K_min + action[3] * (self.K_max - self.K_min)
        K_avg = np.mean(self.current_K)

        # ==================== TILT SAFETY ====================
        if self.is_settling:
            if self._check_stable():
                self.settle_counter += 1
                if self.settle_counter >= self.settle_required:
                    self.is_settling = False
                    self.settle_counter = 0
            else:
                self.settle_counter = 0
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

        # ==================== COMPUTE PUSH VELOCITY ====================
        v_desired = np.zeros(3)

        # ==================== CONTACT LOST RECOVERY ====================
        if not is_touching:
            # PRIORITY: Get back to bottle!
            bottle_direction = bottle_xy - hand_pos[:2]
            bottle_dist = np.linalg.norm(bottle_direction)

            if bottle_dist > 0.02:  # More than 2cm away from bottle
                # Move DIRECTLY towards bottle at high speed
                bottle_dir_normalized = bottle_direction / bottle_dist
                recovery_speed = min(0.06, bottle_dist * 2.0)  # Up to 6cm/s

                v_desired[0] = bottle_dir_normalized[0] * recovery_speed
                v_desired[1] = bottle_dir_normalized[1] * recovery_speed
                v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

                # Debug
                if self.episode_length % 100 == 0:
                    print(
                        f"    CONTACT LOST! Moving to bottle: dist={bottle_dist * 100:.1f}cm, speed={recovery_speed * 100:.1f}cm/s")

                return v_desired

        # ==================== NORMAL PUSH (when touching) ====================

        # 1. FORWARD COMPONENT: Along trajectory tangent (FULL 2D!)
        forward_speed = self.base_forward_speed * (1.0 + forward_mod * 0.3) * speed_mult
        v_forward = forward_vec * forward_speed

        # 2. CORRECTION COMPONENT: Towards trajectory (FULL 2D!)
        if correction_mag > 0.001:
            correction_dir = correction_vec / correction_mag

            # Base + K-dependent correction
            base_correction = self.base_correction_gain * correction_mag
            k_correction = (K_avg / self.K_max) * self.k_correction_gain * correction_mag
            correction_speed = (base_correction + k_correction) * speed_mult

            # Reduce correction when tilting (but not as aggressively)
            if tilt < 0.98:  # Changed from 0.99
                tilt_factor = max((tilt - 0.94) / (0.98 - 0.94), 0.2)  # Min 20%, was 10%
                correction_speed *= tilt_factor
            else:
                tilt_factor = 1.0

            # Cap correction speed
            correction_speed = min(correction_speed, self.max_correction_speed)

            v_correction = correction_dir * correction_speed

            # Debug output
            if self.episode_length % 100 == 0 and correction_mag > 0.01:
                print(f"    Correction: base={base_correction * 1000:.1f} + K={k_correction * 1000:.1f} "
                      f"x tilt_factor={tilt_factor:.2f} = {correction_speed * 1000:.1f} mm/s")
                print(f"    forward_vec={forward_vec}, correction_dir={correction_dir}")
        else:
            v_correction = np.zeros(2)

        # 3. STAY BEHIND BOTTLE (tracking)
        target_pos = self._get_hand_target_position(bottle_xy)
        pos_error = target_pos[:2] - hand_pos[:2]
        v_tracking = pos_error * 4.0 # 2.5  # Increased from 2.0
        v_tracking = np.clip(v_tracking, -0.04, 0.04) # -0.025, 0.025  # Increased from 0.02

        # Combine all components (FULL 2D!)
        v_desired[0] = v_forward[0] + v_correction[0] + v_tracking[0]
        v_desired[1] = v_forward[1] + v_correction[1] + v_tracking[1]

        # Limit total velocity
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.05:  # Increased from 0.04
            v_desired[:2] = v_desired[:2] / v_mag * 0.05

        # 4. Height control
        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        # Log for visualization
        self.correction_log.append({
            'deviation': correction_mag,
            'K': K_avg,
            'correction_speed': np.linalg.norm(v_correction),
            'forward_speed': np.linalg.norm(v_forward)
        })

        return v_desired

    def _get_bottle_tilt(self):
        """Get bottle tilt (1.0 = upright, <1.0 = tilting)"""
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        return bottle_mat[2, 2]

    def _check_stable(self):
        """Check if bottle is stable (upright and not moving)"""
        tilt = self._get_bottle_tilt()
        angvel = np.linalg.norm(self.data.qvel[3:6])
        return tilt > 0.99 and angvel < 0.1

    def _cartesian_to_joint_velocity(self, cart_vel):
        """Convert Cartesian velocity to joint velocitites using Jacobina."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)

        J = jacp[:, 6:13]
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.damping ** 2 * np.eye(3))

        return J_pinv @ cart_vel

    def _compute_wrist_rotation(self, bottle_xy):
        """
        Align gripper's -X axis (push direction) with trajectory tangent.

        At home position (wrist=0):
        - Gripper +X points in +Y direction
        - Gripper -X points in -Y direction (push direction for straight trajectory)

        For curved trajectories, wrist rotates to keep -X aligned with tangent.
        """
        forward_vec = self._get_forward_vector(bottle_xy)

        # Angle of tangent (desired push direction) in world frame
        tangent_angle = np.arctan2(forward_vec[1], forward_vec[0])

        # At home (wrist=0), gripper -X points in -Y direction (angle = -90°)
        # This is stored in self.gripper_push_angle_at_home = -π/2

        # Rotation needed to align gripper -X with tangent
        alignment_rotation = tangent_angle - self.gripper_push_angle_at_home

        # Normalize to [-π, π]
        while alignment_rotation > np.pi:
            alignment_rotation -= 2 * np.pi
        while alignment_rotation < -np.pi:
            alignment_rotation += 2 * np.pi

        # Additional correction rotation when bottle deviates
        correction_vec, deviation_mag = self._get_correction_vector(bottle_xy)
        correction_rotation = 0.0
        if deviation_mag > 0.005:
            K_avg = np.mean(self.current_K)
            K_factor = K_avg / self.K_max
            correction_rotation = correction_vec[0] * K_factor * self.wrist_rotation_gain * 0.5

        # Combine alignment and correction
        target_rotation = alignment_rotation + correction_rotation

        # Clamp to joint limits
        target_rotation = np.clip(target_rotation, -self.max_wrist_rotation, self.max_wrist_rotation)

        # Debug output
        if self.episode_length % 100 == 0:
            print(f"    Wrist: tangent={np.degrees(tangent_angle):.1f}°, "
                  f"align={np.degrees(alignment_rotation):.1f}°, "
                  f"corr={np.degrees(correction_rotation):.1f}°, "
                  f"total={np.degrees(target_rotation):.1f}°")

        return target_rotation


    # ==================================================================
    #                      TRAJECTORY
    # ==================================================================

    def _generate_trajectory(self, bottle_start_xy, traj_type):
        """
        Generate trajectory from bottle start position to fixed goal.

        All trajectories:
        - Start a bottle_start_xy
        - End at self.goal_position (0.4, -0.2)
        - Consist of 50 points
        """
        start = bottle_start_xy.copy()
        n_points: int = 50
        goal = self.goal_position.copy()

        if traj_type == "straight":
            # Linear interpolation from start to goal
            t = np.linspace(0,1,n_points)
            traj = np.outer(1-t, start) + np.outer(t, goal)
            return traj.astype(np.float32)

        elif traj_type == "curved":
            # Quadratic Beizer curve: start -> mid_point -> goal
            t = np.linspace(0,1, n_points)
            mid_point = np.array([0.55, 0.0])

            # Quadratic Bezier formula: P = (1-t)²·start + 2(1-t)t·mid + t²·goal
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid_point[0] + t ** 2 * goal[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid_point[1] + t ** 2 * goal[1]
            return np.stack([x, y], axis=1).astype(np.float32)

        elif traj_type == "s_curve":
            # s-curve from start to goal with sinusoidal X offset that fades out
            t = np.linspace(0, 1, n_points)

            # S-shape: wiggles left-right but ends at goal with sinusoidal X-offset
            # Wiggle fades in and out (zero at start and end)
            wiggle_strength = np.sin(np.pi * t)  # Peaks at middle
            x = start[0] + 0.08 * np.sin(2 * np.pi * t) * wiggle_strength
            y = start[1] + t * (goal[1] - start[1])

            return np.stack([x, y], axis=1).astype(np.float32)

        else:
            # Default: straight line to goal
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1-t, start) + np.outer(t, goal)
            return traj.astype(np.float32)

    def _compute_arc_length(self):
        """Compute total arc length of trajectory."""
        if self.trajectory is None or len(self.trajectory) < 2:
            self.total_arc_length = 0.0
            return
        diffs = np.diff(self.trajectory, axis=0)
        self.total_arc_length = np.sum(np.linalg.norm(diffs, axis=1))

    def _get_closest_point_on_trajectory(self, pos_xy):
        """
        Find closest point on trajectory to given position.

        Returns: (index, closest_point, arc_length_to_point)
        """
        if self.trajectory is None:
            return 0, pos_xy.copy(), 0.0
        distances = np.linalg.norm(self.trajectory - pos_xy, axis=1)
        idx = np.argmin(distances)
        closest_pt = self.trajectory[idx].copy()
        arc_len = np.sum(np.linalg.norm(np.diff(self.trajectory[:idx + 1], axis=0), axis=1)) if idx > 0 else 0.0
        return idx, closest_pt, arc_len

    def _get_path_deviation(self, bottle_xy):
        """
        Get deviation vector from bottle to trajectory.

        Returns: (deviation_vector, deviation_magnitude)
        - deviation_vector: points FROM bottle TO trajectory
        - deviation_magnitude: distance from trajectory
        """
        _, closest_pt, _ = self._get_closest_point_on_trajectory(bottle_xy)
        deviation_vec = closest_pt - bottle_xy
        return deviation_vec.astype(np.float32), float(np.linalg.norm(deviation_vec))

    def _get_path_tangent(self, bottle_xy):
        """
        Get tangent vector at bottle's closest point on trajectory.

        This is the direction the bottle should move - one of the 49 small
        line segments that make up the trajectory.

        Returns: 2D unit vector pointing along trajectory towards goal
        """
        if self.trajectory is None or len(self.trajectory) < 2:
            return np.array([0.0, -1.0], dtype=np.float32)
        idx, _, _ = self._get_closest_point_on_trajectory(bottle_xy)

        # Tangent = vector from current point to next point
        if idx < len(self.trajectory) - 1:
            tangent = self.trajectory[idx + 1] - self.trajectory[idx]
        else:
            tangent = self.trajectory[idx] - self.trajectory[idx - 1]

        # Normalize to unit vector
        norm = np.linalg.norm(tangent)
        return (tangent / norm).astype(np.float32) if norm > 1e-6 else np.array([0.0, -1.0], dtype=np.float32)

    def _get_progress(self, bottle_xy):
        """
        Get progress along trajectory (0.0 = start, 1.0 = goal).
        """
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
    #                      CURRICULUM LEARNING
    # ==================================================================
    def _get_curriculum_trajectories(self):
        """ Get available trajectories based on episode count."""
        available = ["straight"]
        for threshold, trajectories in sorted(self.curriculum_schedule.items()):
            if self.episode_count >= threshold:
                available = trajectories
        return available

    def _select_trajectory_type(self, options=None):
        """Select trajectory type based on mode and options."""
        # Override from options (for testing)
        if options and "trajectory_type" in options:
            return options["trajectory_type"]

        # Curriculum learning
        if self.trajectory_type == "curriculum":
            available = self._get_curriculum_trajectories()
            return np.random.choice(available)

        # Mixed (random from all)
        if self.trajectory_type == "mixed":
            return np.random.choice(["straight", "curved", "s_curve"])

        # Fixed trajectory type
        return self.trajectory_type

    def get_curriculum_stage(self):
        """Get current curriculum stage for logging."""
        available = self._get_curriculum_trajectories()
        if available == ["straight"]:
            return "Stage 1: Straight only"
        elif available == ["straight", "curved"]:
            return "Stage 2: Straight + Curved"
        else:
            return "Stage 3: All trajectories"

    def print_statistics(self):
        """Print training statistics."""
        print(f"\n{'=' * 50}")
        print("TRAINING STATISTICS")
        print(f"{'=' * 50}")
        print(f"Total episodes: {self.episode_count}")
        print(f"Current stage: {self.get_curriculum_stage()}")
        print("\nSuccess rates:")
        for traj_type, data in self.trajectory_stats.items():
            attempts = data["attempts"]
            successes = data["successes"]
            rate = (successes / attempts * 100) if attempts > 0 else 0
            print(f"  {traj_type:10s}: {successes:4d}/{attempts:4d} ({rate:5.1f}%)")
        print(f"{'=' * 50}")

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

        forward_vec = self._get_forward_vector(bottle_xy)
        correction_vec, correction_mag = self._get_correction_vector(bottle_xy)

        tangent = self._get_path_tangent(bottle_xy)
        progress = self._get_progress(bottle_xy)
        contact_force = self._get_contact_force()
        is_touching = np.array([1.0 if self._is_touching() else 0.0], dtype=np.float32)
        tilt = self._get_bottle_tilt()

        K_normalized = (self.current_K - self.K_min) / (self.K_max - self.K_min)
        settling_flag = np.array([1.0 if self.is_settling else 0.0], dtype=np.float32)

        obs = np.concatenate([
            qpos,  # 7
            qvel,  # 7
            hand_pos,  # 3
            bottle_pos,  # 3
            dir_to_bottle,  # 2
            [dist_to_bottle],  # 1
            correction_vec,  # 2
            [correction_mag],  # 1
            tangent,  # 2
            [progress],  # 1
            contact_force,  # 3
            is_touching,  # 1
            K_normalized,  # 2
            settling_flag,  # 1
        ])

        return obs.astype(np.float32)

    # ==================================================================
    #                      REWARD
    # ==================================================================

    def _get_reward(self):
        """compute reward for RL agent."""
        info = {"is_success": False}

        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        hand_pos = self.data.xpos[self.hand_body_id]

        correction_vec, deviation_mag = self._get_correction_vector(bottle_xy)
        progress = self._get_progress(bottle_xy)
        tilt = self._get_bottle_tilt()
        is_touching = self._is_touching()
        force = self._get_contact_force()
        force_mag = np.linalg.norm(force)
        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)

        if self.in_approach:
            # Approach phase: reward for getting close to bottle
            total_reward = -2.0 * dist_to_bottle
            total_reward += -5.0 * abs(hand_pos[2] - self.target_z)
            if is_touching or dist_to_bottle < 0.06:
                total_reward += 10.0
                self.in_approach = False

            if self.episode_length % 50 == 0:
                print(f"Step {self.episode_length} [APPROACH]: dist={dist_to_bottle:.3f}m")
        else:
            #push phase: reward for progress, low deviation, stability
            # Progress reward (moving along trajectory)
            progress_delta = progress - self.prev_progress
            r_progress = 100.0 * max(progress_delta, 0)

            # Deviation reward - LINEAR (continuous feedback)
            max_dev_reward = 5.0
            dev_slope = 100.0
            r_deviation = max_dev_reward - dev_slope * deviation_mag
            r_deviation = max(r_deviation, -15.0)

            # Stability reward (keep bottle upright)
            if tilt > 0.995:
                r_stability = 3.0
            elif tilt > 0.99:
                r_stability = 1.0
            elif tilt > 0.98:
                r_stability = 0.0
            else:
                r_stability = -15.0 * (1 - tilt)

            # Contact reward (maintain contact)
            # r_contact = 0.5 if is_touching else -0.5 * dist_to_bottle
            # r_contact = 1.0 if is_touching else -2.0 * dist_to_bottle

            # ==================== CONTACT REWARD - STRONGER ====================
            if is_touching:
                r_contact = 2.0  # Reward for maintaining contact
            else:
                # Strong penalty for losing contact
                r_contact = -10.0 * dist_to_bottle

                # Extra penalty if very far from bottle
                if dist_to_bottle > 0.08:
                    r_contact -= 5.0

            total_reward = r_progress + r_deviation + r_stability + r_contact

            # Print status
            if self.episode_length % 50 == 0:
                K_avg = np.mean(self.current_K)
                mode = "SETTLE" if self.is_settling else "PUSH"
                contact_status = "CONTACT" if is_touching else "NO CONTACT!"
                wrist_deg = np.degrees(self.wrist_offset)
                side = "LEFT" if correction_vec[0] > 0.005 else "RIGHT" if correction_vec[0] < -0.005 else "CENTER"
                print(f"Step {self.episode_length} [{mode}]: "
                      f"prog={progress:.1%}, dev={deviation_mag:.3f}m ({side}), "
                      f"K={K_avg:.0f}, F={force_mag:.1f}N, tilt={tilt:.4f}, "
                      f"wrist={wrist_deg:.1f}°, {contact_status}")

        #small time penalty
        total_reward -= 0.005

        # Success bonu
        if progress > 0.95 and deviation_mag < self.path_tolerance:
            total_reward += 100.0
            info["is_success"] = True
            print(f"SUCCESS at step {self.episode_length}!")

        # Failure penalties
        if tilt < 0.5:
            total_reward -= 100.0
            info["bottle_fallen"] = True

        if deviation_mag > 0.15:
            total_reward -= 50.0
            info["off_path"] = True

        self.prev_progress = progress

        info["progress"] = progress
        info["deviation"] = deviation_mag
        info["force_magnitude"] = force_mag
        info["bottle_tilt"] = tilt
        info["stiffness"] = np.mean(self.current_K)
        info["trajectory_type"] = self.current_traj_type

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

        # ===== CURRICULUM: SELECT TRAJECTORY TYPE =====
        traj_type = self._select_trajectory_type(options)
        self.current_traj_type = traj_type

        # Update statistics
        self.trajectory_stats[traj_type]["attempts"] += 1

        self.trajectory = self._generate_trajectory(bottle_start[:2], traj_type)
        self._compute_arc_length()

        # Reset state variables
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
        self.correction_log = []

        # ===== INCREMENT EPISODE COUNTER =====
        self.episode_count += 1

        # Print episode info
        print(f"\n{'=' * 50}")
        print(f"EPISODE {self.episode_count}: {traj_type.upper()}")
        print(f"Bottle: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"Goal: ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})")
        print(f"Arc length: {self.total_arc_length:.3f}m")
        if self.trajectory_type == "curriculum":
            print(f"Stage: {self.get_curriculum_stage()}")
        print(f"{'=' * 50}")

        # Print statistics every 100 episodes
        if self.episode_count % 100 == 0:
            self.print_statistics()

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
            target_wrist_rotation = self._compute_wrist_rotation(bottle_xy)

        q_dot = self._cartesian_to_joint_velocity(cart_vel)

        dt = 0.02

        self.current_qpos_target[:6] = np.clip(
            self.current_qpos_target[:6] + q_dot[:6] * dt,
            self.act_low[:6], self.act_high[:6]
        )

        wrist_error = target_wrist_rotation - self.wrist_offset
        wrist_speed = 2.0 * wrist_error
        wrist_speed = np.clip(wrist_speed, -0.5, 0.5)

        self.wrist_offset += wrist_speed * dt
        self.wrist_offset = np.clip(self.wrist_offset, -self.max_wrist_rotation, self.max_wrist_rotation)

        self.current_qpos_target[6] = np.clip(
            self.base_wrist_pos + self.wrist_offset,
            self.act_low[6], self.act_high[6]
        )

        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        force = self._get_contact_force()
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        _, dev = self._get_path_deviation(bottle_xy)

        self.force_profile_log.append(force.copy())
        self.stiffness_profile_log.append(self.current_K.copy())
        self.deviation_log.append(dev)

        obs = self._get_obs()
        reward, info = self._get_reward()
        self.episode_length += 1

        # ===== TRACK SUCCESS STATISTICS =====
        if info.get("is_success"):
            self.trajectory_stats[self.current_traj_type]["successes"] += 1

        terminated = bool(info.get("is_success") or info.get("bottle_fallen") or info.get("off_path"))
        truncated = bool(self.episode_length >= self.max_episode_length)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None