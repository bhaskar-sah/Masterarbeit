"""
Panda Push Environment with Cartesian Space + Impedance Control

"Augmenting Pose Trajectories with Learned Force Profiles"

Action Space: Cartesian position/velocity + Impedance (stiffness)
- RL learns WHERE to push (Cartesian velocity: vx, vy)
- RL learns HOW HARD to push (Force or Stiffness: Fx, Fy OR Kx, Ky)

The impedance controller implements:
    F = K * (x_desired - x_actual) + D * (v_desired - v_actual)

Where K (stiffness) is learned by the RL agent:
- High K → Stiff, tracks position precisely
- Low K  → Compliant, adapts to contact forces

This allows the RL agent to learn the FORCE PROFILE needed to:
1. Push the bottle along the trajectory
2. Correct deviations by applying appropriate forces
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os

from numpy.ma.testutils import approx


class PandaPushTrajectoryEnv(gym.Env):
    """
    Environment with Cartesian Space + Impedance Control.

    Action Space (5D):
        - vx, vy: Desired Cartesian velocity (where to push)
        - Kx, Ky, Kz: Stiffness in each direction (how stiff/compliant)

    The RL agent learns BOTH:
        1. The pushing direction (velocity)
        2. The force profile (via impedance/stiffness)

    This is the hybrid position/force control scheme
    for learning force profiles in manipulation tasks.

    Here first  - approach phase (move to bottle)
                - push phase (push along trajectory with impedance control)
    """

    # This tellls the user (and the code) which visulaization methods the environment supports.
    # "human": for interactive display - opens GUI
    # "rgb_array": This mode runs "headless". Returns a numpy array representing the image
    # pixels (frames) of the current state. Userful for recording videos of the agent.
    # 30 fps: human render mode. this speed ensures that, it does not run too slow or run too fast.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_type="straight", control_mode="impedance"):
        super().__init__()

        # ==================== LOAD MUJOCO MODEL ====================
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "robot_panda_push_force.xml")
        xml_path = os.path.abspath(xml_path)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find XML file at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ==================== BODY IDs ====================
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

        # ==================== ROBOT CONTROL ====================
        self.actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = self.actuator_ranges[:, 0]
        self.act_high = self.actuator_ranges[:, 1]
        self.current_qpos_target = np.zeros(7)

        # ==================== CONTROL MODE ====================
        self.control_mode = control_mode

        # ==================== IMPEDANCE CONTROL PARAMETERS ====================
        # Stiffness bounds (N/m)
        self.K_min = 50.0  # Very compliant
        self.K_max = 2000.0  # Very stiff

        # Damping ratio (for stability)
        self.damping_ratio = 0.7

        # Current stiffness (will be set by RL)
        self.current_K = np.array([500.0, 500.0, 1000.0])  # [Kx, Ky, Kz]

        # Default desired force (N) - for force control mode
        self.F_max = 20.0  # Maximum force command

        # ==================== CARTESIAN CONTROL ====================
        self.target_z = 0.93  # Fixed height
        self.max_cart_vel = 0.05
        self.z_gain = 10.0
        self.damping = 0.01

        # ==================== APPROACH PARAMETERS ====================
        self.approach_offset = 0.05  # Start pushing from 5cm behind bottle
        self.contact_threshold = 0.08  # Distance to consider "close enough"

        # ==================== TRAJECTORY ====================
        self.trajectory_type = trajectory_type
        self.trajectory = None
        self.total_arc_length = 0.0
        self.path_tolerance = 0.05
        self.goal_pos = None

        # ==================== ACTION SPACE ====================
        # [vx, vy, Kx, Ky]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ==================== OBSERVATION SPACE ====================
        # Extended observation includes current stiffness
        # [qpos(7), qvel(7), hand_pos(3), bottle_pos(3), deviation(2),
        #  tangent(2), progress(1), contact_force(3), is_touching(1),
        #  current_stiffness(2)]
        obs_dim = 7 + 7 + 3 + 3 + 2 + 2 + 2 + 1 + 3 + 1 + 2  # = 33
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ==================== STATE ====================
        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0
        self.max_episode_length = 1000
        self.prev_progress = 0.0
        self.contact_made = False

        # For logging learned force profile
        self.in_approach_phase = True  # Start in approach phase
        self.force_profile_log = []
        self.stiffness_profile_log = []

        print(f"\n{'=' * 60}")
        print("Environment: Cartesian + Impedance Control")
        print(f"{'=' * 60}")
        print(f"Action: [vx, vy, Kx, Ky]")
        print(f"Approach offset: {self.approach_offset}m")
        print(f"Target Z: {self.target_z}")
        print(f"{'=' * 60}")

    # ==================================================================
    #           APPROACH PHASE - Move to bottle
    # ==================================================================
    def _get_approach_target(self):
        """
        Get the position where the hand should go to start pushing.
        This is slightly behind the bottle (in the push direction).
        """
        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]

        # Get push direction (trajectory tangent at start)
        tangent = self._get_path_tangent(bottle_xy)

        # Approach position: behind the bottle, opposite to push direction
        approach_xy = bottle_xy - tangent * self.approach_offset
        approach_z = self.target_z

        return np.array([approach_xy[0], approach_xy[1], approach_z])

    def _compute_approach_velocity(self):
        """
        Compute velocity to move hand towards the bottle.
        Returns Cartesian velocity [vx, vy, vz].
        """
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]

        # Target: slightly behind bottle at push height
        target_pos = self._get_approach_target()

        # Direction to target
        error = target_pos - hand_pos
        distance = np.linalg.norm(error[:2]) # XY distance

        # Proportional control with velocity limit
        approach_gain = 2.0
        v_desired = approach_gain * error

        # Limit velocity
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > self.max_cart_vel * 2:
            v_desired[:2] = v_desired[:2] / v_mag * self.max_cart_vel * 2

        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        return v_desired, distance

    def _check_approach_complete(self):
        pass#

    # ==================================================================
    #           IMPEDANCE CONTROLLER
    # ==================================================================

    def _impedance_control(self, desired_vel_xy, stiffness_xy):
        """
        Impedance control in Cartesian space.

        Implements: F = K * (x_d - x) + D * (v_d - v)

        Where:
            K: Stiffness (learned by RL)
            D: Damping (computed from K for stability)
            x_d: Desired position (from trajectory)
            v_d: Desired velocity (from RL action)

        The stiffness K determines the force-position trade-off:
            High K → Track position, resist disturbances
            Low K  → Compliant, adapt to contact
        """
        hand_pos = self.data.xpos[self.hand_body_id]
        hand_vel = self.data.cvel[self.hand_body_id][3:6]  # Linear velocity

        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]

        # Get trajectory information
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)

        # === Compute desired Cartesian force ===

        # 1. Velocity component (pushing direction from RL)
        F_vel = np.zeros(3)
        F_vel[0] = stiffness_xy[0] * desired_vel_xy[0] * 0.1  # Scale velocity to force
        F_vel[1] = stiffness_xy[1] * desired_vel_xy[1] * 0.1

        # 2. Position correction component (bring bottle back to trajectory)
        # The stiffness determines how aggressively we correct deviations
        # This is the KEY part for learning force profiles!
        if deviation_mag > 0.01:
            # Direction to correct: from bottle towards trajectory
            correction_dir = deviation_vec / (deviation_mag + 1e-6)

            # Force magnitude depends on stiffness and deviation
            # High stiffness = stronger correction force
            F_correction = np.zeros(3)
            F_correction[0] = stiffness_xy[0] * deviation_vec[0]
            F_correction[1] = stiffness_xy[1] * deviation_vec[1]
        else:
            F_correction = np.zeros(3)

        # 3. Height maintenance (always stiff in Z)
        z_error = self.target_z - hand_pos[2]
        F_z = self.current_K[2] * z_error

        # 4. Damping (for stability)
        D_xy = 2 * self.damping_ratio * np.sqrt(stiffness_xy)
        F_damping = np.zeros(3)
        F_damping[0] = -D_xy[0] * hand_vel[0]
        F_damping[1] = -D_xy[1] * hand_vel[1]

        # Total desired Cartesian force
        F_desired = np.array([
            F_vel[0] + F_correction[0] + F_damping[0],
            F_vel[1] + F_correction[1] + F_damping[1],
            F_z
        ])

        # Convert Cartesian force to joint torques using Jacobian transpose
        # τ = J^T @ F
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)
        J = jacp[:, 6:13]

        tau = J.T @ F_desired

        return tau, F_desired

    def _cartesian_to_joint_with_impedance(self, desired_vel_xy, stiffness_xy):
        """
        Convert Cartesian velocity to joint velocity, with impedance behavior.

        This is a simplified version that:
        1. Uses the Jacobian for velocity conversion
        2. Applies stiffness-weighted position correction
        3. Maintains Z height
        """
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]

        # Get deviation from trajectory
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)

        # === Build desired Cartesian velocity ===

        # 1. Base velocity from RL action
        v_desired = np.zeros(3)
        v_desired[0] = desired_vel_xy[0]
        v_desired[1] = desired_vel_xy[1]

        # 2. Add correction velocity weighted by stiffness
        # Normalize stiffness to [0, 1] range for weighting
        K_normalized = (stiffness_xy - self.K_min) / (self.K_max - self.K_min)

        if deviation_mag > 0.01:
            # Correction velocity: move hand to push bottle back to trajectory
            correction_scale = 0.5 * K_normalized  # Higher K = stronger correction
            v_correction = correction_scale * deviation_vec * 10  # Scale up
            v_desired[0] += v_correction[0]
            v_desired[1] += v_correction[1]

        # 3. Z velocity for height maintenance
        z_error = self.target_z - hand_pos[2]
        v_desired[2] = self.z_gain * z_error

        # Limit velocities
        v_desired = np.clip(v_desired, -0.1, 0.1)

        # === Convert to joint velocities ===
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)
        J = jacp[:, 6:13]

        # Damped pseudo-inverse
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.damping ** 2 * np.eye(3))

        q_dot = J_pinv @ v_desired

        return q_dot

    # ==================================================================
    #                      TRAJECTORY GENERATION
    # ==================================================================

    def _generate_trajectory(self, bottle_start_xy, traj_type):
        """Generate trajectory for the bottle to follow."""
        start = bottle_start_xy.copy()
        n_points = 50

        if traj_type == "straight":
            end = start + np.array([0.0, -0.4])
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, end)
        elif traj_type == "straight_short":
            end = start + np.array([0.0, -0.25])
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, end)
        elif traj_type == "diagonal":
            end = start + np.array([0.15, -0.3])
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, end)
        elif traj_type == "curved":
            t = np.linspace(0, np.pi / 2, n_points)
            radius = 0.2
            x = start[0] + radius * np.sin(t)
            y = start[1] - radius * (1 - np.cos(t))
            traj = np.stack([x, y], axis=1)
        elif traj_type == "s_curve":
            t = np.linspace(0, 1, n_points)
            x = start[0] + 0.08 * np.sin(2 * np.pi * t)
            y = start[1] - 0.35 * t
            traj = np.stack([x, y], axis=1)
        else:
            end = start + np.array([0.0, -0.3])
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, end)

        return traj.astype(np.float32)

    def _compute_arc_length(self):
        if self.trajectory is None or len(self.trajectory) < 2:
            self.total_arc_length = 0.0
            return
        diffs = np.diff(self.trajectory, axis=0)
        self.total_arc_length = np.sum(np.linalg.norm(diffs, axis=1))

    # ==================================================================
    #                      TRAJECTORY QUERIES
    # ==================================================================

    def _get_closest_point_on_trajectory(self, pos_xy):
        if self.trajectory is None:
            return 0, pos_xy.copy(), 0.0
        distances = np.linalg.norm(self.trajectory - pos_xy, axis=1)
        idx = np.argmin(distances)
        closest_pt = self.trajectory[idx].copy()
        if idx == 0:
            arc_len = 0.0
        else:
            arc_len = np.sum(np.linalg.norm(np.diff(self.trajectory[:idx + 1], axis=0), axis=1))
        return idx, closest_pt, arc_len

    def _get_path_deviation(self, bottle_xy):
        _, closest_pt, _ = self._get_closest_point_on_trajectory(bottle_xy)
        deviation_vec = closest_pt - bottle_xy
        deviation_mag = np.linalg.norm(deviation_vec)
        return deviation_vec.astype(np.float32), float(deviation_mag)

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
            tangent = tangent / norm
        else:
            tangent = np.array([0.0, -1.0])
        return tangent.astype(np.float32)

    def _get_progress(self, bottle_xy):
        if self.total_arc_length < 1e-6:
            return 0.0
        _, _, arc_len = self._get_closest_point_on_trajectory(bottle_xy)
        return float(np.clip(arc_len / self.total_arc_length, 0.0, 1.0))

    # ==================================================================
    #                      CONTACT DETECTION
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
        """Check if robot is in contact with bottle."""
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
        """Build observation (31 dims including current stiffness)."""
        qpos = self.data.qpos[7:14].astype(np.float32)
        qvel = self.data.qvel[6:13].astype(np.float32)
        hand_pos = self.data.xpos[self.hand_body_id].astype(np.float32)
        bottle_pos = self.data.xpos[self.bottle_body_id].astype(np.float32)
        bottle_xy = bottle_pos[:2]

        deviation_vec, _ = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)
        progress = self._get_progress(bottle_xy)
        contact_force = self._get_contact_force()
        is_touching = np.array([1.0 if self._is_touching() else 0.0], dtype=np.float32)

        # Normalized current stiffness
        K_normalized = (self.current_K[:2] - self.K_min) / (self.K_max - self.K_min)

        obs = np.concatenate([
            qpos,  # 7
            qvel,  # 7
            hand_pos,  # 3
            bottle_pos,  # 3
            deviation_vec,  # 2
            tangent,  # 2
            [progress],  # 1
            contact_force,  # 3
            is_touching,  # 1
            K_normalized,  # 2 (current stiffness)
        ])

        return obs.astype(np.float32)

    # ==================================================================
    #                      REWARD
    # ==================================================================

    def _get_reward(self):
        """Reward function for force profile learning."""
        info = {"is_success": False}

        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]
        hand_pos = self.data.xpos[self.hand_body_id]

        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)
        progress = self._get_progress(bottle_xy)

        is_touching = self._is_touching()
        contact_force = self._get_contact_force()
        force_mag = np.linalg.norm(contact_force)

        if is_touching:
            self.contact_made = True

        # ===== REWARD COMPONENTS =====

        # 1. Progress reward
        progress_delta = progress - self.prev_progress
        contact_multiplier = 1.0 if is_touching else 0.0
        r_progress = 100.0 * progress_delta * contact_multiplier

        # 2. Deviation penalty
        if deviation_mag < self.path_tolerance:
            r_deviation = 1.0
        else:
            r_deviation = -15.0 * deviation_mag

        # 3. Contact reward
        if is_touching:
            r_contact = 2.0

            # Force alignment
            force_xy = contact_force[:2]
            force_norm = np.linalg.norm(force_xy)
            if force_norm > 0.5:
                force_dir = force_xy / force_norm
                alignment = np.dot(force_dir, tangent)
                r_force_align = 3.0 * max(0, alignment)
            else:
                r_force_align = 0.0

            # Reward for appropriate force magnitude (not too much, not too little)
            ideal_force = 5.0  # N
            force_error = abs(force_mag - ideal_force)
            r_force_mag = -0.5 * force_error
        else:
            dist_xy = np.linalg.norm(hand_pos[:2] - bottle_xy)
            r_contact = -1.0 * dist_xy
            r_force_align = 0.0
            r_force_mag = 0.0

        # 4. Stability
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        bottle_upright = bottle_mat[2, 2]
        if bottle_upright < 0.9:
            r_stability = -10.0 * (1.0 - bottle_upright)
        else:
            r_stability = 0.0

        # 5. Stiffness regularization (encourage reasonable stiffness values)
        K_mean = np.mean(self.current_K[:2])
        if K_mean < 100 or K_mean > 1500:
            r_stiffness = -0.1
        else:
            r_stiffness = 0.0

        # Total reward
        total_reward = (
                r_progress +
                r_deviation +
                r_contact +
                r_force_align +
                r_force_mag +
                r_stability +
                r_stiffness
        )

        total_reward -= 0.05  # Time penalty

        # Success
        if progress > 0.95 and deviation_mag < self.path_tolerance:
            total_reward += 50.0
            info["is_success"] = True
            print(f"SUCCESS at step {self.episode_length}!")

        # Failures
        if bottle_upright < 0.5:
            total_reward -= 50.0
            info["bottle_fallen"] = True

        if deviation_mag > 0.2:
            total_reward -= 50.0
            info["off_path"] = True

        self.prev_progress = progress

        # Info for logging
        info["progress"] = progress
        info["deviation"] = deviation_mag
        info["force_magnitude"] = force_mag
        info["is_touching"] = is_touching
        info["hand_z"] = hand_pos[2]
        info["stiffness_x"] = self.current_K[0]
        info["stiffness_y"] = self.current_K[1]

        # Debug
        if self.episode_length % 100 == 0:
            print(f"Step {self.episode_length}: "
                  f"prog={progress:.1%}, "
                  f"dev={deviation_mag:.3f}m, "
                  f"F={force_mag:.1f}N, "
                  f"K=[{self.current_K[0]:.0f},{self.current_K[1]:.0f}]")

        return total_reward, info

    # ==================================================================
    #                      RESET
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

        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_start = self.data.xpos[self.bottle_body_id].copy()

        traj_type = self.trajectory_type
        if options and "trajectory_type" in options:
            traj_type = options["trajectory_type"]

        self.trajectory = self._generate_trajectory(bottle_start[:2], traj_type)
        self._compute_arc_length()

        self.goal_pos = np.array([
            self.trajectory[-1, 0],
            self.trajectory[-1, 1],
            bottle_start[2]
        ], dtype=np.float32)

        # Reset state
        self.prev_progress = 0.0
        self.contact_made = False
        self.episode_length = 0
        self.current_K = np.array([500.0, 500.0, 1000.0])

        # Clear logs
        self.force_profile_log = []
        self.stiffness_profile_log = []

        print(f"\n{'=' * 50}")
        print(f"NEW EPISODE - {traj_type}")
        print(f"Bottle: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"Hand Z: {hand_pos[2]:.3f} (target: {self.target_z})")
        print(f"{'=' * 50}")

        return self._get_obs(), {}

    # ==================================================================
    #                      STEP
    # ==================================================================

    def step(self, action):
        """
        Execute one step with impedance control.

        Action: [vx, vy, Kx, Ky] or [vx, vy, Fx, Fy]
        """
        # Parse action
        desired_vel_xy = action[:2] * self.max_cart_vel

        if self.control_mode == "impedance":
            # Map [0, 1] to [K_min, K_max]
            self.current_K[0] = self.K_min + action[2] * (self.K_max - self.K_min)
            self.current_K[1] = self.K_min + action[3] * (self.K_max - self.K_min)
            stiffness_xy = self.current_K[:2]
        else:
            # Force mode: map [-1, 1] to [-F_max, F_max]
            desired_force_xy = action[2:4] * self.F_max
            # Convert force to equivalent stiffness (simplified)
            stiffness_xy = np.array([500.0, 500.0])  # Fixed for force mode

        # Convert to joint velocities using impedance control
        q_dot = self._cartesian_to_joint_with_impedance(desired_vel_xy, stiffness_xy)

        # Integrate to position
        dt = 0.02
        self.current_qpos_target = np.clip(
            self.current_qpos_target + q_dot * dt,
            self.act_low,
            self.act_high
        )
        self.data.ctrl[:7] = self.current_qpos_target

        # Step simulation
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Log force profile
        contact_force = self._get_contact_force()
        self.force_profile_log.append(contact_force.copy())
        self.stiffness_profile_log.append(self.current_K[:2].copy())

        obs = self._get_obs()
        reward, info = self._get_reward()

        self.episode_length += 1

        terminated = (
                info.get("is_success", False) or
                info.get("bottle_fallen", False) or
                info.get("off_path", False)
        )
        truncated = self.episode_length >= self.max_episode_length

        return obs, reward, terminated, truncated, info

    # ==================================================================
    #                      RENDER & UTILITY
    # ==================================================================

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None