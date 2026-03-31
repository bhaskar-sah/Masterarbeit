"""
Panda Push Environment with PROPER Deviation Correction

FIXED: The robot now pushes the bottle BACK to the trajectory when it deviates.

The key insight:
- When bottle deviates LEFT of trajectory → Hand must push it RIGHT (back to path)
- When bottle deviates RIGHT of trajectory → Hand must push it LEFT (back to path)

The correction force/velocity must be applied in the direction FROM bottle TO trajectory.
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os


class PandaPushTrajectoryEnv(gym.Env):
    """
    Environment with proper deviation correction.
    Robot pushes bottle back to trajectory when it drifts.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_type="straight",
                 control_mode="impedance"):
        super().__init__()

        # ==================== LOAD MUJOCO MODEL ====================
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "../robot_panda_push_force.xml")
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

        # ==================== IMPEDANCE PARAMETERS ====================
        self.K_min = 100.0  # Minimum stiffness
        self.K_max = 2000.0  # Maximum stiffness
        self.current_K = np.array([500.0, 500.0, 1000.0])

        # ==================== CARTESIAN CONTROL ====================
        self.target_z = 0.93
        self.max_cart_vel = 0.05
        self.z_gain = 10.0
        self.damping = 0.01

        # ==================== APPROACH PARAMETERS ====================
        self.approach_offset = 0.05
        self.contact_threshold = 0.08

        # ==================== TRAJECTORY ====================
        self.trajectory_type = trajectory_type
        self.trajectory = None
        self.total_arc_length = 0.0
        self.path_tolerance = 0.05
        self.goal_pos = None

        # ==================== ACTION SPACE ====================
        # [vx, vy, Kx, Ky] - Cartesian velocity + Stiffness
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ==================== OBSERVATION SPACE ====================
        obs_dim = 7 + 7 + 3 + 3 + 2 + 1 + 2 + 2 + 1 + 3 + 1 + 2  # = 34
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
        self.in_approach_phase = True

        self.force_profile_log = []
        self.stiffness_profile_log = []

        print(f"\n{'=' * 60}")
        print("Environment: Impedance Control with DEVIATION CORRECTION")
        print(f"{'=' * 60}")

    # ==================================================================
    #           APPROACH PHASE
    # ==================================================================

    def _get_approach_target(self):
        """Position behind bottle to start pushing."""
        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]
        tangent = self._get_path_tangent(bottle_xy)

        # Behind bottle, opposite to push direction
        approach_xy = bottle_xy - tangent * self.approach_offset
        return np.array([approach_xy[0], approach_xy[1], self.target_z])

    def _compute_approach_velocity(self):
        """Velocity to move towards bottle."""
        hand_pos = self.data.xpos[self.hand_body_id]
        target_pos = self._get_approach_target()

        error = target_pos - hand_pos
        distance = np.linalg.norm(error[:2])

        approach_gain = 2.0
        v_desired = approach_gain * error

        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > self.max_cart_vel * 2:
            v_desired[:2] = v_desired[:2] / v_mag * self.max_cart_vel * 2

        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        return v_desired, distance

    def _check_approach_complete(self):
        """Check if close enough to start pushing."""
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]

        dist_xy = np.linalg.norm(hand_pos[:2] - bottle_pos[:2])
        height_ok = abs(hand_pos[2] - self.target_z) < 0.03

        return dist_xy < self.contact_threshold and height_ok

    # ==================================================================
    #           PUSH PHASE WITH DEVIATION CORRECTION
    # ==================================================================

    def _compute_push_velocity(self, action):
        """
        Compute velocity for pushing with PROPER deviation correction.

        Key insight:
        - deviation_vec points FROM bottle TO closest point on trajectory
        - To correct, hand must push bottle IN THE DIRECTION of deviation_vec
        - So hand must be on the OPPOSITE side and push towards trajectory
        """
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]

        # Get trajectory info
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)

        # Parse action
        desired_vel_xy = action[:2] * self.max_cart_vel

        # Get stiffness from action
        self.current_K[0] = self.K_min + action[2] * (self.K_max - self.K_min)
        self.current_K[1] = self.K_min + action[3] * (self.K_max - self.K_min)

        # Normalize stiffness for correction weighting
        K_normalized = (self.current_K[:2] - self.K_min) / (self.K_max - self.K_min)
        K_avg = np.mean(K_normalized)

        # === BUILD PUSH VELOCITY ===
        v_desired = np.zeros(3)

        # 1. BASE PUSH: Along trajectory tangent (forward progress)
        #    RL action modulates this
        push_along_tangent = tangent * (0.02 + desired_vel_xy[1] * 0.03)  # Base + RL
        v_desired[0] += push_along_tangent[0]
        v_desired[1] += push_along_tangent[1]

        # 2. DEVIATION CORRECTION: Push bottle back to trajectory
        #    This is the KEY fix!
        if deviation_mag > 0.01:
            # deviation_vec points from bottle to trajectory
            # To push bottle towards trajectory, hand must push in direction of deviation_vec
            # Hand needs to be positioned to push in that direction

            # Correction direction (normalized)
            if deviation_mag > 0.001:
                correction_dir = deviation_vec / deviation_mag
            else:
                correction_dir = np.zeros(2)

            # Correction strength based on:
            # - How far off path (deviation_mag)
            # - How stiff we want to be (K_normalized) - HIGHER K = STRONGER CORRECTION
            correction_strength = K_avg * 0.3 * min(deviation_mag * 10, 1.0)

            # Add correction velocity
            # Hand moves to push bottle towards trajectory
            v_correction = correction_dir * correction_strength
            v_desired[0] += v_correction[0]
            v_desired[1] += v_correction[1]

            # Debug
            if self.episode_length % 50 == 0 and deviation_mag > 0.02:
                print(f"  CORRECTION: dev={deviation_mag:.3f}, K_avg={K_avg:.2f}, "
                      f"correction=[{v_correction[0]:.3f}, {v_correction[1]:.3f}]")

        # 3. STAY BEHIND BOTTLE: Hand should stay behind bottle (relative to push direction)
        hand_to_bottle = bottle_xy - hand_pos[:2]

        # Position where hand should be: behind bottle
        ideal_hand_pos = bottle_xy - tangent * 0.05  # 5cm behind
        pos_error = ideal_hand_pos - hand_pos[:2]

        # Add velocity to maintain position behind bottle
        v_desired[0] += pos_error[0] * 2.0
        v_desired[1] += pos_error[1] * 2.0

        # 4. HEIGHT: Maintain pushing height
        z_error = self.target_z - hand_pos[2]
        v_desired[2] = self.z_gain * z_error

        # Clip velocities
        v_desired = np.clip(v_desired, -0.15, 0.15)

        return v_desired

    def _cartesian_to_joint_velocity(self, cart_vel):
        """Convert Cartesian velocity to joint velocity."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)
        J = jacp[:, 6:13]

        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.damping ** 2 * np.eye(3))

        q_dot = J_pinv @ cart_vel
        return q_dot

    # ==================================================================
    #                      TRAJECTORY
    # ==================================================================

    def _generate_trajectory(self, bottle_start_xy, traj_type):
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
        """
        Get deviation vector FROM bottle TO trajectory.
        This tells us which direction to push the bottle to correct.
        """
        _, closest_pt, _ = self._get_closest_point_on_trajectory(bottle_xy)
        deviation_vec = closest_pt - bottle_xy  # Points towards trajectory!
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
    #                      CONTACT
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

    # ==================================================================
    #                      OBSERVATION
    # ==================================================================

    def _get_obs(self):
        qpos = self.data.qpos[7:14].astype(np.float32)
        qvel = self.data.qvel[6:13].astype(np.float32)
        hand_pos = self.data.xpos[self.hand_body_id].astype(np.float32)
        bottle_pos = self.data.xpos[self.bottle_body_id].astype(np.float32)
        bottle_xy = bottle_pos[:2]

        # Direction and distance to bottle
        hand_to_bottle = bottle_pos[:2] - hand_pos[:2]
        dist_to_bottle = np.linalg.norm(hand_to_bottle)
        if dist_to_bottle > 0.001:
            dir_to_bottle = hand_to_bottle / dist_to_bottle
        else:
            dir_to_bottle = np.zeros(2)

        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)
        progress = self._get_progress(bottle_xy)
        contact_force = self._get_contact_force()
        is_touching = np.array([1.0 if self._is_touching() else 0.0], dtype=np.float32)

        K_normalized = (self.current_K[:2] - self.K_min) / (self.K_max - self.K_min)

        obs = np.concatenate([
            qpos,  # 7
            qvel,  # 7
            hand_pos,  # 3
            bottle_pos,  # 3
            dir_to_bottle,  # 2
            [dist_to_bottle],  # 1
            deviation_vec,  # 2
            tangent,  # 2
            [progress],  # 1
            contact_force,  # 3
            is_touching,  # 1
            K_normalized,  # 2
        ])  # Total: 34

        return obs.astype(np.float32)

    # ==================================================================
    #                      REWARD
    # ==================================================================

    def _get_reward(self):
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

        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)

        if is_touching:
            self.contact_made = True
            self.in_approach_phase = False

        # ===== REWARD =====

        if self.in_approach_phase:
            # APPROACH: Get to bottle
            r_approach = -2.0 * dist_to_bottle
            height_error = abs(hand_pos[2] - self.target_z)
            r_height = -5.0 * height_error
            r_contact_bonus = 10.0 if is_touching else 0.0

            total_reward = r_approach + r_height + r_contact_bonus

            if self.episode_length % 50 == 0:
                print(f"Step {self.episode_length} [APPROACH]: "
                      f"dist={dist_to_bottle:.3f}m, Z={hand_pos[2]:.3f}")
        else:
            # PUSH: Progress + Stay on path
            progress_delta = progress - self.prev_progress
            r_progress = 150.0 * max(progress_delta, 0)  # Increased reward

            # DEVIATION: Strong penalty for leaving path, bonus for staying
            if deviation_mag < 0.02:
                r_deviation = 2.0  # Good - on path
            elif deviation_mag < self.path_tolerance:
                r_deviation = 0.5  # OK - within tolerance
            else:
                r_deviation = -20.0 * deviation_mag  # Bad - off path

            # CONTACT
            if is_touching:
                r_contact = 1.0

                # Force alignment with tangent (pushing in right direction)
                force_xy = contact_force[:2]
                force_norm = np.linalg.norm(force_xy)
                if force_norm > 0.5:
                    force_dir = force_xy / force_norm
                    alignment = np.dot(force_dir, tangent)
                    r_force_align = 2.0 * max(0, alignment)
                else:
                    r_force_align = 0.0

                # CORRECTION REWARD: If deviating, reward force towards trajectory
                if deviation_mag > 0.02 and force_norm > 0.5:
                    correction_dir = deviation_vec / (deviation_mag + 1e-6)
                    correction_alignment = np.dot(force_dir, correction_dir)
                    r_correction = 3.0 * max(0, correction_alignment)
                else:
                    r_correction = 0.0
            else:
                r_contact = -2.0 * dist_to_bottle
                r_force_align = 0.0
                r_correction = 0.0

            # Stability
            bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
            bottle_upright = bottle_mat[2, 2]
            r_stability = -10.0 * (1.0 - bottle_upright) if bottle_upright < 0.9 else 0.0

            total_reward = (
                    r_progress +
                    r_deviation +
                    r_contact +
                    r_force_align +
                    r_correction +
                    r_stability
            )

            if self.episode_length % 100 == 0:
                print(f"Step {self.episode_length} [PUSH]: "
                      f"prog={progress:.1%}, "
                      f"dev={deviation_mag:.3f}m, "
                      f"F={force_mag:.1f}N, "
                      f"K=[{self.current_K[0]:.0f},{self.current_K[1]:.0f}]")

        total_reward -= 0.05  # Time penalty

        # Success
        if progress > 0.95 and deviation_mag < self.path_tolerance:
            total_reward += 100.0
            info["is_success"] = True
            print(f"SUCCESS at step {self.episode_length}!")

        # Failures
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        bottle_upright = bottle_mat[2, 2]

        if bottle_upright < 0.5:
            total_reward -= 50.0
            info["bottle_fallen"] = True

        if deviation_mag > 0.2:
            total_reward -= 50.0
            info["off_path"] = True

        self.prev_progress = progress

        info["progress"] = progress
        info["deviation"] = deviation_mag
        info["force_magnitude"] = force_mag
        info["is_touching"] = is_touching
        info["hand_z"] = hand_pos[2]
        info["in_approach_phase"] = self.in_approach_phase

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

        self.prev_progress = 0.0
        self.contact_made = False
        self.in_approach_phase = True
        self.episode_length = 0
        self.current_K = np.array([500.0, 500.0, 1000.0])

        self.force_profile_log = []
        self.stiffness_profile_log = []

        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_start[:2])

        print(f"\n{'=' * 50}")
        print(f"NEW EPISODE - {traj_type}")
        print(f"Bottle: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"Hand: ({hand_pos[0]:.2f}, {hand_pos[1]:.2f}, {hand_pos[2]:.3f})")
        print(f"Distance: {dist_to_bottle:.3f}m | Phase: APPROACH")
        print(f"{'=' * 50}")

        return self._get_obs(), {}

    # ==================================================================
    #                      STEP
    # ==================================================================

    def step(self, action):
        if self.in_approach_phase:
            approach_vel, dist = self._compute_approach_velocity()
            approach_vel[0] += action[0] * 0.01
            approach_vel[1] += action[1] * 0.01
            cart_vel = approach_vel

            if self._check_approach_complete() or self._is_touching():
                self.in_approach_phase = False
                print(f"Step {self.episode_length}: APPROACH COMPLETE → PUSH phase")
        else:
            cart_vel = self._compute_push_velocity(action)

        q_dot = self._cartesian_to_joint_velocity(cart_vel)

        dt = 0.02
        self.current_qpos_target = np.clip(
            self.current_qpos_target + q_dot * dt,
            self.act_low,
            self.act_high
        )
        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

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
