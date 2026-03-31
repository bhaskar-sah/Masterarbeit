"""
Panda Push Environment - Matches User's Diagram

BEHAVIOR (from user's drawing):
1. Hand approaches bottle from behind
2. Hand pushes bottle along trajectory towards goal
3. When bottle deviates from trajectory:
   - Hand goes BEHIND the bottle (on the side away from trajectory)
   - Hand pushes bottle BACK towards the trajectory
4. Force profile (stiffness K) determines how strong the correction is

    ┌───┐ Hand
    │   │
    └───┘
        ↘
         ↘ Approach from behind
          ↘
           🍾──────────────────────► Goal
            ↘    ↗
             ↘  ↗ Deviation → Correction
              🍾
              ↑
           ┌───┐ Hand repositions behind deviated bottle
           │   │──► Pushes back to trajectory
           └───┘
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os


class PandaPushTrajectoryEnv(gym.Env):
    """
    Hand goes BEHIND bottle to push it towards trajectory.
    Learns force profile (stiffness) for correction strength.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_type="straight"):
        super().__init__()

        # ==================== LOAD MODEL ====================
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "../robot_panda_push_force.xml")
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

        # ==================== CONTROL ====================
        self.actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = self.actuator_ranges[:, 0]
        self.act_high = self.actuator_ranges[:, 1]
        self.current_qpos_target = np.zeros(7)

        # ==================== PUSH PARAMETERS ====================
        self.push_speed = 0.015  # Base push speed (m/s)
        self.correction_speed = 0.02  # Speed when correcting
        self.behind_distance = 0.04  # How far behind bottle (4cm)

        # ==================== STIFFNESS (FORCE PROFILE) ====================
        # This is what the RL agent learns!
        self.K_min = 100.0  # Minimum stiffness → gentle correction
        self.K_max = 1000.0  # Maximum stiffness → strong correction
        self.current_K = np.array([300.0, 300.0])

        # ==================== SAFETY ====================
        self.max_force = 8.0
        self.safe_tilt = 0.95

        # ==================== CARTESIAN ====================
        self.target_z = 0.93
        self.z_gain = 10.0
        self.damping = 0.01

        # ==================== TRAJECTORY ====================
        self.trajectory_type = trajectory_type
        self.trajectory = None
        self.total_arc_length = 0.0
        self.path_tolerance = 0.05

        # ==================== ACTION SPACE ====================
        # [push_speed_mod, correction_mod, Kx, Ky]
        # Kx, Ky: Stiffness for X and Y correction (FORCE PROFILE!)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ==================== OBSERVATION ====================
        obs_dim = 34
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ==================== STATE ====================
        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0
        self.max_episode_length = 2000
        self.prev_progress = 0.0
        self.in_approach_phase = True

        # Logging for analysis
        self.force_profile_log = []
        self.stiffness_profile_log = []
        self.deviation_log = []

        print(f"\n{'=' * 60}")
        print("PUSH WITH CORRECTION (matching user's diagram)")
        print(f"{'=' * 60}")
        print("- Hand goes behind bottle")
        print("- When bottle deviates → Hand repositions behind it")
        print("- Pushes bottle back to trajectory")
        print("- Stiffness K = learned force profile")
        print(f"{'=' * 60}")

    # ==================================================================
    #           COMPUTE WHERE HAND SHOULD BE
    # ==================================================================

    def _get_push_direction(self, bottle_xy):
        """
        Get the direction the bottle should be pushed.

        If bottle is ON trajectory → push along tangent (towards goal)
        If bottle is OFF trajectory → push towards trajectory + along tangent
        """
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)

        if deviation_mag < 0.01:
            # On trajectory - push along tangent
            push_dir = tangent
        else:
            # Off trajectory - push towards trajectory AND along tangent
            # Blend: more deviation = more correction, less forward
            correction_weight = min(deviation_mag * 10, 0.8)  # Max 80% correction
            forward_weight = 1.0 - correction_weight

            # Correction direction (towards trajectory)
            correction_dir = deviation_vec / (deviation_mag + 1e-6)

            # Combined push direction
            push_dir = forward_weight * tangent + correction_weight * correction_dir
            push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)

        return push_dir

    def _get_hand_target_position(self, bottle_xy):
        """
        Compute where the hand should be to push the bottle correctly.

        KEY: Hand should be BEHIND the bottle, opposite to push direction!

        If bottle needs to go RIGHT → Hand should be on LEFT of bottle
        If bottle needs to go FORWARD → Hand should be BEHIND bottle
        """
        push_dir = self._get_push_direction(bottle_xy)

        # Hand position: behind bottle (opposite to push direction)
        hand_target_xy = bottle_xy - push_dir * self.behind_distance

        return np.array([hand_target_xy[0], hand_target_xy[1], self.target_z])

    # ==================================================================
    #           APPROACH PHASE
    # ==================================================================

    def _compute_approach_velocity(self):
        """Move to position behind bottle."""
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]

        target_pos = self._get_hand_target_position(bottle_xy)

        error = target_pos - hand_pos
        dist = np.linalg.norm(error[:2])

        v_desired = error * 2.0
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.05:
            v_desired[:2] = v_desired[:2] / v_mag * 0.05

        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        return v_desired, dist

    def _check_approach_complete(self):
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]

        dist_xy = np.linalg.norm(hand_pos[:2] - bottle_pos[:2])
        height_ok = abs(hand_pos[2] - self.target_z) < 0.03

        return dist_xy < 0.08 and height_ok

    # ==================================================================
    #           PUSH PHASE - THE MAIN LOGIC
    # ==================================================================

    def _get_bottle_tilt(self):
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        return bottle_mat[2, 2]

    def _compute_push_velocity(self, action):
        """
        Compute velocity to:
        1. Stay behind bottle
        2. Push bottle in the right direction (towards trajectory + goal)

        The STIFFNESS (K) from action determines correction strength!
        """
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]

        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)
        push_dir = self._get_push_direction(bottle_xy)

        tilt = self._get_bottle_tilt()
        force = self._get_contact_force()
        force_mag = np.linalg.norm(force)

        # Parse action
        speed_mod = action[0]  # Modifies push speed

        # STIFFNESS from action - THIS IS THE FORCE PROFILE!
        self.current_K[0] = self.K_min + action[2] * (self.K_max - self.K_min)
        self.current_K[1] = self.K_min + action[3] * (self.K_max - self.K_min)
        K_avg = np.mean(self.current_K)
        K_normalized = (K_avg - self.K_min) / (self.K_max - self.K_min)

        # === SAFETY MULTIPLIER ===
        safety_mult = 1.0
        if tilt < self.safe_tilt:
            safety_mult = 0.3
        if force_mag > self.max_force:
            safety_mult *= 0.5

        # === BUILD VELOCITY ===
        v_desired = np.zeros(3)

        # 1. REPOSITION: Stay behind bottle (this is continuous!)
        target_hand_pos = self._get_hand_target_position(bottle_xy)
        reposition_error = target_hand_pos[:2] - hand_pos[:2]
        reposition_dist = np.linalg.norm(reposition_error)

        # Reposition velocity - always try to stay behind bottle
        if reposition_dist > 0.005:  # More than 5mm off
            v_reposition = reposition_error * 2.0  # P-control
            v_reposition = np.clip(v_reposition, -0.03, 0.03)
            v_desired[0] += v_reposition[0]
            v_desired[1] += v_reposition[1]

        # 2. PUSH: Push in the push direction
        # Speed depends on: base speed + action modifier + stiffness
        # Higher stiffness → can push faster during correction
        if deviation_mag > 0.02:
            # Correcting - use correction speed, scaled by stiffness
            speed = self.correction_speed * (0.5 + 0.5 * K_normalized)
        else:
            # Normal pushing along trajectory
            speed = self.push_speed * (1.0 + speed_mod * 0.3)

        speed *= safety_mult

        v_push = push_dir * speed
        v_desired[0] += v_push[0]
        v_desired[1] += v_push[1]

        # 3. HEIGHT
        z_error = self.target_z - hand_pos[2]
        v_desired[2] = self.z_gain * z_error

        # LIMITS
        v_desired[:2] = np.clip(v_desired[:2], -0.05, 0.05)
        v_desired[2] = np.clip(v_desired[2], -0.1, 0.1)

        # Debug
        if self.episode_length % 100 == 0:
            mode = "CORRECTING" if deviation_mag > 0.02 else "PUSHING"
            print(f"Step {self.episode_length} [{mode}]: "
                  f"prog={self._get_progress(bottle_xy):.1%}, "
                  f"dev={deviation_mag:.3f}m, "
                  f"K={K_avg:.0f}, "
                  f"tilt={tilt:.3f}")

        return v_desired

    def _cartesian_to_joint_velocity(self, cart_vel):
        """Convert Cartesian velocity to joint velocity (position only)."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)

        J = jacp[:, 6:13]  # Position Jacobian only

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
        Returns:
        - deviation_vec: Vector FROM bottle TO trajectory (correction direction)
        - deviation_mag: Distance from trajectory
        """
        _, closest_pt, _ = self._get_closest_point_on_trajectory(bottle_xy)
        deviation_vec = closest_pt - bottle_xy
        deviation_mag = np.linalg.norm(deviation_vec)
        return deviation_vec.astype(np.float32), float(deviation_mag)

    def _get_path_tangent(self, bottle_xy):
        """Get direction along trajectory (towards goal)."""
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

        hand_to_bottle = bottle_pos[:2] - hand_pos[:2]
        dist_to_bottle = np.linalg.norm(hand_to_bottle)
        dir_to_bottle = hand_to_bottle / (dist_to_bottle + 1e-6)

        deviation_vec, _ = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)
        progress = self._get_progress(bottle_xy)
        contact_force = self._get_contact_force()
        is_touching = np.array([1.0 if self._is_touching() else 0.0], dtype=np.float32)

        K_normalized = (self.current_K - self.K_min) / (self.K_max - self.K_min)

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
        ])

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
        progress = self._get_progress(bottle_xy)
        tilt = self._get_bottle_tilt()
        is_touching = self._is_touching()
        force = self._get_contact_force()
        force_mag = np.linalg.norm(force)
        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)

        if is_touching:
            self.in_approach_phase = False

        if self.in_approach_phase:
            # Approach reward
            total_reward = -2.0 * dist_to_bottle
            total_reward += -5.0 * abs(hand_pos[2] - self.target_z)
            if is_touching:
                total_reward += 10.0

            if self.episode_length % 50 == 0:
                print(f"Step {self.episode_length} [APPROACH]: dist={dist_to_bottle:.3f}m")
        else:
            # === PUSH REWARDS ===

            # Progress (most important!)
            progress_delta = progress - self.prev_progress
            r_progress = 80.0 * max(progress_delta, 0)

            # Stay on path
            if deviation_mag < 0.02:
                r_deviation = 3.0
            elif deviation_mag < self.path_tolerance:
                r_deviation = 1.0
            else:
                r_deviation = -10.0 * deviation_mag

            # Contact
            r_contact = 1.0 if is_touching else -2.0 * dist_to_bottle

            # Force (not too strong)
            if 2.0 < force_mag < 6.0:
                r_force = 1.0
            elif force_mag > 10.0:
                r_force = -3.0
            else:
                r_force = 0.0

            # Stability (VERY important!)
            if tilt > 0.98:
                r_stability = 3.0
            elif tilt > 0.95:
                r_stability = 1.0
            elif tilt > 0.90:
                r_stability = -2.0
            else:
                r_stability = -20.0

            # Hand positioning (behind bottle)
            target_hand = self._get_hand_target_position(bottle_xy)
            hand_error = np.linalg.norm(target_hand[:2] - hand_pos[:2])
            if hand_error < 0.02:
                r_position = 1.0
            else:
                r_position = -0.5 * hand_error

            total_reward = (
                    r_progress +
                    r_deviation +
                    r_contact +
                    r_force +
                    r_stability +
                    r_position
            )

        total_reward -= 0.01  # Time penalty

        # Success
        if progress > 0.95 and deviation_mag < self.path_tolerance:
            total_reward += 100.0
            info["is_success"] = True
            print(f"SUCCESS at step {self.episode_length}!")

        # Failures
        if tilt < 0.5:
            total_reward -= 100.0
            info["bottle_fallen"] = True

        if deviation_mag > 0.25:
            total_reward -= 30.0
            info["off_path"] = True

        self.prev_progress = progress

        # Log for analysis
        self.deviation_log.append(deviation_mag)

        info["progress"] = progress
        info["deviation"] = deviation_mag
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

        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_start = self.data.xpos[self.bottle_body_id].copy()

        traj_type = self.trajectory_type
        if options and "trajectory_type" in options:
            traj_type = options["trajectory_type"]

        self.trajectory = self._generate_trajectory(bottle_start[:2], traj_type)
        self._compute_arc_length()

        self.prev_progress = 0.0
        self.in_approach_phase = True
        self.episode_length = 0
        self.current_K = np.array([300.0, 300.0])

        self.force_profile_log = []
        self.stiffness_profile_log = []
        self.deviation_log = []

        print(f"\n{'=' * 50}")
        print(f"EPISODE: {traj_type}")
        print(f"Bottle: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"Goal: ({self.trajectory[-1, 0]:.2f}, {self.trajectory[-1, 1]:.2f})")
        print(f"{'=' * 50}")

        return self._get_obs(), {}

    def step(self, action):
        if self.in_approach_phase:
            cart_vel, _ = self._compute_approach_velocity()
            if self._check_approach_complete() or self._is_touching():
                self.in_approach_phase = False
                print(f"Step {self.episode_length}: → PUSH phase")
        else:
            cart_vel = self._compute_push_velocity(action)

        q_dot = self._cartesian_to_joint_velocity(cart_vel)

        dt = 0.02
        self.current_qpos_target = np.clip(
            self.current_qpos_target + q_dot * dt,
            self.act_low, self.act_high
        )
        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Log
        force = self._get_contact_force()
        self.force_profile_log.append(force.copy())
        self.stiffness_profile_log.append(self.current_K.copy())

        obs = self._get_obs()
        reward, info = self._get_reward()
        self.episode_length += 1

        terminated = info.get("is_success", False) or info.get("bottle_fallen", False) or info.get("off_path", False)
        truncated = self.episode_length >= self.max_episode_length

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
