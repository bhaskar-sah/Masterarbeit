import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os


class PandaPushTrajectoryEnv(gym.Env):
    """
    Push environment with wrist glide feature.
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
        self.push_speed = 0.006
        self.behind_distance = 0.04
        self.reposition_speed = 0.02

        # ==================== WRIST GLIDE (NEW!) ====================
        # Joint 7 index (0-indexed)
        self.wrist_joint_idx = 6
        # Max wrist rotation for gliding (radians, ~20 degrees)
        self.max_wrist_rotation = 0.35
        # How fast wrist rotates
        self.wrist_rotation_speed = 1.5
        # Current wrist offset
        self.wrist_offset = 0.0
        # Base wrist position (saved at start of episode)
        self.base_wrist_pos = 0.0

        # ==================== FORCE TARGETS ====================
        self.target_force = 3.0
        self.high_force_threshold = 5.0

        # ==================== STIFFNESS (What RL learns!) ====================
        self.K_min = 100.0
        self.K_max = 500.0
        self.current_K = np.array([300.0, 300.0])

        # ==================== TILT THRESHOLDS ====================
        self.tilt_ok = 0.995
        self.tilt_slow = 0.99
        self.tilt_stop = 0.98
        self.tilt_settle = 0.97

        # Settle state
        self.is_settling = False
        self.settle_counter = 0
        self.settle_required = 30

        # Cartesian control
        self.target_z = 0.93
        self.z_gain = 10.0
        self.damping = 0.01

        # Trajectory
        self.trajectory_type = trajectory_type
        self.trajectory = None
        self.total_arc_length = 0.0
        self.path_tolerance = 0.05

        # Phase
        self.in_approach = True

        # Action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space
        obs_dim = 34
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State
        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0
        self.max_episode_length = 3000
        self.prev_progress = 0.0

        # Logging
        self.force_log = []
        self.stiffness_log = []
        self.deviation_log = []

        print(f"\n{'=' * 60}")
        print("PUSH WITH WRIST GLIDE")
        print(f"{'=' * 60}")
        print("Existing push logic + wrist rotation for gliding")
        print(f"Push speed: {self.push_speed * 1000:.1f} mm/s")
        print(f"Max wrist rotation: {np.degrees(self.max_wrist_rotation):.1f}°")
        print(f"{'=' * 60}")

    # ==================================================================
    #           PUSH DIRECTION (existing logic)
    # ==================================================================

    def _get_push_direction(self, bottle_xy):
        """
        Compute push direction: blend of forward + correction.
        (This is the existing logic)
        """
        tangent = self._get_path_tangent(bottle_xy)
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)

        correction_threshold = 0.01

        if deviation_mag < correction_threshold:
            return tangent

        correction_dir = deviation_vec / (deviation_mag + 1e-6)

        K_avg = np.mean(self.current_K)
        K_factor = K_avg / self.K_max

        correction_weight = min((deviation_mag - correction_threshold) * K_factor * 8.0, 0.4)
        forward_weight = 1.0 - correction_weight

        push_dir = forward_weight * tangent + correction_weight * correction_dir
        push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)

        return push_dir

    def _get_hand_target_position(self, bottle_xy, push_dir):
        """Get target position behind bottle."""
        hand_xy = bottle_xy - push_dir * self.behind_distance
        return np.array([hand_xy[0], hand_xy[1], self.target_z])

    # ==================================================================
    #           WRIST GLIDE (NEW!)
    # ==================================================================

    def _compute_wrist_glide(self, bottle_xy):
        """
        Compute wrist rotation to help hand glide on bottle surface.

        - deviation < 1cm: No rotation (wrist centered)
        - deviation > 1cm: Rotate wrist to glide around bottle

        The wrist rotates OPPOSITE to deviation direction so the hand
        can glide around the bottle to stay behind it.
        """
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)

        # Only glide when deviation is significant
        if deviation_mag < 0.01:
            return 0.0  # Center wrist

        # K factor from learned stiffness
        K_avg = np.mean(self.current_K)
        K_factor = K_avg / self.K_max

        # Rotation based on deviation in X direction (for straight trajectory)
        # deviation_vec[0] > 0: bottle RIGHT of trajectory → rotate wrist LEFT (negative)
        # deviation_vec[0] < 0: bottle LEFT of trajectory → rotate wrist RIGHT (positive)
        target_rotation = -deviation_vec[0] * K_factor * 8.0

        # Clamp to max rotation
        target_rotation = np.clip(target_rotation, -self.max_wrist_rotation, self.max_wrist_rotation)

        return target_rotation

    # ==================================================================
    #           VELOCITY COMPUTATION (existing logic)
    # ==================================================================

    def _get_bottle_tilt(self):
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        return bottle_mat[2, 2]

    def _check_stable(self):
        tilt = self._get_bottle_tilt()
        angvel = np.linalg.norm(self.data.qvel[3:6])
        return tilt > 0.995 and angvel < 0.05

    def _compute_approach_velocity(self):
        """Move to position behind bottle."""
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        push_dir = self._get_push_direction(bottle_xy)
        target_pos = self._get_hand_target_position(bottle_xy, push_dir)

        error = target_pos - hand_pos
        dist = np.linalg.norm(error[:2])

        v_desired = error * 1.5
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.03:
            v_desired[:2] = v_desired[:2] / v_mag * 0.03

        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        return v_desired, dist

    def _compute_push_velocity(self, action):
        """Compute push velocity (existing logic)."""
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        tilt = self._get_bottle_tilt()

        # Parse action
        speed_mod = action[0]
        self.current_K[0] = self.K_min + action[1] * (self.K_max - self.K_min)
        self.current_K[1] = self.K_min + action[2] * (self.K_max - self.K_min)

        push_dir = self._get_push_direction(bottle_xy)
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)

        # Settling
        if self.is_settling:
            if self._check_stable():
                self.settle_counter += 1
                if self.settle_counter >= self.settle_required:
                    self.is_settling = False
                    self.settle_counter = 0
            else:
                self.settle_counter = 0

            target_pos = self._get_hand_target_position(bottle_xy, push_dir)
            error = target_pos - hand_pos
            v_desired = error * 0.3
            v_desired[:2] = np.clip(v_desired[:2], -0.005, 0.005)
            v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])
            return v_desired

        if tilt < self.tilt_settle:
            self.is_settling = True
            self.settle_counter = 0

        # Speed control
        if tilt > self.tilt_ok:
            speed_mult = 1.0
        elif tilt > self.tilt_slow:
            speed_mult = 0.5
        elif tilt > self.tilt_stop:
            speed_mult = 0.2
        else:
            speed_mult = 0.0

        # Compute velocity
        v_desired = np.zeros(3)
        target_pos = self._get_hand_target_position(bottle_xy, push_dir)
        pos_error = target_pos[:2] - hand_pos[:2]
        pos_error_mag = np.linalg.norm(pos_error)

        # Repositioning based on deviation
        if deviation_mag > 0.03:
            reposition_gain = 4.0 + deviation_mag * 30.0
            v_reposition = pos_error * reposition_gain
            v_reposition = np.clip(v_reposition, -0.05, 0.05)
            deviation_factor = max(1.0 - deviation_mag * 8.0, 0.2)
            push_speed = self.push_speed * deviation_factor * speed_mult
            v_push = push_dir * push_speed

        elif deviation_mag > 0.01:
            reposition_gain = 2.0 + deviation_mag * 20.0
            v_reposition = pos_error * reposition_gain
            v_reposition = np.clip(v_reposition, -0.03, 0.03)
            push_speed = self.push_speed * 0.7 * speed_mult
            v_push = push_dir * push_speed

        else:
            reposition_gain = 0.5
            v_reposition = pos_error * reposition_gain
            v_reposition = np.clip(v_reposition, -0.008, 0.008)
            push_speed = self.push_speed * (1.0 + speed_mod * 0.3) * speed_mult
            v_push = push_dir * push_speed

        v_desired[0] = v_reposition[0] + v_push[0]
        v_desired[1] = v_reposition[1] + v_push[1]
        v_desired[:2] = np.clip(v_desired[:2], -0.04, 0.04)
        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        return v_desired

    def _cartesian_to_joint_velocity(self, cart_vel):
        """Convert Cartesian to joint velocity."""
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
        start = bottle_start_xy.copy()
        n_points = 50

        if traj_type == "straight":
            end = start + np.array([0.0, -0.4])
        elif traj_type == "diagonal":
            end = start + np.array([0.15, -0.3])
        elif traj_type == "curved":
            t = np.linspace(0, np.pi / 2, n_points)
            radius = 0.2
            x = start[0] + radius * np.sin(t)
            y = start[1] - radius * (1 - np.cos(t))
            return np.stack([x, y], axis=1).astype(np.float32)
        elif traj_type == "s_curve":
            t = np.linspace(0, 1, n_points)
            x = start[0] + 0.1 * np.sin(2 * np.pi * t)
            y = start[1] - 0.35 * t
            return np.stack([x, y], axis=1).astype(np.float32)
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
        arc_len = np.sum(np.linalg.norm(np.diff(self.trajectory[:idx + 1], axis=0), axis=1)) if idx > 0 else 0.0
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
        return (tangent / norm).astype(np.float32) if norm > 1e-6 else np.array([0.0, -1.0], dtype=np.float32)

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

        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)
        progress = self._get_progress(bottle_xy)
        contact_force = self._get_contact_force()
        is_touching = np.array([1.0 if self._is_touching() else 0.0], dtype=np.float32)
        tilt = np.array([self._get_bottle_tilt()], dtype=np.float32)

        K_normalized = (self.current_K - self.K_min) / (self.K_max - self.K_min)

        obs = np.concatenate([
            qpos,  # 7
            qvel,  # 7
            hand_pos,  # 3
            bottle_pos,  # 3
            dir_to_bottle,  # 2
            [dist_to_bottle],  # 1
            deviation_vec,  # 2
            [deviation_mag],  # 1
            tangent,  # 2
            [progress],  # 1
            contact_force[:2],  # 2
            is_touching,  # 1
            K_normalized,  # 2
        ])

        return obs.astype(np.float32)

    # ==================================================================
    #                      REWARD
    # ==================================================================

    def _get_reward(self):
        info = {"is_success": False}

        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        hand_pos = self.data.xpos[self.hand_body_id]

        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        progress = self._get_progress(bottle_xy)
        tilt = self._get_bottle_tilt()
        is_touching = self._is_touching()
        force = self._get_contact_force()
        force_mag = np.linalg.norm(force)
        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)

        if self.in_approach:
            total_reward = -2.0 * dist_to_bottle - 5.0 * abs(hand_pos[2] - self.target_z)
            if is_touching or dist_to_bottle < 0.06:
                total_reward += 10.0
                self.in_approach = False

            if self.episode_length % 50 == 0:
                print(f"Step {self.episode_length} [APPROACH]: dist={dist_to_bottle:.3f}m")
        else:
            # Progress
            progress_delta = progress - self.prev_progress
            r_progress = 150.0 * max(progress_delta, 0)

            # Deviation
            if deviation_mag < 0.02:
                r_deviation = 5.0
            elif deviation_mag < 0.04:
                r_deviation = 2.0
            elif deviation_mag < 0.08:
                r_deviation = -5.0 * deviation_mag
            else:
                r_deviation = -15.0 * deviation_mag

            # Stability
            if tilt > 0.995:
                r_stability = 5.0
            elif tilt > 0.99:
                r_stability = 2.0
            elif tilt > 0.98:
                r_stability = 0.0
            elif tilt > 0.97:
                r_stability = -5.0
            else:
                r_stability = -20.0

            # Contact
            r_contact = 0.5 if is_touching else -0.3

            # Force
            if force_mag > self.high_force_threshold:
                r_force = -5.0 * (force_mag - self.high_force_threshold)
            elif force_mag > self.target_force:
                r_force = -1.0 * (force_mag - self.target_force)
            elif force_mag > 1.0:
                r_force = 2.0
            else:
                r_force = 0.0

            total_reward = r_progress + r_deviation + r_stability + r_contact + r_force

            # Debug
            if self.episode_length % 50 == 0:
                mode = "SETTLE" if self.is_settling else "PUSH"
                K_avg = np.mean(self.current_K)
                side = "LEFT" if deviation_vec[0] > 0.005 else "RIGHT" if deviation_vec[0] < -0.005 else "CENTER"
                wrist_deg = np.degrees(self.wrist_offset)
                print(f"Step {self.episode_length} [{mode}]: "
                      f"prog={progress:.1%}, dev={deviation_mag:.3f}m ({side}), "
                      f"K={K_avg:.0f}, F={force_mag:.1f}N, tilt={tilt:.4f}, "
                      f"wrist={wrist_deg:.1f}°")

        total_reward -= 0.005

        # Success
        if progress > 0.95 and deviation_mag < self.path_tolerance:
            total_reward += 150.0
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
        info["force_magnitude"] = force_mag
        info["bottle_tilt"] = tilt

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

        # Save base wrist position
        self.base_wrist_pos = self.current_qpos_target[6]
        self.wrist_offset = 0.0

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
        self.current_K = np.array([300.0, 300.0])
        self.is_settling = False
        self.settle_counter = 0

        self.force_log = []
        self.stiffness_log = []
        self.deviation_log = []

        print(f"\n{'=' * 50}")
        print(f"EPISODE: {traj_type}")
        print(f"Bottle: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"{'=' * 50}")

        return self._get_obs(), {}

    def step(self, action):
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        if self.in_approach:
            cart_vel, dist = self._compute_approach_velocity()
            target_wrist = 0.0  # No wrist rotation during approach
            if dist < 0.05 or self._is_touching():
                self.in_approach = False
                print(f"Step {self.episode_length}: → PUSH phase")
        else:
            cart_vel = self._compute_push_velocity(action)
            # Compute wrist glide rotation based on deviation
            target_wrist = self._compute_wrist_glide(bottle_xy)

        # EXISTING: Cartesian to joint velocity for joints 1-6
        q_dot = self._cartesian_to_joint_velocity(cart_vel)

        dt = 0.02

        # Update joints 1-6 from Jacobian (existing logic)
        self.current_qpos_target[:6] = np.clip(
            self.current_qpos_target[:6] + q_dot[:6] * dt,
            self.act_low[:6], self.act_high[:6]
        )

        # NEW: Update joint 7 (wrist) separately for gliding
        # Smoothly move wrist towards target
        wrist_error = target_wrist - self.wrist_offset
        wrist_speed = self.wrist_rotation_speed * wrist_error
        wrist_speed = np.clip(wrist_speed, -0.3, 0.3)  # Limit rotation speed

        self.wrist_offset += wrist_speed * dt
        self.wrist_offset = np.clip(self.wrist_offset, -self.max_wrist_rotation, self.max_wrist_rotation)

        # Set wrist joint target
        self.current_qpos_target[6] = np.clip(
            self.base_wrist_pos + self.wrist_offset,
            self.act_low[6], self.act_high[6]
        )

        # Send to actuators
        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Log
        force = self._get_contact_force()
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        _, dev = self._get_path_deviation(bottle_xy)

        self.force_log.append(np.linalg.norm(force))
        self.stiffness_log.append(np.mean(self.current_K))
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
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None