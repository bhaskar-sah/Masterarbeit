"""
Panda Push Environment with Cartesian Space + Impedance Control
FIXED VERSION: Includes approach phase to reach the bottle first!

The robot must:
1. APPROACH: Move towards the bottle until contact
2. PUSH: Push the bottle along trajectory with learned force profile
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os


class PandaPushTrajectoryEnv(gym.Env):
    """
    Environment with:
    - Approach phase (move to bottle)
    - Push phase (push along trajectory with impedance control)
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

        # ==================== CONTROL MODE ====================
        self.control_mode = control_mode

        # ==================== IMPEDANCE PARAMETERS ====================
        self.K_min = 50.0
        self.K_max = 2000.0
        self.damping_ratio = 0.7
        self.current_K = np.array([500.0, 500.0, 1000.0])
        self.F_max = 20.0

        # ==================== CARTESIAN CONTROL ====================
        self.target_z = 0.93
        self.max_cart_vel = 0.05  # Increased for approach phase
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
        # Added: distance_to_bottle, direction_to_bottle
        # qpos(7) + qvel(7) + hand(3) + bottle(3) + dir(2) + dist(1) + dev(2) + tan(2) + prog(1) + force(3) + touch(1) + K(2)
        # 7 + 7 + 3 + 3 + 2 + 1 + 2 + 2 + 1 + 3 + 1 + 2 = 34
        obs_dim = 34
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
        self.in_approach_phase = True  # Start in approach phase

        self.force_profile_log = []
        self.stiffness_profile_log = []

        print(f"\n{'=' * 60}")
        print("Environment: Cartesian + Impedance (WITH APPROACH PHASE)")
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
        distance = np.linalg.norm(error[:2])  # XY distance

        # Proportional control with velocity limit
        approach_gain = 2.0
        v_desired = approach_gain * error

        # Limit velocity
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > self.max_cart_vel * 2:  # Allow faster approach
            v_desired[:2] = v_desired[:2] / v_mag * self.max_cart_vel * 2

        # Strong Z control
        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        return v_desired, distance

    def _check_approach_complete(self):
        """Check if we're close enough to start pushing."""
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]

        # XY distance to bottle
        dist_xy = np.linalg.norm(hand_pos[:2] - bottle_pos[:2])

        # Height check
        height_ok = abs(hand_pos[2] - self.target_z) < 0.03

        # Close enough and at right height
        return dist_xy < self.contact_threshold and height_ok

    # ==================================================================
    #           PUSH PHASE - Impedance control
    # ==================================================================

    def _cartesian_to_joint_velocity(self, cart_vel):
        """Convert 3D Cartesian velocity to joint velocity."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)
        J = jacp[:, 6:13]

        # Damped pseudo-inverse
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.damping ** 2 * np.eye(3))

        q_dot = J_pinv @ cart_vel
        return q_dot

    def _compute_push_velocity(self, action):
        """
        Compute velocity for pushing phase.
        FIXED: Inverted correction direction for proper steering.
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

        # === Build push velocity ===
        v_desired = np.zeros(3)

        # 1. Base push direction from RL
        v_desired[0] = desired_vel_xy[0]
        v_desired[1] = desired_vel_xy[1]

        # 2. Add correction based on deviation and stiffness
        # [CRITICAL FIX]
        # To push bottle towards path (along deviation_vec), hand must move
        # OPPOSITE to deviation_vec (to hit the other side of the bottle).
        K_normalized = (self.current_K[:2] - self.K_min) / (self.K_max - self.K_min)

        if deviation_mag > 0.01:
            # Scale correction by stiffness (learnable gain)
            # We multiply by -1.0 to move hand to the "pushing side"
            correction_scale = K_normalized * 2.0  # Increased gain for snappier response
            v_correction = correction_scale * deviation_vec * 5.0

            # SUBTRACT the deviation vector to move hand to the opposite side
            v_desired[0] -= v_correction[0]
            v_desired[1] -= v_correction[1]

        # 3. Stay close to bottle (track bottle position)
        hand_to_bottle = bottle_xy - hand_pos[:2]
        dist_to_bottle = np.linalg.norm(hand_to_bottle)

        if dist_to_bottle > 0.05:  # Relaxed threshold slightly
            # Add velocity towards bottle to maintain contact
            track_vel = hand_to_bottle * 2.0
            v_desired[0] += track_vel[0]
            v_desired[1] += track_vel[1]

        # 4. Height maintenance
        z_error = self.target_z - hand_pos[2]
        v_desired[2] = self.z_gain * z_error

        # Clip velocities
        v_desired = np.clip(v_desired, -0.2, 0.2)  # Increased limit slightly

        return v_desired

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
        """Observation includes approach info."""
        qpos = self.data.qpos[7:14].astype(np.float32)
        qvel = self.data.qvel[6:13].astype(np.float32)
        hand_pos = self.data.xpos[self.hand_body_id].astype(np.float32)
        bottle_pos = self.data.xpos[self.bottle_body_id].astype(np.float32)
        bottle_xy = bottle_pos[:2]

        # Direction and distance to bottle (for approach)
        hand_to_bottle = bottle_pos[:2] - hand_pos[:2]
        dist_to_bottle = np.linalg.norm(hand_to_bottle)
        if dist_to_bottle > 0.001:
            dir_to_bottle = hand_to_bottle / dist_to_bottle
        else:
            dir_to_bottle = np.zeros(2)

        deviation_vec, _ = self._get_path_deviation(bottle_xy)
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
            dir_to_bottle,  # 2 (direction to bottle)
            [dist_to_bottle],  # 1 (distance to bottle)
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

        # Distance to bottle
        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)

        if is_touching:
            self.contact_made = True
            self.in_approach_phase = False  # Switch to push phase

        # ===== REWARD COMPONENTS =====

        if self.in_approach_phase:
            # APPROACH PHASE: Reward getting close to bottle
            r_approach = -2.0 * dist_to_bottle  # Closer = better

            # Height reward
            # keep height reward and remove reward for joint controls. Let the robot control the joints by itself.
            height_error = abs(hand_pos[2] - self.target_z)
            r_height = -5.0 * height_error

            # Bonus for making contact
            if is_touching:
                r_contact_bonus = 10.0
            else:
                r_contact_bonus = 0.0

            total_reward = r_approach + r_height + r_contact_bonus

            # Debug
            if self.episode_length % 50 == 0:
                print(f"Step {self.episode_length} [APPROACH]: "
                      f"dist={dist_to_bottle:.3f}m, "
                      f"Z={hand_pos[2]:.3f}, "
                      f"touch={is_touching}")
        else:
            # PUSH PHASE: Original reward
            progress_delta = progress - self.prev_progress
            r_progress = 100.0 * max(progress_delta, 0)

            if deviation_mag < self.path_tolerance:
                r_deviation = 1.0
            else:
                r_deviation = -15.0 * deviation_mag

            if is_touching:
                r_contact = 2.0
                force_xy = contact_force[:2]
                force_norm = np.linalg.norm(force_xy)
                if force_norm > 0.5:
                    force_dir = force_xy / force_norm
                    alignment = np.dot(force_dir, tangent)
                    r_force_align = 3.0 * max(0, alignment)
                else:
                    r_force_align = 0.0
            else:
                r_contact = -1.0 * dist_to_bottle
                r_force_align = 0.0

            # Stability
            bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
            bottle_upright = bottle_mat[2, 2]
            if bottle_upright < 0.9:
                r_stability = -10.0 * (1.0 - bottle_upright)
            else:
                r_stability = 0.0

            total_reward = (
                    r_progress + # reward for moving
                    r_deviation + # reward for not maintaining the path
                    r_contact + # reward for being in contact with the bottle
                    r_force_align + # reward for aligining along the trajectory
                    r_stability # bottle being upright (basically checking for fallen condition)
                # removed joint control reward
            )

            # Debug
            if self.episode_length % 100 == 0:
                print(f"Step {self.episode_length} [PUSH]: "
                      f"prog={progress:.1%}, "
                      f"dev={deviation_mag:.3f}m, "
                      f"F={force_mag:.1f}N, "
                      f"K=[{self.current_K[0]:.0f},{self.current_K[1]:.0f}]")

        # Time penalty
        total_reward -= 0.05

        # Success
        if progress > 0.95 and deviation_mag < self.path_tolerance:
            total_reward += 50.0
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

        # Info
        info["progress"] = progress
        info["deviation"] = deviation_mag
        info["force_magnitude"] = force_mag
        info["is_touching"] = is_touching
        info["hand_z"] = hand_pos[2]
        info["in_approach_phase"] = self.in_approach_phase
        info["dist_to_bottle"] = dist_to_bottle

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
        self.in_approach_phase = True  # Start in approach phase!
        self.episode_length = 0
        self.current_K = np.array([500.0, 500.0, 1000.0])

        self.force_profile_log = []
        self.stiffness_profile_log = []

        # Calculate initial distance
        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_start[:2])

        print(f"\n{'=' * 50}")
        print(f"NEW EPISODE - {traj_type}")
        print(f"Bottle: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"Hand: ({hand_pos[0]:.2f}, {hand_pos[1]:.2f}, {hand_pos[2]:.3f})")
        print(f"Distance to bottle: {dist_to_bottle:.3f}m")
        print(f"Phase: APPROACH")
        print(f"{'=' * 50}")

        return self._get_obs(), {}

    # ==================================================================
    #                      STEP
    # ==================================================================

    def step(self, action):
        """
        Execute one step.

        In APPROACH phase: Move towards bottle (action influences direction)
        In PUSH phase: Push with impedance control (action = velocity + stiffness)
        """

        if self.in_approach_phase:
            # === APPROACH PHASE ===
            # Compute velocity towards bottle
            approach_vel, dist = self._compute_approach_velocity()

            # RL action can slightly modify approach direction
            approach_vel[0] += action[0] * 0.01
            approach_vel[1] += action[1] * 0.01

            cart_vel = approach_vel

            # Check if approach is complete
            if self._check_approach_complete() or self._is_touching():
                self.in_approach_phase = False
                print(f"Step {self.episode_length}: APPROACH COMPLETE! Switching to PUSH phase.")
        else:
            # === PUSH PHASE ===
            cart_vel = self._compute_push_velocity(action)

        # Convert to joint velocities
        q_dot = self._cartesian_to_joint_velocity(cart_vel)

        # Integrate
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

        # Log
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
