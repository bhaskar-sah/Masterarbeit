import gymnasium as gym
from PIL.ImageOps import contain
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os

class PandaPushTrajectoryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_type="straight"):
        super().__init__()

        # ==================== LOAD MUJOCO MODEL ====================
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "robot_panda_push_force.xml")

        # Optional: Convert to an absolute path to avoid potential issues with relative paths later
        xml_path = os.path.abspath(xml_path)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find XML file at: {xml_path}")

        print(f"Loading XML from: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ==================== BODY IDs ====================
        self.home_key_id = self.model.key("home").id
        self.bottle_body_id = self.model.body("bottle").id
        self.hand_body_id = self.model.body("hand").id
        self.goal_site_id = self.model.site("goal").id
        self.left_finger_body_id = self.model.body("left_finger").id
        self.right_finger_body_id = self.model.body("right_finger").id

        # All bodies that count as "robot touching"
        self.robot_contact_bodies = {
            self.hand_body_id,
            self.left_finger_body_id,
            self.right_finger_body_id
        }

        # ==================== ROBOT CONTROL ====================
        self.actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = self.actuator_ranges[:, 0]
        self.act_high = self.actuator_ranges[:, 1]
        self.current_ctrl = np.zeros(7)

        # ==================== TRAJECTORY ====================
        self.trajectory_type = trajectory_type
        self.trajectory = None # will be set in reset
        self.total_arc_length = 0.0
        self.path_tolerance = 0.05 # 5cm - acceptable deviation from path
        self.goal_pos = None  # Will be set in reset()

        # ==================== ACTION SPACE ====================
        # Agent outputs: 7 joint velocity commands
        # The agent learns how to push (direction + force through motion)
        # Clear actions for clear defined goal (the goal is to push the bottle following the trajectory and apply forces to correct the path)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # ==================== OBSERVATION SPACE ====================
        # What the agent observs:
        # Robot joints positions (7)
        # Robot joints velocities (7)
        # Hand position (3)
        # Bottle position (3)
        # Path deviation vector (2) - How far and which direction from trajectory
        # Path tangent (2) - Which direction to push along trajectory
        # Progress (1) - How far along trajectory (0 to 1)
        # Contact Force (3) - Force from MuJoCo
        # Is touching (1) - Bindary contact flag
        obs_dim = 7 + 7 + 3 + 3 + 2 + 2 + 1 + 3 + 1  # = 29
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),  # 7 qpos + 7 qvel + 3 target + 4 hand Quat (w, x, y, z)
            dtype=np.float32
        )

        # ==================== STATE ====================
        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0
        self.max_episode_length = 500
        self.prev_progress = 0.0
        # self.goal_pos = None
        self.contact_made = False

        print(f"Environment created:")
        print(f"  Trajectory type: {trajectory_type}")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Action dim: 7 (joint velocities)")

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
            traj = np.outer(1-t, start) + np.outer(t, end)

        elif traj_type == "straight_short":
            end = start + np.array([0.0, -0.25])
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1-t, start) + np.outer(t, end)

        elif traj_type == "diagonal":
            end = start + np.array([0.15, -0.3])
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1-t, start) + np.outer(t, end)

        elif traj_type == "curved":
            t = np.linspace(0, np.pi/2, n_points)
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
            # Default: straight
            end = start + np.array([0.0, -0.3])
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1-t, start) + np.outer(t, end)

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

        # Arc length to this point
        if idx == 0:
            arc_len = 0.0
        else:
            arc_len = np.sum(np.linalg.norm(np.diff(self.trajectory[:idx+1], axis=0), axis=1))

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
            tangent = tangent/norm
        else:
            tangent = np.array([0.0, -1.0])

        return tangent.astype(np.float32)

    def _get_progress(self, bottle_xy):
        if self.total_arc_length < 1e-6:
            return 0.0
        _, _, arc_len = self._get_closest_point_on_trajectory(bottle_xy)
        return float(np.clip(arc_len / self.total_arc_length, 0.0, 1.0))

    # ==================================================================
    #                      CONTACT DETECTION (from MuJoCo)
    # ==================================================================

    def _get_contact_force(self):
        """Get contact force between robot and bottle from MuJoCo."""
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
                # Transform to world frame
                frame = contact.frame.reshape(3,3)
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

        # Backup: distance check
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        dist_xy = np.linalg.norm(hand_pos[:2] - bottle_pos[:2])
        height_diff = hand_pos[2] - bottle_pos[2]

        if dist_xy < 0.08 and 0.08 < height_diff < 0.18:
            return True

        return False

    # ==================================================================
    #                      OBSERVATION
    # ==================================================================
    def _get_obs(self):
        """Simple observation: robot state + target position"""
        # robot state
        qpos = self.data.qpos[7:14].astype(np.float32)
        qvel = self.data.qvel[6:13].astype(np.float32)

        # object positions
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]

        # Trajectory information
        deviation_vec, _ = self._get_path_deviation(bottle_xy)  # (2)
        tangent = self._get_path_tangent(bottle_xy)  # (2)
        progress = self._get_progress(bottle_xy)  # (1)

        # Contact information (from MuJoCo)
        contact_force = self._get_contact_force()  # (3)
        is_touching = np.array([1.0 if self._is_touching() else 0.0], dtype=np.float32)  # (1)

        # Total: 7 + 7 + 3 + 3 + 2 + 2 + 1 + 3 + 1 = 29
        obs = np.concatenate([
            qpos,  # 7: joint positions
            qvel,  # 7: joint velocities
            hand_pos,  # 3: hand position
            bottle_pos,  # 3: bottle position
            deviation_vec,  # 2: deviation from trajectory
            tangent,  # 2: trajectory direction (push direction)
            [progress],  # 1: progress along trajectory
            contact_force,  # 3: contact force from MuJoCo
            is_touching,  # 1: contact flag
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

        # ============ Trajectory metrics ============
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)
        progress = self._get_progress(bottle_xy)

        # ============ CONTACT CHECK ============
        is_touching = self._is_touching()
        contact_force = self._get_contact_force()
        force_mag = np.linalg.norm(contact_force)

        if is_touching:
            self.contact_made = True

        # ===== REWARD COMPONENTS =====

        # 1. PROGRESS REWARD - Move along trajectory (the main driver)
        progress_delta = progress - self.prev_progress
        # r_progress = 50.0 * max(progress_delta, 0)  # Big reward for progress
        r_progress = 100.0 * progress_delta

        # 2. DEVIATION PENALTY - Stay on trajectory
        # r_deviation = -20.0 * deviation_mag  # Penalty for being off-track
        if deviation_mag < self.path_tolerance:
            r_deviation = 0.0
        else:
            # Quadratic penalty is often smoother than linear
            r_deviation = -10.0 * deviation_mag

        # # 3. ON-PATH BONUS - Extra reward for staying on path
        # if deviation_mag < self.path_tolerance:
        #     r_on_path = 2.0
        # else:
        #     r_on_path = -2.0

        # 3. REACHING & HEIGHT REWARD (Replaces simple contact reward)
        dist_xy = np.linalg.norm(hand_pos[:2] - bottle_xy)
        dist_z = abs(hand_pos[2] - bottle_pos[2])  # Vertical distance

        # Penalize being too high up (Important fix for "moving up")
        r_height = -5.0 * dist_z

        if is_touching:
            # Small bonus for contact to encourage staying close
            r_approach = 1.0
        else:
            # Reward for getting close (XY) + Penalty for being high (Z)
            # Shaping: Max value is 0 (at contact), negative otherwise
            r_approach = -1.0 * dist_xy + r_height

        # # 4. CONTACT REWARD - Encourage maintaining contact while pushing
        # if is_touching:
        #     r_contact = 1.0
        #
        #     # Bonus for pushing in the right direction
        #     force_xy = contact_force[:2]
        #     if np.linalg.norm(force_xy) > 0.5:
        #         force_dir = force_xy / np.linalg.norm(force_xy)
        #         alignment = np.dot(force_dir, tangent)
        #         r_force_direction = 3.0 * alignment  # Reward pushing along trajectory
        #     else:
        #         r_force_direction = 0.0
        # else:
        #     r_contact = 0.0
        #     r_force_direction = 0.0
        #
        #     # Encourage approaching bottle when not in contact
        #     dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)
        #     r_contact = -2.0 + 3.0 * np.exp(-5.0 * dist_to_bottle)

        # 4. FORCE ALIGNMENT
        if is_touching and force_mag > 0:
            force_xy = contact_force[:2]
            force_dir = force_xy / (np.linalg.norm(force_xy) + 1e-6)
            alignment = np.dot(force_dir, tangent)
            # Only reward pushing in the CORRECT direction
            r_force_direction = 2.0 * max(0, alignment)
        else:
            r_force_direction = 0.0

        # # 5. STABILITY - Keep bottle upright
        # bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        # bottle_upright = bottle_mat[2, 2]  # Z-component of up vector
        # r_stability = -10.0 * (1.0 - bottle_upright)

        # 5. STABILITY
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        bottle_upright = bottle_mat[2, 2]
        # Only penalize if it actually starts tipping
        if bottle_upright < 0.9:
            r_stability = -10.0 * (1.0 - bottle_upright)
        else:
            r_stability = 0.0

        # ===== TOTAL REWARD =====
        total_reward = (
                r_progress +
                r_deviation +
                # r_on_path +
                # r_contact +
                r_approach +
                r_force_direction +
                r_stability
        )

        # Small living penalty to encourage speed (optional)
        total_reward -= 0.05

        # ===== SUCCESS CONDITION =====
        if progress > 0.95 and deviation_mag < self.path_tolerance:
            # total_reward += 200.0  # Big bonus for completing trajectory
            total_reward += 50.0 # One-time large bonus
            info["is_success"] = True
            print(f"SUCCESS! Trajectory completed at step {self.episode_length}")

        # ===== FAILURE CONDITIONS =====
        # Bottle fell over
        if bottle_upright < 0.5:
            total_reward -= 50.0
            info["bottle_fallen"] = True

        # Bottle too far from trajectory
        # if deviation_mag > 3 * self.path_tolerance:
        if deviation_mag > 0.2: # 20 cm off path
            total_reward -= 50.0
            info["off_path"] = True

        # ===== UPDATE STATE =====
        self.prev_progress = progress

        # ===== INFO FOR LOGGING =====
        info["progress"] = progress
        info["deviation"] = deviation_mag
        info["force_magnitude"] = force_mag
        info["is_touching"] = is_touching

        # ===== DEBUG PRINT =====
        if self.episode_length % 100 == 0:
            print(f"Step {self.episode_length}: "
                  f"progress={progress:.1%}, "
                  f"deviation={deviation_mag:.3f}m, "
                  f"force={force_mag:.1f}N, "
                  f"contact={is_touching}")

        return total_reward, info

    # ==================================================================
    #                      RESET
    # ==================================================================
    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset MuJoCo to keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

        self.current_ctrl = self.data.qpos[7:14].copy()

        # Settle physics
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # Get bottle start position (from keyframe: 0.4, 0.2, 0.805)
        bottle_start = self.data.xpos[self.bottle_body_id].copy()

        # Generate trajectory
        traj_type = self.trajectory_type
        if options and "trajectory_type" in options:
            traj_type = options["trajectory_type"]

        self.trajectory = self._generate_trajectory(bottle_start[:2], traj_type)
        self._compute_arc_length()

        # Set goal position (end of trajectory)
        self.goal_pos = np.array([
            self.trajectory[-1, 0],
            self.trajectory[-1, 1],
            bottle_start[2]
        ], dtype=np.float32)

        # Reset state
        self.prev_progress = 0.0
        self.contact_made = False
        self.episode_length = 0

        print(f"\n{'=' * 50}")
        print(f"NEW EPISODE - Trajectory: {traj_type}")
        print(f"Bottle start: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"Trajectory end: ({self.trajectory[-1, 0]:.2f}, {self.trajectory[-1, 1]:.2f})")
        print(f"Path length: {self.total_arc_length:.2f}m")
        print(f"{'=' * 50}")

        obs = self._get_obs()

        return obs, {}

    # ==================================================================
    #                      STEP
    # ==================================================================
    def step(self, action):
        """
        Execute one step.

        Action: 7 joint velocity deltas (what the agent controls)
        The agent learns the force profile implicitly through
        learning how to move to push effectively.
        """
        # Apply action to joints
        step_size = 0.03
        self.current_ctrl = np.clip(
            self.current_ctrl + action * step_size,
            self.act_low,
            self.act_high
        )
        self.data.ctrl[:7] = self.current_ctrl

        # Step simulation
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Get observation and reward
        obs = self._get_obs()
        reward, info = self._get_reward()

        self.episode_length += 1

        # Check termination
        terminated = (
                info.get("is_success", False) or
                info.get("bottle_fallen", False) or
                info.get("off_path", False)
        )
        truncated = self.episode_length >= self.max_episode_length

        return obs, reward, terminated, truncated, info

    # ==================================================================
    #                      RENDER
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