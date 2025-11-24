import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os


class PandaPushEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "panda_robot.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.home_key_id = self.model.key("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # --- IMPORTANT BODY/SITE IDs ---
        self.bottle_body_id = self.model.body("bottle").id
        self.gripper_site_id = self.model.site("gripper_site").id
        # NOTE: self.target_site_id is kept for observation, but not used in reward.
        self.target_site_id = self.model.site("goal").id

        # We need the joint address because the bottle is DYNAMIC (has a free joint)
        self.bottle_joint_adr = self.model.jnt("bottle_joint").qposadr[0]

        # --- Define Action and Observation Spaces ---
        actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = actuator_ranges[:, 0]
        self.act_high = actuator_ranges[:, 1]

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # Observation size remains 23 (Robot QPos/QVel + Gripper/Bottle/Goal Pos)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0

    def _get_obs(self):
        """Assembles the observation vector."""
        # Indices [7:14] and [6:13] correctly skip the bottle's 7 DOFs (0-6)
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        goal_pos = self.data.site_xpos[self.target_site_id]

        return np.concatenate([
            self.data.qpos[7:14],  # Arm Joint Positions (7 values)
            self.data.qvel[6:13],  # Arm Joint Velocities (7 values)
            gripper_pos,  # Gripper Site Position (3 values)
            bottle_pos,  # Bottle Position (3 values)
            goal_pos  # Goal Position (3 values)
        ]).astype(np.float32)

    def _get_reward(self):
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        goal_pos = self.data.site_xpos[self.target_site_id]

        # --- 1. Calculate Push Direction (Bottle → Goal) ---
        bottle_xy = bottle_pos[:2]
        goal_xy = goal_pos[:2]

        push_direction = goal_xy - bottle_xy
        push_direction_norm = np.linalg.norm(push_direction)

        if push_direction_norm < 1e-6:
            unit_push_direction = np.array([1.0, 0.0])
        else:
            unit_push_direction = push_direction / push_direction_norm

        # --- 2. Define Pre-Target Position ---
        approach_distance = 0.20  # 20cm behind bottle
        pre_target_xy = bottle_xy - unit_push_direction * approach_distance
        target_height = 0.85  # Fixed height for pushing
        pre_target = np.array([pre_target_xy[0], pre_target_xy[1], target_height])

        # --- 3. POSITION REWARDS (XY + Z) ---

        # 3a. XY Position
        gripper_xy = gripper_pos[:2]
        dist_xy = np.linalg.norm(gripper_xy - pre_target_xy)

        xy_reward = 100.0 * np.exp(-3.0 * dist_xy) - 20.0 * dist_xy

        if dist_xy < 0.15:
            xy_reward += 100.0
        if dist_xy < 0.10:
            xy_reward += 150.0
        if dist_xy < 0.05:
            xy_reward += 250.0

        # 3b. Z Position
        gripper_z = gripper_pos[2]
        height_error = abs(gripper_z - target_height)

        z_reward = 80.0 * np.exp(-5.0 * height_error) - 30.0 * height_error

        if height_error < 0.10:
            z_reward += 80.0
        if height_error < 0.05:
            z_reward += 120.0
        if height_error < 0.03:
            z_reward += 200.0

        # 3c. Combined position bonus
        dist_3d = np.linalg.norm(gripper_pos - pre_target)
        if dist_3d < 0.08:
            combined_bonus = 300.0
        elif dist_3d < 0.12:
            combined_bonus = 150.0
        else:
            combined_bonus = 0.0

        position_reward = xy_reward + z_reward + combined_bonus

        # --- 4. FULL ORIENTATION CONTROL ---
        # Get gripper's orientation matrix (3x3 rotation matrix)
        gripper_mat = self.data.site_xmat[self.gripper_site_id].reshape(3, 3)

        # Extract all three axes
        gripper_x_axis = gripper_mat[:, 0]  # X-axis (forward/backward)
        gripper_y_axis = gripper_mat[:, 1]  # Y-axis (left/right)
        gripper_z_axis = gripper_mat[:, 2]  # Z-axis (up/down)

        # Create 3D push direction
        push_direction_3d = np.array([unit_push_direction[0], unit_push_direction[1], 0.0])
        push_direction_3d = push_direction_3d / (np.linalg.norm(push_direction_3d) + 1e-6)

        # Calculate alignments (MOVED OUTSIDE conditional) ← FIX
        y_alignment = np.dot(gripper_y_axis[:2], unit_push_direction)

        desired_z_direction = np.array([0.0, 0.0, -1.0])
        z_alignment = np.dot(gripper_z_axis, desired_z_direction)

        x_axis_z_component = abs(gripper_x_axis[2])

        orientation_reward = 0.0

        # Only apply orientation rewards when position is reasonable
        if dist_3d < 0.20:  # Within 20cm of target

            # --- 4a. Y-axis Alignment with Push Direction ---
            y_alignment_reward = 120.0 * y_alignment

            if y_alignment > 0.9:
                y_alignment_reward += 80.0
            if y_alignment > 0.95:
                y_alignment_reward += 120.0
            if y_alignment > 0.98:
                y_alignment_reward += 200.0

            # --- 4b. Z-axis Should Point Down (Stable Vertical) ---
            z_alignment_reward = 100.0 * z_alignment

            # Bonus for keeping gripper pointing down
            if z_alignment > 0.9:  # Within ~25 degrees of vertical
                z_alignment_reward += 60.0
            if z_alignment > 0.95:  # Within ~18 degrees
                z_alignment_reward += 100.0
            if z_alignment > 0.98:  # Within ~11 degrees - very stable!
                z_alignment_reward += 150.0

            # --- 4c. X-axis Should Be Horizontal (No Roll) ---
            x_stability_penalty = -50.0 * x_axis_z_component  # Penalty if X-axis tilts up/down

            if x_axis_z_component < 0.1:  # X-axis is mostly horizontal
                x_stability_bonus = 50.0
            else:
                x_stability_bonus = 0.0

            # --- 4d. Combined Orientation Reward ---
            orientation_reward = (y_alignment_reward +
                                  z_alignment_reward +
                                  x_stability_penalty +
                                  x_stability_bonus)

            # Extra bonus for perfect orientation (all axes correct)
            if y_alignment > 0.95 and z_alignment > 0.95 and x_axis_z_component < 0.1:
                orientation_reward += 300.0  # Big bonus for stable, aligned EE

        # --- 5. COLLISION AVOIDANCE ---
        dist_xy_to_bottle = np.linalg.norm(gripper_xy - bottle_xy)
        bottle_radius = 0.03

        collision_penalty = 0.0

        if dist_xy_to_bottle < (bottle_radius + 0.15):
            collision_penalty = -5.0
        if dist_xy_to_bottle < (bottle_radius + 0.10):
            collision_penalty = -20.0
        if dist_xy_to_bottle < (bottle_radius + 0.05):
            collision_penalty = -100.0
        if dist_xy_to_bottle < (bottle_radius + 0.02):
            collision_penalty = -400.0

        # --- 6. WRIST STABILIZATION ---
        wrist_joint_idx = 12
        wrist_position = self.data.qpos[wrist_joint_idx]
        home_wrist_angle = -0.7853
        wrist_deviation = abs(wrist_position - home_wrist_angle)

        wrist_position_penalty = -15.0 * wrist_deviation

        wrist_velocity = abs(self.data.qvel[12])
        wrist_velocity_penalty = -3.0 * wrist_velocity

        wrist_action = abs(self.data.ctrl[6])
        wrist_action_penalty = -5.0 * wrist_action

        # --- 7. Smooth Movement ---
        arm_velocity_penalty = -0.001 * np.linalg.norm(self.data.qvel[6:12])
        arm_action_penalty = -0.001 * np.linalg.norm(self.data.ctrl[0:6])

        # --- Total Reward ---
        reward = (position_reward +
                  orientation_reward +  # Full orientation control!
                  collision_penalty +
                  wrist_position_penalty +
                  wrist_velocity_penalty +
                  wrist_action_penalty +
                  arm_velocity_penalty +
                  arm_action_penalty)

        # --- DEBUG ---
        if self.episode_length % 100 == 0:
            print(f"\n=== Step {self.episode_length} ===")
            print(f"Position: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
            print(f"Target:   [{pre_target[0]:.3f}, {pre_target[1]:.3f}, {pre_target[2]:.3f}]")
            print(f"XY dist: {dist_xy:.3f}m, Z error: {height_error:.3f}m, 3D: {dist_3d:.3f}m")
            print(f"\n--- Orientation ---")
            print(f"X-axis: [{gripper_x_axis[0]:.3f}, {gripper_x_axis[1]:.3f}, {gripper_x_axis[2]:.3f}]")
            print(f"Y-axis: [{gripper_y_axis[0]:.3f}, {gripper_y_axis[1]:.3f}, {gripper_y_axis[2]:.3f}]")
            print(f"Z-axis: [{gripper_z_axis[0]:.3f}, {gripper_z_axis[1]:.3f}, {gripper_z_axis[2]:.3f}]")
            print(f"Y alignment (push): {y_alignment:.3f} (target: 1.0)")
            print(f"Z alignment (down): {z_alignment:.3f} (target: 1.0)")
            print(f"X tilt (horizontal): {x_axis_z_component:.3f} (target: 0.0)")
            print(f"\n--- Rewards ---")
            print(f"Position: {position_reward:.1f}, Orientation: {orientation_reward:.1f}")
            print(f"Wrist angle: {wrist_position:.3f} rad (home: {home_wrist_angle:.3f})")
            print(f"Total: {reward:.1f}")

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mujoco.mj_forward(self.model, self.data)

        # 💡 Settle the bottle: Since the bottle is dynamic and starts on the table (Z=0.88),
        # we still run a few steps to ensure it is completely settled before the episode begins.
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.episode_length = 0

        observations = self._get_obs()
        info = {}

        return observations, info

    def step(self, action):
        max_step_size = 0.01

        delta_action = action * max_step_size

        current_qpos = self.data.qpos[7:14]
        new_target_qpos = current_qpos + delta_action
        clipped_target_qpos = np.clip(new_target_qpos, self.act_low, self.act_high)

        self.data.ctrl[:7] = clipped_target_qpos

        # Keep the gripper open
        # self.data.ctrl[7] = 255.0

        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        observations = self._get_obs()
        reward = self._get_reward()

        self.episode_length += 1

        # 💡 Termination/Truncation: The task is pure reaching, so we only use the episode limit.
        terminated = False
        truncated = bool(self.episode_length >= 500)

        # --- NEW TERMINATION LOGIC ---

        # bottle_pos = self.data.xpos[self.bottle_body_id]
        # gripper_pos = self.data.site_xpos[self.gripper_site_id]
        #
        # # Recalculate collision distance
        # dist_gripper_to_bottle_side = np.linalg.norm(gripper_pos[:2] - bottle_pos[:2]) - 0.03
        #
        # # The collision check must be consistent with the reward function's threshold
        # COLLISION_THRESHOLD = 0.01  # 1 cm clearance
        #
        # if dist_gripper_to_bottle_side < COLLISION_THRESHOLD:
        #     # 💡 Terminate immediately on collision/near-collision
        #     terminated = True

        # -----------------------------

        info = {}

        return observations, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None