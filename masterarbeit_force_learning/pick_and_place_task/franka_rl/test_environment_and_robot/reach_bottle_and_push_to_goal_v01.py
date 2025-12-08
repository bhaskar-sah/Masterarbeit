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
        self.bottle_body_id = self.model.body("bottle").id
        self.hand_body_id = self.model.body("hand").id

        # Action space
        self.actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = self.actuator_ranges[:, 0]
        self.act_high = self.actuator_ranges[:, 1]
        self.current_ctrl = np.zeros(7)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation space: 27 dimensions (added push_direction xy)
        # [joint_pos(7), joint_vel(7), hand_pos(3), relative_vec(3), hand_quat(4), push_dir(2), phase_bit(1)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0

        # Bottle geometry (from XML)
        self.bottle_geom_offset_z = 0.08
        self.bottle_radius = 0.03
        self.bottle_half_height = 0.08

        # State Variables
        self.pushing_phase = False
        self.goal_pos = np.array([0.4, -0.2, 0.80])
        self.transition_bonus_given = False

        # Computed at reset
        self.reach_target = None
        self.push_direction = None
        self.home_hand_quat = None

        # For delta-based rewards
        self.prev_bottle_to_goal_distance = None
        self.prev_hand_to_target_distance = None

    def _compute_push_direction(self):
        """Direction from bottle to goal (XY plane only, normalized)."""
        bottle_pos = self.data.xpos[self.bottle_body_id]
        goal_vector = self.goal_pos - bottle_pos
        goal_vector[2] = 0
        distance = np.linalg.norm(goal_vector)
        if distance < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return goal_vector / distance

    def _compute_reach_target(self):
        """Position BEHIND the bottle, on the line from goal through bottle."""
        bottle_body_pos = self.data.xpos[self.bottle_body_id]
        push_direction = self._compute_push_direction()

        reach_target = bottle_body_pos.copy()
        reach_target[:2] -= push_direction[:2] * 0.12  # 12cm behind bottle
        reach_target[2] = bottle_body_pos[2] + 0.05  # Pushing height

        return reach_target

    def _get_hand_x_axis(self):
        """
        Get the hand's local X-axis in world coordinates.

        The rotation matrix is stored in xmat as a flattened 3x3 matrix.
        Columns of the rotation matrix are the local axes in world frame:
        - Column 0: local X-axis in world frame
        - Column 1: local Y-axis in world frame
        - Column 2: local Z-axis in world frame
        """
        hand_rot_matrix = self.data.xmat[self.hand_body_id].reshape(3, 3)
        hand_x_axis = hand_rot_matrix[:, 0]  # First column = local X in world frame
        return hand_x_axis

    def _get_hand_z_axis(self):
        """Get the hand's local Z-axis in world coordinates."""
        hand_rot_matrix = self.data.xmat[self.hand_body_id].reshape(3, 3)
        hand_z_axis = hand_rot_matrix[:, 2]  # Third column = local Z in world frame
        return hand_z_axis

    def _compute_x_axis_alignment(self, push_direction):
        """
        Compute how well the hand's negative X-axis aligns with the push direction.

        The hand's -X axis is the pushing direction (toward the gripper fingers).
        We want: (-hand_x_axis) parallel to push_direction, SAME direction

        Perfect alignment: dot(-hand_x, push_dir) = +1
        Which is equivalent to: dot(hand_x, push_dir) = -1

        Returns value in [0, 1] where 1 = perfectly aligned.
        """
        hand_x_axis = self._get_hand_x_axis()

        # Project to XY plane and normalize
        hand_x_xy = hand_x_axis[:2]
        hand_x_xy_norm = np.linalg.norm(hand_x_xy)

        if hand_x_xy_norm < 1e-6:
            return 0.0

        hand_x_xy = hand_x_xy / hand_x_xy_norm
        push_dir_xy = push_direction[:2]

        # We want -X aligned with push_direction
        # dot(-hand_x, push_dir) = -dot(hand_x, push_dir)
        # Perfect: dot(hand_x, push_dir) = -1 → alignment = 1
        # Opposite: dot(hand_x, push_dir) = +1 → alignment = 0
        dot_product = np.dot(hand_x_xy, push_dir_xy)

        # Convert: when dot = -1, alignment = 1; when dot = +1, alignment = 0
        alignment = (-dot_product + 1.0) / 2.0

        return alignment

    def _compute_z_axis_down(self):
        """
        Check if the hand's Z-axis points downward (gripper pointing down).

        Returns value in [0, 1] where 1 = perfectly pointing down.
        """
        hand_z_axis = self._get_hand_z_axis()

        # We want Z to point down: [0, 0, -1]
        # dot(hand_z, [0,0,-1]) = -hand_z[2]
        # If hand_z = [0,0,-1], then -hand_z[2] = 1
        downward_alignment = -hand_z_axis[2]

        # Clamp to [0, 1]
        return max(0.0, downward_alignment)

    def _check_bottle_fallen(self):
        """Check if bottle has fallen completely."""
        bottle_quat = self.data.qpos[3:7]
        w, x, y, z = bottle_quat
        up_z = 1 - 2 * (x * x + y * y)
        return up_z < 0.35  # Fallen if tilted > 70 degrees

    def _compute_position_alignment(self, hand_pos, bottle_pos, push_direction):
        """
        Check if hand is positioned behind bottle (opposite to push direction).
        Returns value in [0, 1] where 1 = perfectly behind.
        """
        bottle_to_hand = hand_pos[:2] - bottle_pos[:2]
        dist = np.linalg.norm(bottle_to_hand)

        if dist < 1e-6:
            return 0.0

        bottle_to_hand_normalized = bottle_to_hand / dist

        # Hand should be OPPOSITE to push direction
        alignment = -np.dot(bottle_to_hand_normalized, push_direction[:2])

        return (alignment + 1.0) / 2.0

    def _get_obs(self):
        hand_pos = self.data.xpos[self.hand_body_id]
        hand_quat = self.data.xquat[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        push_direction = self._compute_push_direction()

        if self.pushing_phase:
            relative_vec = self.goal_pos - bottle_pos
            phase_bit = 1.0
        else:
            reach_target = self._compute_reach_target()
            relative_vec = reach_target - hand_pos
            phase_bit = 0.0

        return np.concatenate([
            self.data.qpos[7:14],  # Joint positions (7)
            self.data.qvel[6:13],  # Joint velocities (7)
            hand_pos,  # Hand position (3)
            relative_vec,  # Task-relevant vector (3)
            hand_quat,  # Hand orientation (4)
            push_direction[:2],  # Push direction XY (2) - this is new one don't forget
            [phase_bit]  # Phase indicator (1)
        ]).astype(np.float32)

    def _get_reward(self):
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]

        # Control penalty
        reward_ctrl = -0.001 * np.square(self.data.ctrl[:7]).sum()

        push_direction = self._compute_push_direction()

        # === ORIENTATION REWARDS ===
        # 1. Z-axis pointing down (gripper down)
        z_down_reward = self._compute_z_axis_down()

        # 2. X-axis aligned with push direction (yaw alignment)
        x_align_reward = self._compute_x_axis_alignment(push_direction)

        info = {
            "is_success": False,
            "phase": "PUSH" if self.pushing_phase else "REACH",
            "should_transition": False,
            "bottle_fallen": False,
            "z_down": z_down_reward,
            "x_align": x_align_reward
        }

        # Check for bottle fall
        if self._check_bottle_fallen():
            info["bottle_fallen"] = True
            return -5.0, info

        total_reward = 0.0

        if not self.pushing_phase:
            # ==================== PHASE 1: REACHING ====================
            reach_target = self._compute_reach_target()
            hand_to_target_distance = np.linalg.norm(hand_pos - reach_target)

            # Delta-based progress
            progress_reward = 0.0
            if self.prev_hand_to_target_distance is not None:
                progress = self.prev_hand_to_target_distance - hand_to_target_distance
                progress_reward = 10.0 * progress
            self.prev_hand_to_target_distance = hand_to_target_distance

            # Distance reward
            reward_distance = np.exp(-30.0 * hand_to_target_distance)

            # Position alignment (behind bottle)
            position_alignment = self._compute_position_alignment(hand_pos, bottle_pos, push_direction)

            # Gate Z-down by proximity (only important when close)
            proximity_gate = np.exp(-3.0 * hand_to_target_distance)
            reward_z_down = proximity_gate * z_down_reward

            # X-alignment should ALWAYS be active - not gated!
            # Robot needs to rotate correctly even while approaching
            reward_x_align = x_align_reward  # NOT gated!

            # Bonuses
            bonus = 0.0
            if hand_to_target_distance < 0.10:
                bonus += 0.2
            if hand_to_target_distance < 0.06:
                bonus += 0.3
            # Big bonus for good position AND orientation
            if hand_to_target_distance < 0.05 and z_down_reward > 0.8 and x_align_reward > 0.8:
                bonus += 2.0

            total_reward = (
                    progress_reward +
                    10.0 * reward_distance +
                    2.0 * reward_z_down +  # Gripper pointing down (gated)
                    3.0 * reward_x_align +  # X-axis aligned - ALWAYS ACTIVE, HIGH WEIGHT!
                    2.0 * position_alignment +  # Behind bottle
                    reward_ctrl +
                    bonus
            )

            # Transition requires STRICT X-alignment!
            if (hand_to_target_distance < 0.06 and
                    z_down_reward > 0.7 and
                    x_align_reward > 0.75 and  # Must be well aligned!
                    position_alignment > 0.5):
                info["should_transition"] = True

            info["reach_distance"] = hand_to_target_distance
            info["position_alignment"] = position_alignment

        else:
            # ==================== PHASE 2: PUSHING ====================

            # Distance from bottle to goal
            bottle_to_goal_distance = np.linalg.norm(bottle_pos[:2] - self.goal_pos[:2])

            # Delta-based progress
            progress_reward = 0.0
            if self.prev_bottle_to_goal_distance is not None:
                progress = self.prev_bottle_to_goal_distance - bottle_to_goal_distance
                progress_reward = 100.0 * progress
            self.prev_bottle_to_goal_distance = bottle_to_goal_distance

            # Position alignment (stay behind bottle)
            position_alignment = self._compute_position_alignment(hand_pos, bottle_pos, push_direction)

            # Contact distance
            # donn't
            hand_to_bottle_dist = np.linalg.norm(hand_pos[:2] - bottle_pos[:2]) # take the FULL DISTANCE AND not just the xy axis.
            reward_contact = np.exp(-8.0 * hand_to_bottle_dist)

            # Height reward
            target_height = bottle_pos[2] + 0.05
            height_error = abs(hand_pos[2] - target_height)
            reward_height = np.exp(-15.0 * height_error)

            # Goal proximity
            reward_goal_proximity = np.exp(-3.0 * bottle_to_goal_distance)

            total_reward = (
                    progress_reward +
                    8.0 * reward_goal_proximity +
                    10.0 * x_align_reward +  # Keep X aligned during push! CRITICAL!
                    4.0 * z_down_reward +  # Keep gripper down
                    4.0 * position_alignment +  # Stay behind bottle
                    3.0 * reward_contact +
                    2.0 * reward_height +
                    reward_ctrl
            )

            # Success
            if bottle_to_goal_distance < 0.08:
                total_reward += 50.0
                info["is_success"] = True

            info["bottle_to_goal"] = bottle_to_goal_distance
            info["position_alignment"] = position_alignment
            info["hand_to_bottle"] = hand_to_bottle_dist

        return total_reward, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        self.current_ctrl = self.data.qpos[7:14].copy()

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.home_hand_quat = self.data.xquat[self.hand_body_id].copy()
        self.push_direction = self._compute_push_direction()
        self.reach_target = self._compute_reach_target()

        self.pushing_phase = False
        self.transition_bonus_given = False
        self.episode_length = 0

        bottle_pos = self.data.xpos[self.bottle_body_id]
        hand_pos = self.data.xpos[self.hand_body_id]
        self.prev_bottle_to_goal_distance = np.linalg.norm(bottle_pos[:2] - self.goal_pos[:2])
        self.prev_hand_to_target_distance = np.linalg.norm(hand_pos - self.reach_target)

        # Debug info
        print(f"[Reset] Push direction: {self.push_direction[:2]}")
        print(f"[Reset] Hand X-axis: {self._get_hand_x_axis()[:2]}")
        print(f"[Reset] X-alignment: {self._compute_x_axis_alignment(self.push_direction):.2f}")

        return self._get_obs(), {}

    def step(self, action):
        step_size = 0.005
        self.current_ctrl = np.clip(
            self.current_ctrl + (action * step_size),
            self.act_low,
            self.act_high
        )
        self.data.ctrl[:7] = self.current_ctrl

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        reward, info = self._get_reward()

        # Phase transition
        if not self.pushing_phase and info.get("should_transition", False):
            self.pushing_phase = True
            self.transition_bonus_given = True
            bottle_pos = self.data.xpos[self.bottle_body_id]
            self.prev_bottle_to_goal_distance = np.linalg.norm(bottle_pos[:2] - self.goal_pos[:2])
            reward += 10.0
            print(f"[Step {self.episode_length}] → PUSH! X-align: {info['x_align']:.2f}, Z-down: {info['z_down']:.2f}")

        obs = self._get_obs()
        self.episode_length += 1

        terminated = info["is_success"] or info.get("bottle_fallen", False)
        truncated = self.episode_length >= 500

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