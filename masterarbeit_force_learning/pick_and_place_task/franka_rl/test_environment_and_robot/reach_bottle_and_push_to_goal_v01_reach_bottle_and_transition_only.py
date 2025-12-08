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
        self.goal_site_id = self.model.site("goal").id

        # Action space
        self.actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = self.actuator_ranges[:, 0]
        self.act_high = self.actuator_ranges[:, 1]
        self.current_ctrl = np.zeros(7)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

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
        # self.goal_pos = np.array([0.8, 0.0, 0.80])
        self.transition_bonus_given = False

        # Computed at reset
        self.reach_target = None
        self.push_direction = None
        self.home_hand_quat = None

        # For delta-based rewards
        self.prev_bottle_to_goal_distance = None
        self.prev_hand_to_target_distance = None

    @property
    def goal_pos(self):
        return self.data.site_xpos[self.goal_site_id]

    def _compute_push_direction(self):
        """Direction from bottle to goal (XY plane only, normalized)."""
        bottle_pos = self.data.xpos[self.bottle_body_id]
        goal_vector = self.goal_pos - bottle_pos
        goal_vector[2] = 0
        distance = np.linalg.norm(goal_vector)
        if distance < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = goal_vector / distance
        return direction

    def _compute_reach_target(self):
        """Position BEHIND the bottle, on the line from goal through bottle."""
        bottle_pos = self.data.xpos[self.bottle_body_id]
        push_direction = self._compute_push_direction()

        reach_target = bottle_pos.copy()
        reach_target[:2] -= push_direction[:2] * 0.15  # 12cm behind bottle
        reach_target[2] = bottle_pos[2] + 0.15  # Pushing height

        return reach_target

    def _get_hand_x_axis(self):
        hand_rot_matrix = self.data.xmat[self.hand_body_id].reshape(3, 3)
        hand_x_axis = hand_rot_matrix[:, 0]  # First column = local X in world frame
        return hand_x_axis

    def _get_hand_z_axis(self):
        """Get the hand's local Z-axis in world coordinates."""
        hand_rot_matrix = self.data.xmat[self.hand_body_id].reshape(3, 3)
        hand_z_axis = hand_rot_matrix[:, 2]  # Third column = local Z in world frame
        return hand_z_axis

    def _compute_x_axis_alignment(self, push_direction):
        hand_x_axis = self._get_hand_x_axis()

        # Project to XY plane and normalize
        hand_x_xy = hand_x_axis[:2]
        hand_x_xy_norm = np.linalg.norm(hand_x_xy)

        if hand_x_xy_norm < 1e-6:
            return 0.0

        hand_x_xy = hand_x_xy / hand_x_xy_norm
        push_dir_xy = push_direction[:2]
        dot_product = np.dot(hand_x_xy, push_dir_xy)

        alignment = (-dot_product + 1.0) / 2.0

        return alignment

    def _compute_z_axis_down(self):
        hand_z_axis = self._get_hand_z_axis()
        downward_alignment = -hand_z_axis[2]

        return max(0.0, downward_alignment)

    def _check_bottle_fallen(self):
        """Check if bottle has fallen completely."""
        bottle_quat = self.data.qpos[3:7]
        w, x, y, z = bottle_quat
        up_z = 1 - 2 * (x * x + y * y)
        return up_z < 0.35  # Fallen if tilted > 70 degrees

    def _compute_position_alignment(self, hand_pos, bottle_pos, push_direction):
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
        info = {"is_success": False,
                "should_transition": False,
                "phase": "REACH" if not self.pushing_phase else "PUSH",
                "x_align": 0.0,
                "z_down": 0.0
        }
        hand_pos = self.data.xpos[self.hand_body_id]
        hand_quat = self.data.xquat[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_quat = self.data.xquat[self.bottle_body_id] # get bottle orientation

        # -- Dynamic target logic
        push_direction = self._compute_push_direction()
        if not self.pushing_phase:
            # phase1: target is behind the bottle (approach)
            reach_target = bottle_pos.copy()
            reach_target[:2] -= push_direction[:2] * 0.15
            reach_target = bottle_pos[2] + 0.15
        else:
            #phase2: target is the bottle(contact and push)
            reach_target = bottle_pos.copy()
            reach_target[2] = bottle_pos[2] + 0.15 # keep height, but move xy forward

        # Control penalty
        reward_ctrl = -0.0001 * np.square(self.data.ctrl[:7]).sum()
        ##################
        # Calculate Bottle-to-Goal distance for Pushing Reward
        bottle_dist = np.linalg.norm(bottle_pos[:2] - self.goal_pos[:2])

        # Initialize prev_bottle_dist if it's None (first step)
        if self.prev_bottle_to_goal_distance is None:
            self.prev_bottle_to_goal_distance = bottle_dist
        ##################

        total_reward = 0.0

        if not self.pushing_phase:
            # ==================== PHASE 1: REACHING ====================
            hand_to_target_distance_3d = np.linalg.norm(hand_pos - reach_target)
            # hand_to_target_distance_xy = np.linalg.norm(hand_pos[:2] - reach_target[:2])
            # height_error = np.abs(hand_pos[2] - reach_target[2])

            # Distance reward
            EXP_COEFF = 2.0
            reward_distance = np.exp(-EXP_COEFF * hand_to_target_distance_3d)

            # -- ORIENTATION: Match home quaternion --
            home_quat = self.home_hand_quat
            quat_diff1 = np.linalg.norm(hand_quat - home_quat)
            quat_diff2 = np.linalg.norm(hand_quat + home_quat)
            quat_error = min(quat_diff1, quat_diff2)

            reward_orientation = 0.5 * (1.0 - np.tanh(1.0 * quat_error))

            total_reward = (
                10.0 * reward_distance +
                reward_orientation +
                reward_ctrl
            )

            # --- FORCE TRANSITION LOGIC ---
            # If we are close enough (12cm), switch phases.
            # We IGNORE quat_error here to prevent the deadlock.
            if hand_to_target_distance_3d < 0.12:
                info["should_transition"] = True
                total_reward += 5.0
                print(f" !!! TRANSITION: Close enough ({hand_to_target_distance_3d:.3f}) !!!")

            # Debug log
            if self.episode_length % 10 == 0:
                print(f"[Phase 1 Debug] Dist: {hand_to_target_distance_3d:.3f} (Req < 0.12)")
                print(f" Distance from home to pre_target: {hand_to_target_distance_3d:.3f}")
        else:
            # ==================== PHASE 2: align & PUSHING ====================
            # 1. Stay Near Target
            # hand_to_target_distance_3d = np.linalg.norm(hand_pos - reach_target)
            # reward_stay = np.exp(-10.0 * hand_to_target_distance_3d)

            # 1. pulls the hand to the bottle, forcing contact.
            dist_to_bottle = np.linalg.norm(hand_pos - reach_target)
            reward_contact = np.exp(-4.0 * dist_to_bottle)

            align_x = self._compute_x_axis_alignment(push_direction)
            align_z = self._compute_z_axis_down()
            ###############################################
            # 3. *** NEW: PUSH REWARD ***
            # Reward for actually moving the bottle closer to the goal
            progress = self.prev_bottle_to_goal_distance - bottle_dist
            # Big multiplier (100.0) so moving the bottle is the most profitable thing to do
            reward_push = 100.0 * progress
            ##############################################

            # 4. *** NEW: BOTTLE ORIENTATION REWARD ***
            # Penalize misalignment with bottle orientation
            # Calculate angle between hand X-axis and bottle X-axis
            # hand_x_axis = self._get_hand_x_axis()
            # bottle_x_axis = self.data.xmat[self.bottle_body_id].reshape(3, 3)[:, 0]
            # angle_diff = np.arccos(np.clip(np.dot(hand_x_axis, bottle_x_axis), -1.0, 1.0))
            # reward_bottle_align = -np.abs(angle_diff)  # Penalize misalignment

            total_reward = (
                6.0 * reward_contact +
                5.0 * align_x +
                1.0 * align_z +
            #######################
            reward_push +
            #######################
                reward_ctrl
            )
            ###############################
            # Log for debugging
            # info["x_align"] = align_x
            # info["z_down"] = align_z

            # if self.episode_length % 10 == 0:
            #    print(f"[Phase 2] Align X: {align_x:.3f} | Align Z: {align_z:.3f} | Dist: {hand_to_target_distance_3d:.3f}")
            ##################################
            # Success Check
            if bottle_dist < 0.05:
                info["is_success"] = True
                total_reward += 100.0
                print("\n *** GOAL REACHED! ***")

            # Update history
            self.prev_bottle_to_goal_distance = bottle_dist

            info["x_align"] = align_x

            if self.episode_length % 10 == 0:
                print(f"[Phase 2] Align X: {align_x:.3f} | Contact Dist: {dist_to_bottle:.3f} | Bottle Moved: {progress:.4f}")

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

        # bottle_pos = self.data.xpos[self.bottle_body_id]
        # hand_pos = self.data.xpos[self.hand_body_id]
        # self.prev_bottle_to_goal_distance = np.linalg.norm(bottle_pos[:2] - self.goal_pos[:2])
        # self.prev_hand_to_target_distance = np.linalg.norm(hand_pos - self.reach_target)

        # Debug info
        # print(f"[Reset] Push direction: {self.push_direction[:2]}")
        # print(f"[Reset] Hand X-axis: {self._get_hand_x_axis()[:2]}")
        # print(f"[Reset] X-alignment: {self._compute_x_axis_alignment(self.push_direction):.2f}")

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
        if not self.pushing_phase and info["should_transition"]:
            self.pushing_phase = True
            print(f"\n[Step {self.episode_length}] *** TRANSITION! Aligning to Goal... ***")

        obs = self._get_obs()
        self.episode_length += 1

        # Only terminate if Phase 2 is done (we haven't defined that yet) or timeout
        terminated = info["is_success"]  # This will be False for now, which is good
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