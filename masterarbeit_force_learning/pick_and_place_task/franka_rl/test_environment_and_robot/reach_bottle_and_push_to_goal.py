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

        # Observation space: 25 dimensions
        # [joint_pos(7), joint_vel(7), hand_pos(3), relative_vec(3), hand_quat(4), phase_bit(1)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0

        #specific to bottle
        self.bottle_geom_offset_z = 0.08
        self.bottle_radius = 0.03
        self.bottle_half_height = 0.08

        # State Variables
        self.pushing_phase = False  # The Switch
        self.goal_pos = np.array([0.4, -0.2, 0.80])  # Final Goal
        self.transition_bonus_given = False

        # Computed at reset
        self.reach_target = None
        self.push_direction = None
        self.home_hand_quat = None

        # for delta-based rewards (progress tracking)
        self.prev_bottle_to_goal_distance = None
        self.prev_hand_to_target_distance = None

    def _get_obs(self):
        hand_pos = self.data.xpos[self.hand_body_id]
        hand_quat = self.data.xquat[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]

        # === OBSERVATION SWITCHING ===
        if self.pushing_phase:
            # PHASE 2: Input vector is relative to the GOAL (for the bottle)
            # However, providing vector from Bottle -> Goal is usually better for pushing
            relative_vec = self.goal_pos - bottle_pos
            phase_bit = 1.0
        else:
            # PHASE 1: Input vector is relative to the REACH TARGET (for the hand)
            reach_target = self._compute_reach_target()
            relative_vec = reach_target - hand_pos
            phase_bit = 0.0

        return np.concatenate([
            self.data.qpos[7:14],
            self.data.qvel[6:13],
            hand_pos,
            relative_vec,
            hand_quat,
            [phase_bit]  # <--- Neural Network now knows what task it is doing
        ]).astype(np.float32)

    # def _get_target_pos(self):
    #     bottle_pos = self.data.xpos[self.bottle_body_id]
    #     target_goal_pos = np.array([0.4, -0.2, 0.80])
    #
    #     vec = target_goal_pos - bottle_pos
    #     vec[2] = 0  # ignore z
    #     dist = np.linalg.norm(vec)
    #
    #     if dist < 1e-6:
    #         direction = np.array([1.0, 0.0, 0.0])
    #     else:
    #         direction = vec / dist
    #
    #     reach_target = bottle_pos - (direction * 0.18)
    #     reach_target[2] = 0.95  # table (0.80) + offset
    #
    #     return reach_target, bottle_pos.copy(), direction
    def _compute_push_direction(self):
        bottle_pos = self.data.xpos[self.bottle_body_id]
        goal_vector = self.goal_pos - bottle_pos
        goal_vector[2] = 0 # only xy-plane
        distance_goal_vector = np.linalg.norm(goal_vector)
        if distance_goal_vector < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return goal_vector / distance_goal_vector

    def _compute_reach_target(self):
        bottle_body_pos = self.data.xpos[self.bottle_body_id]
        push_direction = self._compute_push_direction()

        bottle_center_z = bottle_body_pos[2] + self.bottle_geom_offset_z

        reach_target = bottle_body_pos.copy()
        reach_target[:2] -= push_direction[:2] * 0.15
        reach_target[2] = bottle_center_z + 0.02

        return reach_target


    def _get_reward(self):
        hand_pos = self.data.xpos[self.hand_body_id]
        hand_quat = self.data.xquat[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        home_quat = self.home_hand_quat

        # Control Penalty (Always active)
        reward_ctrl = -0.001 * np.square(self.data.ctrl[:7]).sum()

        # Orientation Calculation
        quat_diff1 = np.linalg.norm(hand_quat - home_quat)
        quat_diff2 = np.linalg.norm(hand_quat + home_quat)
        quat_error = min(quat_diff1, quat_diff2)

        info = {"is_success": False,
                "phase": "PUSH" if self.pushing_phase else "REACH",
                "should_transition": False
        }
        total_reward = 0.0

        # ==========================================================
        # LOGIC SPLIT: ARE WE REACHING OR PUSHING?
        # ==========================================================

        if not self.pushing_phase:
            # === PHASE 1: REACHING (Your working code) ===
            reach_target = self._compute_reach_target()
            hand_to_target_distance = np.linalg.norm(hand_pos - reach_target)

            # Delta-based progress reward
            progress_reward  = 0.0
            if self.prev_hand_to_target_distance is not None:
                progress = self.prev_hand_to_target_distance - hand_to_target_distance
                progress_reward = 10.0 * progress # Reward for getting closer
            self.prev_hand_to_target_distance = hand_to_target_distance

            EXP_COEFF = 50.0
            reward_distance = np.exp(-EXP_COEFF * hand_to_target_distance)

            proximity_gate = np.exp(-3.0 * hand_to_target_distance)
            reward_orientation = proximity_gate * (1.0 - np.tanh(5.0 * quat_error))

            # -- bonus for reaching TARGET --
            # reached_position = distance_to_target_3d < 0.08
            # reached_with_orientation = distance_to_target_3d < 0.05 and quat_error < 0.1

            bonus = 0.0
            if hand_to_target_distance < 0.10:
                bonus += 0.3
            if hand_to_target_distance < 0.06:
                bonus += 0.5
            if hand_to_target_distance < 0.04 and quat_error < 0.15:
                bonus += 1.0

            total_reward = (
                    progress_reward +
                    5.0 * reward_distance +
                    2.0 * reward_orientation +
                    reward_ctrl +
                    bonus
            )

            # Check for Transition
            if hand_to_target_distance < 0.05 and quat_error < 0.20:
                # We have arrived! Switch to Pushing.
                # We give a one-time "Handoff Bonus" to encourage this transition
                info["should_transition"] = True

            # Debug info
            info["reach_distance"] = hand_to_target_distance
            info["quat_error"] = quat_error

        else:
            # === PHASE 2: PUSHING ===
            # Current push direction (updated dynamically)
            push_direction = self._compute_push_direction()

            # Distance from bottle to goal (XY only)
            bottle_to_goal_distance = np.linalg.norm(bottle_pos[:2] - self.goal_pos[:2])

            # Delta-based progress reward
            progress_reward = 0.0
            if self.prev_bottle_to_goal_distance is not None:
                progress = self.prev_bottle_to_goal_distance - bottle_to_goal_distance
                progress_reward = 50.0 * progress #strong reward for moving bottle toward goal
            self.prev_bottle_to_goal_distance = bottle_to_goal_distance

            # Contact reward: hand should stay behind bottle
            # Account for bottle geometry: center is body_pos + geom_offset
            bottle_center_z = bottle_pos[2] + self.bottle_geom_offset_z
            contact_point = bottle_pos.copy()
            contact_point[:2] -= push_direction[:2] * 0.08
            contact_point[2] = bottle_center_z
            hand_to_contact_distance = np.linalg.norm(hand_pos - contact_point)
            reward_contact = np.exp(-8.0 * hand_to_contact_distance)

            # Orientation reward (keep gripper pointing down)
            reward_orientation = 1.0 - np.tanh(3.0 * quat_error)

            # Goal proximity shaping
            reward_goal_proximity = np.exp(-3.0 * bottle_to_goal_distance)

            total_reward = (
                    progress_reward +               # Primary learning signal
                    5.0 * reward_goal_proximity +   # Shaping
                    4.0 * reward_contact +          # keeping contact
                    2.0 * reward_orientation +      # Keep good orientation
                    reward_ctrl
            )

            # Check for Final Success
            if bottle_to_goal_distance < 0.08:
                total_reward += 50.0  # JACKPOT
                info["is_success"] = True

            #Debug info
            info["bottle_to_goal"] = bottle_to_goal_distance
            info["hand_to_contact"] = hand_to_contact_distance

        return total_reward, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        self.current_ctrl = self.data.qpos[7:14].copy()

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # home orientation
        self.home_hand_quat = self.data.xquat[self.hand_body_id].copy()

        # Calculate initial targets
        self.push_direction = self._compute_push_direction()
        self.reach_target = self._compute_reach_target()

        # reset state
        self.pushing_phase = False
        self.transition_bonus_given = False
        self.episode_length = 0

        # reset progress tracking
        bottle_pos = self.data.xpos[self.bottle_body_id]
        hand_pos = self.data.xpos[self.hand_body_id]
        self.prev_bottle_to_goal_distance = np.linalg.norm(bottle_pos[:2] - self.goal_pos[:2])
        self.prev_hand_to_target_distance = np.linalg.norm(hand_pos - self.reach_target)

        return self._get_obs(), {}

    def step(self, action):
        step_size = 0.005  # reduced step size for precision
        self.current_ctrl = np.clip(
            self.current_ctrl + (action * step_size),
            self.act_low,
            self.act_high
        )
        self.data.ctrl[:7] = self.current_ctrl

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # obs = self._get_obs()
        reward, info = self._get_reward()
        # self.episode_length += 1

        # === HANDLING PHASE TRANSITION ===
        if not self.pushing_phase and info.get("should_transition", False):
            self.pushing_phase = True
            self.transition_bonus_given = True
            bottle_pos = self.data.xpos[self.bottle_body_id]
            self.prev_bottle_to_goal_distance = np.linalg.norm(bottle_pos[:2] - self.goal_pos[:2])
            reward += 10.0
            print(f"[Step {self.episode_length}] Transitioning to PUSH phase!")

            #print("Transitioning to PUSH phase!")
            #self.pushing_phase = True
        obs = self._get_obs()
        self.episode_length += 1

        terminated = info["is_success"]
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