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

        # Body IDs
        self.bottle_body_id = self.model.body("bottle").id
        self.hand_body_id = self.model.body("hand").id

        # Action space: 7 robot joints
        self.actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = self.actuator_ranges[:, 0]
        self.act_high = self.actuator_ranges[:, 1]

        self.current_ctrl = np.zeros(7)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0

        # Will be set in reset()
        self.initial_bottle_pos = None
        self.reach_target = None

    def _get_obs(self):
        hand_pos = self.data.xpos[self.hand_body_id]
        relative_vec = self.reach_target - hand_pos
        hand_quat = self.data.xquat[self.hand_body_id]

        return np.concatenate([
            self.data.qpos[7:14],
            self.data.qvel[6:13],
            relative_vec,
            hand_quat
        ]).astype(np.float32)

    def _compute_reach_target(self):
        """Compute reach target: 0.18m along +Y from bottle, at Z=0.95"""
        bottle_pos = self.data.xpos[self.bottle_body_id].copy()

        # Target is 0.18m along +Y axis from bottle
        reach_target = bottle_pos.copy()
        reach_target[0] = bottle_pos[0]  # Same X as bottle (0.4)
        reach_target[1] = bottle_pos[1] + 0.18  # 0.18m along +Y (0.2 + 0.18 = 0.38)
        reach_target[2] = 0.95  # Fixed height

        return bottle_pos, reach_target

    def _get_reward(self):
        hand_pos = self.data.xpos[self.hand_body_id]
        hand_mat = self.data.xmat[self.hand_body_id].reshape(3, 3)

        reach_target = self.reach_target

        # === 1. DISTANCE TO TARGET (XY) ===
        distance_xy = np.linalg.norm(hand_pos[:2] - reach_target[:2])
        reward_dist = 1.0 - np.tanh(5.0 * distance_xy)

        # === 2. HEIGHT ERROR ===
        height_error = np.abs(hand_pos[2] - reach_target[2])
        reward_height = 1.0 - np.tanh(10.0 * height_error)

        # === 3. ORIENTATION: Hand Z-axis should point DOWN [0, 0, -1] ===
        hand_z_axis = hand_mat[:, 2]
        desired_down = np.array([0.0, 0.0, -1.0])
        z_alignment = np.dot(hand_z_axis, desired_down)
        reward_z_down = (z_alignment + 1.0) / 2.0

        # === 4. CONTROL PENALTY ===
        reward_ctrl = -0.01 * np.square(self.data.ctrl[:7]).sum()

        # === TOTAL REWARD ===
        total_reward = (
                4.0 * reward_dist +
                4.0 * reward_height +
                1.0 * reward_z_down +
                reward_ctrl
        )

        # === DEBUG PRINTS ===
        if self.episode_length % 50 == 0:
            print(f"\n{'=' * 60}")
            print(f"Step: {self.episode_length}")
            print(f"{'=' * 60}")
            print(f"Hand pos:           [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
            print(f"Reach target:       [{reach_target[0]:.3f}, {reach_target[1]:.3f}, {reach_target[2]:.3f}]")
            print(f"-" * 60)
            print(f"Distance XY:      {distance_xy:.4f}")
            print(f"Height error:     {height_error:.4f}")
            print(f"Z-down alignment: {z_alignment:.4f}")
            print(f"-" * 60)
            print(f"R_dist:   {4.0 * reward_dist:.3f}")
            print(f"R_height: {4.0 * reward_height:.3f}")
            print(f"R_z_down: {1.0 * reward_z_down:.3f}")
            print(f"R_ctrl:   {reward_ctrl:.3f}")
            print(f"TOTAL:    {total_reward:.3f}")

        info = {"is_success": False}
        if distance_xy < 0.05 and height_error < 0.05 and z_alignment > 0.95:
            total_reward += 10.0
            info["is_success"] = True
            print(f"\n*** SUCCESS! ***")

        return total_reward, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

        self.current_ctrl = self.data.qpos[7:14].copy()

        # Settle physics
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # Compute reach target from initial bottle position
        self.initial_bottle_pos, self.reach_target = self._compute_reach_target()

        print(f"DEBUG: Bottle pos = {self.initial_bottle_pos}")
        print(f"DEBUG: Reach target = {self.reach_target}")

        self.episode_length = 0
        return self._get_obs(), {}

    def step(self, action):
        step_size = 0.005
        self.current_ctrl = self.current_ctrl + (action * step_size)
        self.current_ctrl = np.clip(self.current_ctrl, self.act_low, self.act_high)

        self.data.ctrl[:7] = self.current_ctrl

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, info = self._get_reward()

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