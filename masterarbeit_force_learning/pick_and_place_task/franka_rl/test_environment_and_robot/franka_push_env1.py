import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os


class PandaPushEnv(gym.Env):
    """
    Custom Environment for Franka Panda Curriculum Learning.
    Switch between 'reach' (approach bottle) and 'push' (push bottle to goal).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, mode="reach"):
        super().__init__()
        # --- 1. Load your Model ---
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "panda_robot.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Store initial states
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Get body and site IDs
        self.bottle_body_id = self.model.body("bottle").id
        self.gripper_site_id = self.model.site("gripper_site").id
        self.goal_site_id = self.model.site("goal").id
        self.target_pos = self.data.site(self.goal_site_id).xpos.copy()

        self.episode_length = 0
        self.render_mode = render_mode
        self.viewer = None

        # ---- Curriculum Mode ----
        # 'reach' or 'push'
        assert mode in ("reach", "push"), "mode must be 'reach' or 'push'"
        self.mode = mode

        # Action space: 7 DOF for arm
        actuator_ranges = self.model.actuator_ctrlrange
        self.action_space = spaces.Box(
            low=actuator_ranges[:7, 0],
            high=actuator_ranges[:7, 1],
            dtype=np.float32
        )

        # Observation: Gripper (3), Bottle (3), Goal (3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

    def _get_obs(self):
        gripper_pos = self.data.site(self.gripper_site_id).xpos
        bottle_pos = self.data.body(self.bottle_body_id).xpos
        return np.concatenate([gripper_pos, bottle_pos, self.target_pos])

    def _get_reward(self):
        gripper_pos = self.data.site(self.gripper_site_id).xpos
        bottle_pos = self.data.body(self.bottle_body_id).xpos
        dist_gripper_to_bottle = np.linalg.norm(gripper_pos - bottle_pos)
        dist_bottle_to_target = np.linalg.norm(bottle_pos - self.target_pos)
        reward_velocity = -0.01 * np.linalg.norm(self.data.qvel[:7])

        if self.mode == "reach":
            # Curriculum phase 1: Reward getting close to the bottle
            reward = -dist_gripper_to_bottle + reward_velocity
            # Big bonus for success
            if dist_gripper_to_bottle < 0.03:
                reward += 100
        elif self.mode == "push":
            # Full task: Reward for pushing bottle to goal
            reward_bottle = -dist_bottle_to_target
            reward_gripper = -0.5 * dist_gripper_to_bottle
            reward = reward_bottle + reward_gripper + reward_velocity
            if dist_bottle_to_target < 0.05:
                reward += 100
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. Reset to start state
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mujoco.mj_forward(self.model, self.data)

        # 2. Randomize bottle position
        bottle_qpos_adr = self.model.jnt("bottle_joint").qposadr[0]
        random_x = self.np_random.uniform(low=0.4, high=0.6)
        random_y = self.np_random.uniform(low=-0.1, high=0.1)
        self.data.qpos[bottle_qpos_adr: bottle_qpos_adr + 3] = [random_x, random_y, 0.88]

        self.episode_length = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.data.ctrl[:7] = action
        self.data.ctrl[7] = 255  # Lock gripper

        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._get_reward()
        self.episode_length += 1

        bottle_pos = self.data.body(self.bottle_body_id).xpos
        gripper_pos = self.data.site(self.gripper_site_id).xpos
        dist_gripper_to_bottle = np.linalg.norm(gripper_pos - bottle_pos)
        dist_bottle_to_target = np.linalg.norm(bottle_pos - self.target_pos)

        if self.mode == "reach":
            terminated = dist_gripper_to_bottle < 0.03
        elif self.mode == "push":
            terminated = dist_bottle_to_target < 0.05

        truncated = (self.episode_length >= 500)
        info = {}
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
