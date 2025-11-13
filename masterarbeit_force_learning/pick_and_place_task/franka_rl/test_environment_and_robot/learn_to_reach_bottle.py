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

        self.bottle_body_id = self.model.body("bottle").id
        self.gripper_site_id = self.model.site("gripper_site").id
        self.bottle_joint_adr = self.model.jnt("bottle_joint").qposadr[0]

        actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = actuator_ranges[:, 0]
        self.act_high = actuator_ranges[:, 1]

        self.action_space = spaces.Box(
            # low = actuator_ranges[:7, 0],
            low = -1.0,
            # high = actuator_ranges[:7, 1],
            high = 1.0,
            shape = (7,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (20,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0

    def _get_obs(self):
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]

        return np.concatenate([
            self.data.qpos[7:14],
            self.data.qvel[6:13],
            gripper_pos,
            bottle_pos
        ]).astype(np.float32)

    def _get_reward(self):
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]

        dist_gripper_to_bottle = np.linalg.norm(gripper_pos - bottle_pos)
        reward_gripper = -0.1 * dist_gripper_to_bottle

        reward_velocity = -0.01 * np.linalg.norm(self.data.qvel[6:13])
        reward_action = -0.001 * np.linalg.norm(self.data.ctrl[:7])

        reward = reward_gripper + reward_velocity + reward_action

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mujoco.mj_forward(self.model, self.data)

        random_x = self.np_random.uniform(low=0.4, high=0.6)
        random_y = self.np_random.uniform(low=-0.1, high=0.1)

        self.data.qpos[self.bottle_joint_adr : self.bottle_joint_adr + 3] = [random_x, random_y, 0.88]

        self.episode_length = 0

        observations = self._get_obs()
        info = {}  # Empty info dict

        return observations, info

    def step(self, action):
        scaled_action = self.act_low + (action + 1.0) * 0.5 * (self.act_high - self.act_low)
        self.data.ctrl[:7] = scaled_action

        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        observations = self._get_obs()
        reward = self._get_reward()

        self.episode_length += 1
        dist_gripper_to_bottle = np.linalg.norm(
            self.data.site_xpos[self.gripper_site_id] - self.data.xpos[self.bottle_body_id]
        )
        terminated = bool(dist_gripper_to_bottle < 0.05)
        truncated = bool(self.episode_length >= 500)  # Max 500 steps

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