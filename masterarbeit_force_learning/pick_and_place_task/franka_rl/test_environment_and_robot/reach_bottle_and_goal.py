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
        self.target_site_id = self.model.site("goal").id
        self.bottle_joint_adr = self.model.jnt("bottle_joint").qposadr[0]

        # self.robot_base_xy = np.array([-0.4, 0.0])
        # self.safe_radius = 0.8 # 1.0 previous data, but it iw wrong. Real max 0.855

        actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = actuator_ranges[:, 0]
        self.act_high = actuator_ranges[:, 1]

        # This specifies that the action space is a box in continuous space(a multi - dimensional rectangle / cube).Actions
        # are real numbers, not discrete choices(like "left" or "right").
        self.action_space = spaces.Box(
            # low = actuator_ranges[:7, 0],
            low = -1.0,
            # high = actuator_ranges[:7, 1],
            high = 1.0,
            shape = (7,),
            dtype=np.float32
        )

        # Observation size:
        # qpos[7:14] (7 arm joints) + qvel[6:13] (7 arm velocities) + Gripper Pos (3) + Bottle Pos (3) + Goal Pos (3) = 23
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (23,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0

    def _get_obs(self):
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        goal_pos = self.data.site_xpos[self.target_site_id]

        return np.concatenate([
            self.data.qpos[7:14],
            self.data.qvel[6:13],
            gripper_pos,
            bottle_pos,
            goal_pos
        ]).astype(np.float32)

    def _get_reward(self):
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]
        goal_pos = self.data.site_xpos[self.target_site_id]

        bottle_base_target_pos = bottle_pos.copy()
        bottle_base_target_pos[2] = 0.84

        dist_gripper_to_bottle_base = np.linalg.norm(gripper_pos - bottle_base_target_pos)
        # reward_gripper = -0.1 * dist_gripper_to_bottle

        dist_bottle_to_goal = np.linalg.norm(bottle_pos - goal_pos)

        # --- 3. Define the Reward Components ---
        # We use the "sharpened" values from before

        # This is the "maintain contact" reward.
        # It's always active.
        reward_reach = -dist_gripper_to_bottle_base * 2.0

        # This is the "jackpot" for pushing.
        # It's only active when we're close.
        reward_push_jackpot = -dist_bottle_to_goal + 2.0

        # Penalties for smooth motion
        reward_velocity = -0.01 * np.linalg.norm(self.data.qvel[6:13])
        reward_action = -0.01 * np.linalg.norm(self.data.ctrl[:7])

        # --- 4. NEW Additive Reward Logic (Based on YOUR idea) ---

        # The agent is ALWAYS rewarded for being close to the bottle's base.
        reward = reward_reach

        # Set the "contact" threshold
        contact_threshold = 0.02  # 8 cm

        # IF the agent is "in contact" (close enough),
        # it ALSO gets the reward for pushing to the goal.
        if dist_gripper_to_bottle_base < contact_threshold:
            reward += reward_push_jackpot

        # ALWAYS apply the penalties
        reward += (reward_velocity + reward_action)

        return reward

    # def _get_random_safe_pos(self):
    #     while True:
    #         x = self.np_random.uniform(low=0.1, high=0.6)
    #         y = self.np_random.uniform(low=-0.4, high=0.4)
    #         pos_xy = np.array([x, y])
    #
    #         dist_from_base = np.linalg.norm(pos_xy -self.robot_base_xy)
    #
    #         if dist_from_base < self.safe_radius:
    #             return pos_xy

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mujoco.mj_forward(self.model, self.data)

        # goal_xy = np.array([0.5, 0.0])

        # while True:
        #     # goal_xy = self._get_random_safe_pos()
        #     bottle_xy = self._get_random_safe_pos()
        #     if np.linalg.norm(goal_xy - bottle_xy) > 0.15:
        #         break
        #
        # # random_x = self.np_random.uniform(low=0.4, high=0.6)
        # # random_y = self.np_random.uniform(low=-0.1, high=0.1)
        #
        # self.data.qpos[self.bottle_joint_adr : self.bottle_joint_adr + 3] = [bottle_xy[0], bottle_xy[1], 0.88]
        #
        # self.model.site_pos[self.target_site_id] = [goal_xy[0], goal_xy[1], 0.82]

        # for _ in range(100):
        #     mujoco.mj_step(self.model, self.data) # update the positions, velocities, and other physical properties, ahead by one tick

        self.episode_length = 0

        observations = self._get_obs()
        info = {}  # Empty info dict

        return observations, info

    def step(self, action):
        max_step_size = 0.01
        delta_action = action * max_step_size

        # Get the robot's current joint positions
        current_qpos = self.data.qpos[7:14]

        # Calculate the new target position by adding the delta
        new_target_qpos = current_qpos + delta_action

        # Clip the new target to make sure it's within the robot's joint limits
        clipped_target_qpos = np.clip(new_target_qpos, self.act_low, self.act_high)

        # Set this new clipped target as the control command
        self.data.ctrl[:7] = clipped_target_qpos

        for _ in range(70):
            mujoco.mj_step(self.model, self.data)

        observations = self._get_obs()
        reward = self._get_reward()

        self.episode_length += 1
        # dist_gripper_to_bottle = np.linalg.norm(self.data.site_xpos[self.gripper_site_id] - self.data.xpos[self.bottle_body_id])
        # terminated = bool(dist_gripper_to_bottle < 0.05)

        bottle_pos = self.data.xpos[self.bottle_body_id]
        goal_pos = self.data.site_xpos[self.target_site_id]
        dist_bottle_to_goal = np.linalg.norm(bottle_pos - goal_pos)

        terminated = bool(dist_bottle_to_goal < 0.05)
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