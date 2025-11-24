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

        # Body IDs
        self.bottle_body_id = self.model.body("bottle").id
        self.hand_body_id = self.model.body("hand").id
        self.target_site_id = self.model.site("goal").id

        # Action space: 7 robot joints
        actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = actuator_ranges[:, 0]
        self.act_high = actuator_ranges[:, 1]

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # Observation: robot joints + gripper position + target position
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),  # 7 qpos + 7 qvel + 3 target
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0

    def _get_obs(self):
        """Simple observation: robot state + target position"""
        hand_pos = self.data.xpos[self.hand_body_id]
        hand_mat = self.data.xmat[self.hand_body_id].reshape(3, 3)
        hand_y_axis = hand_mat[:, 1]

        # Gripper position (10cm backward from hand)
        gripper_pos = hand_pos - 0.1 * hand_y_axis

        # Calculate target: 20cm behind bottle on X-axis
        bottle_pos = self.data.xpos[self.bottle_body_id]
        target_pos = np.array([
            bottle_pos[0] - 0.20,  # 20cm behind on X-axis
            bottle_pos[1],  # Same Y
            0.85  # Fixed height
        ])

        return np.concatenate([
            self.data.qpos[7:14],  # Robot joints (7)
            self.data.qvel[6:13],  # Robot velocities (7)
            target_pos  # Target position (3)
        ]).astype(np.float32)

    def _get_reward(self):
        """Simple distance-based reward"""
        # Get gripper position
        hand_pos = self.data.xpos[self.hand_body_id]
        hand_rot = self.data.xmat[self.hand_body_id].reshape(3,3)
        hand_x_axis = hand_rot[:, 0]
        # hand_mat = self.data.xmat[self.hand_body_id].reshape(3, 3)
        # hand_y_axis = hand_mat[:, 1]
        # gripper_pos = hand_pos - 0.1 * hand_y_axis

        # Calculate target position
        bottle_pos = self.data.xpos[self.bottle_body_id]
        """
        target_pos = np.array([
            bottle_pos[0] - 0.20,  # 20cm behind bottle
            bottle_pos[1],
            1.2
        ])
        """

        target_goal_pos = self.data.xpos[self.target_site_id]
        dist_to_goal_vec = target_goal_pos - bottle_pos
        dist_to_goal = np.linalg.norm(dist_to_goal_vec)

        target_pos = np.array([
            dist_to_goal_vec[0] - 0.10, # 10 cm behind the bottle
            dist_to_goal_vec[1],
            1.2
        ])

        if dist_to_goal < 1e-6:
            unit_push_direction = np.array([1.0, 0.0])
        else:
            unit_push_direciton = (target_goal_pos - bottle_pos)

        align_hand_x_axis_and_dist_to_goal_vec = np.dot(hand_pos)

        # Calculate distance
        distance = np.linalg.norm(gripper_pos - target_pos)

        # Simple exponential reward
        reward = 100.0 * np.exp(-5.0 * distance)

        # Bonus for getting close
        if distance < 0.10:
            reward += 200.0
        if distance < 0.05:
            reward += 300.0
        if distance < 0.02:
            reward += 500.0

        # Corrected joint velocity penalties (using right indices)
        joint_4_penalty = -10.0 * (self.data.qvel[9] ** 2)  # High penalty
        joint_5_penalty = -5.0 * (self.data.qvel[10] ** 2)
        joint_6_penalty = -5.0 * (self.data.qvel[11] ** 2)
        joint_7_penalty = -3.0 * (self.data.qvel[12] ** 2)

        # Small penalty for movement (encourage efficiency)
        velocity_penalty = -0.01 * np.linalg.norm(self.data.qvel[6:13])

        total_reward = (reward +
                        velocity_penalty +
                        joint_4_penalty +
                        joint_5_penalty +
                        joint_6_penalty +
                        joint_7_penalty
                        )

        # Debug output
        if self.episode_length % 100 == 0:
            print(f"\n=== Step {self.episode_length} ===")
            print(f"Gripper pos: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
            print(f"Target pos:  [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            print(f"Distance: {distance:.3f}m")
            print(f"Reward: {total_reward:.1f}")
            print(f"All joint velocities: {self.data.qvel[:7]}")

        return total_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset to home position
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel

        # Lower the arm
        # self.data.qpos[8] = 0.5  # joint2
        # self.data.qpos[10] = -2.0  # joint4
        # self.data.qpos[12] = 1.8  # joint6

        mujoco.mj_forward(self.model, self.data)

        # Settle physics
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.episode_length = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Small step size for smooth motion
        max_step_size = 0.005 # before 0.01
        delta_action = action * max_step_size

        # Apply action
        current_qpos = self.data.qpos[7:14]
        new_target_qpos = current_qpos + delta_action
        clipped_target_qpos = np.clip(new_target_qpos, self.act_low, self.act_high)

        self.data.ctrl[:7] = clipped_target_qpos

        # Step physics
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Get observation and reward
        obs = self._get_obs()
        reward = self._get_reward()

        self.episode_length += 1

        # Episode ends after 500 steps
        terminated = False
        truncated = bool(self.episode_length >= 500)

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