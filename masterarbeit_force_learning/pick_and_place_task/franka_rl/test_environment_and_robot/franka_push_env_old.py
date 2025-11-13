import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os


class PandaPushEnv(gym.Env):
    """
    Custom Environment for Franka Panda Pushing Task.
    Goal: Push a bottle to a target location.
    """

    # Metadata for gym
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # --- 1. Load your Model ---
        # Get path to your main XML file
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "panda_robot.xml")  # Your scene file

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.home_key_id = self.model.key("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)

        # Store initial qpos and qvel to reset to
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Get body and site IDs for faster access
        self.bottle_body_id = self.model.body("bottle").id
        self.gripper_site_id = self.model.site("gripper_site").id  # IMPORTANT: See note below

        # --- 2. Define Action and Observation Spaces ---

        # Action space: 7-DOF for the arm. We will lock the gripper.
        # We find the actuator control ranges from the XML
        actuator_ranges = self.model.actuator_ctrlrange
        # We only take the first 7 actuators (the arm)
        self.action_space = spaces.Box(low=actuator_ranges[:7, 0],
                                       high=actuator_ranges[:7, 1],
                                       dtype=np.float32)

        # Observation space: Gripper pos (3), Bottle pos (3), Target pos (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(9,), dtype=np.float32)

        # --- 3. Define Goal and State ---
        self.target_pos = np.array([0.5, 0.0, 0.88])  # Fixed target on the table
        self.episode_length = 0

        # --- 4. Visualization ---
        self.render_mode = render_mode
        self.viewer = None

    def _get_obs(self):
        """Helper function to get the current observation."""
        # Get gripper position
        gripper_pos = self.data.site(self.gripper_site_id).xpos

        # Get bottle position
        bottle_pos = self.data.body(self.bottle_body_id).xpos

        # Concatenate into a single array
        return np.concatenate([gripper_pos, bottle_pos, self.target_pos])

    def _get_reward(self):
        """Helper function to calculate the reward for PUSHING."""

        # Get positions
        gripper_pos = self.data.site(self.gripper_site_id).xpos
        bottle_pos = self.data.body(self.bottle_body_id).xpos

        # Calculate distances
        # 1. Distance from bottle to the target (THIS IS THE MAIN GOAL)
        dist_bottle_to_target = np.linalg.norm(bottle_pos - self.target_pos)

        # 2. Distance from gripper to the bottle (for reward shaping)
        dist_gripper_to_bottle = np.linalg.norm(gripper_pos - bottle_pos)

        # --- Reward Shaping ---
        reward_bottle = -dist_bottle_to_target
        reward_gripper = -0.1 * dist_gripper_to_bottle

        # --- Penalties (NEW) ---
        # 1. Penalize high joint velocities (encourages slow movement)
        reward_velocity = -0.01 * np.linalg.norm(self.data.qvel[:7])

        # 2. Penalize large control signals (encourages using less force)
        reward_action = -0.001 * np.linalg.norm(self.data.ctrl[:7])

        reward = reward_bottle + reward_gripper + reward_velocity + reward_action

        # Bonus for success
        if dist_bottle_to_target < 0.05:  # 5cm tolerance
            reward += 100

        return reward

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        # 1. Reset the simulation to its initial state
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mujoco.mj_forward(self.model, self.data)

        # 2. Randomize the bottle's starting position on the table
        # Table is at pos=(0.5, 0, 0.4) and size=(0.6, 0.6, 0.4)
        # Bottle should be on top (z=0.88)
        # We find the free joint for the bottle
        bottle_qpos_adr = self.model.jnt("bottle_joint").qposadr[0]  # Assumes joint is named 'bottle_joint'

        # Random (x, y) on the table, plus bottle height (z=0.88)
        random_x = self.np_random.uniform(low=0.4, high=0.6)
        random_y = self.np_random.uniform(low=-0.1, high=0.1)

        self.data.qpos[bottle_qpos_adr: bottle_qpos_adr + 3] = [random_x, random_y, 0.88]

        # 3. Reset episode timer
        self.episode_length = 0

        # Get the initial observation
        observation = self._get_obs()
        info = {}  # Empty info dict

        return observation, info

    def step(self, action):
        """Applies an action and steps the simulation."""

        # 1. Apply the action to the controls
        # Action is 7-dim. We set the 8th control (gripper) to "closed".
        self.data.ctrl[:7] = action
        self.data.ctrl[7] = 255  # Lock the gripper in a closed-ish state

        # 2. Step the simulation
        for _ in range(10):  # 10 physics steps per environment step
            mujoco.mj_step(self.model, self.data)

        # 3. Get the results
        observation = self._get_obs()
        reward = self._get_reward()

        # 4. Check for 'terminated' (task success)
        dist_bottle_to_target = np.linalg.norm(self.data.body(self.bottle_body_id).xpos - self.target_pos)
        terminated = (dist_bottle_to_target < 0.05)  # Success

        # 5. Check for 'truncated' (ran out of time)
        self.episode_length += 1
        truncated = (self.episode_length >= 500)  # Max 500 steps

        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renders the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                # launch_passive is non-blocking
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            # Sync the viewer with the current simulation state
            self.viewer.sync()

    def close(self):
        """Closes the viewer."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None