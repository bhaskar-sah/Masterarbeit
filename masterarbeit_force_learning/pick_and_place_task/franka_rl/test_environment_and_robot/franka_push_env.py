import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os

class PandaPushEnv(gym.Env):
    """
    Custom Environment for Franka Panda Pushing Task. I need to also add velocity information, so that the robot
    will have better understanding of the dynamics.
    Goal: slowly approach bottle and then push it slowly to the target location.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # --- Load the Model ---
        # Get path to the main XML file
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "panda_robot.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Get the ID of the 'home' keyframe from the XML
        self.home_key_id = self.model.key("home").id

        # Set the simulation to this stable "home" pose FIRST
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)

        # Store initial qpos and qvel to reset to
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Get body and site IDs for faster access
        self.bottle_body_id = self.model.body("bottle").id   # gives a integer value to the string "bottle" so that we can have faster access
        self.gripper_site_id = self.model.site("gripper_site").id
        self.goal_site_id = self.model.site("goal").id

        # --- Define Action and Observation Spaces ---
        # Action space: 7-DOF for the arm. We will lock the gripper.
        # We find the actuator control ranges from the XML
        actuator_ranges = self.model.actuator_ctrlrange

        # scale down the action space for smoother movements and thereby reduce the maximum torque
        self.action_scale = 0.3
        # Only take the first 7 actuators (the arm)
        self.action_space = spaces.Box(
            low=actuator_ranges[:7, 0] * self.action_scale,
            high=actuator_ranges[:7, 1] * self.action_scale,
            dtype=np.float32
        )

        # Observation space: Gripper pos (3), Bottle pos (3), Target pos (3)
        # but this is not enough. The velocities are also necessary. Altogether 12 velocitites
        # thereby making altogether 21-D
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,), # add also 6 more dimensions for relative distances
            dtype=np.float32
        )

        # --- State and get the goal site ID --
        # Get the target position directly from the model
        self.target_pos = self.data.site(self.goal_site_id).xpos.copy()

        # episode trackingS
        self.episode_length = 0
        self.max_episode_steps = 500

        # track previous distance for reward shaping
        self.prev_dist_to_target = None
        self.prev_dist_to_bottle = None

        # --- Visualization, important for mujoco ---
        self.render_mode = render_mode
        self.viewer = None

    def _get_obs(self):
        """
        Only the positions as previous will not work, for that reason also add velocities.
        This helps the agent to understand the dynamics!
        """
        # Get gripper positions (9D)
        gripper_pos = self.data.site(self.gripper_site_id).xpos.copy()
        bottle_pos = self.data.body(self.bottle_body_id).xpos.copy()
        target_pos = self.target_pos.copy()

        gripper_to_bottle = bottle_pos - gripper_pos  # relative position from gripper to bottle, helps agent learn better
        gripper_to_target = target_pos - gripper_pos  # relative position from gripper to target, helps agent learn better

        #Define velocities (13-D total)
        # get the gripper velocity from the parent body
        # find the body that the gripper site is attached to
        gripper_body_id = self.model.site_bodyid[self.gripper_site_id]
        gripper_vel = self.data.cvel[gripper_body_id][:3].copy()

        # bottle_vel = self.data.body(self.bottle_body_id).cvel[:3].copy()
        bottle_vel = self.data.cvel[self.bottle_body_id][:3].copy()

        joint_vel = self.data.qvel[:7].copy()

        #clip the velocities to prevent extreme values
        gripper_vel = np.clip(gripper_vel, -5.0, 5.0)
        bottle_vel = np.clip(bottle_vel, -5.0, 5.0)
        joint_vel = np.clip(joint_vel, -5.0, 5.0)

        # normalize to a same range
        gripper_vel_norm = gripper_vel / 5.0
        bottle_vel_norm = bottle_vel / 5.0
        joint_vel_norm = joint_vel / 5.0
        gripper_pos_norm = gripper_pos / np.array([1.0, 1.0, 1.5])
        gripper_to_bottle_norm = gripper_to_bottle / 2.0  # Max distance ~2m
        gripper_to_target_norm = gripper_to_target / 2.0

        # Concatenate 9 + 3 + 3 + 7 = 22D
        obs = np.concatenate([
            gripper_pos_norm,        #3
            # bottle_pos,       #3
            # target_pos,       #3
            gripper_to_bottle_norm,  # (3,) vector from gripper to bottle
            gripper_to_target_norm,  # (3,) vector from gripper to target
            gripper_vel_norm,        #3
            bottle_vel_norm,         #3
            joint_vel_norm           #7
        ])

        return obs.astype(np.float32)

    def _get_reward(self):
        """
        Multi-stage reward function with BALANCED penalties.
        """

        # Get positions
        gripper_pos = self.data.site(self.gripper_site_id).xpos
        bottle_pos = self.data.body(self.bottle_body_id).xpos

        # Calculate distances
        dist_gripper_to_bottle = np.linalg.norm(gripper_pos - bottle_pos)
        dist_bottle_to_target = np.linalg.norm(bottle_pos - self.target_pos)

        # get velocities for smoothness penalties
        joint_vel = self.data.qvel[:7]
        gripper_body_id = self.model.site_bodyid[self.gripper_site_id]
        gripper_vel = self.data.cvel[gripper_body_id][:3]
        bottle_vel = self.data.cvel[self.bottle_body_id][:3]

        # initialize reward
        reward = 0.0

        # ==========================================
        # 1. APPROACH PHASE
        # ==========================================
        # Make the "progress" reward much stronger
        if self.prev_dist_to_bottle is not None:
            approach_progress = self.prev_dist_to_bottle - dist_gripper_to_bottle
            reward += 10.0 * approach_progress  # <-- INCREASED from 2.0

        # Make the "stay near" penalty smaller
        reward += -0.1 * dist_gripper_to_bottle  # <-- DECREASED from -0.5

        # ==========================================
        # 2. CONTACT BONUS
        # ==========================================
        if dist_gripper_to_bottle < 0.05:
            reward += 5.0

            # ==========================================
            # 3. PUSHING PHASE
            # ==========================================
            if self.prev_dist_to_target is not None:
                push_progress = self.prev_dist_to_target - dist_bottle_to_target
                reward += 20.0 * push_progress  # <-- INCREASED from 10.0

            reward += -1.0 * dist_bottle_to_target  # <-- DECREASED from -2.0

        else:
            reward += -1.0 * dist_bottle_to_target

        # ==========================================
        # 4. SMOOTHNESS PENALTIES (THE MAIN FIX)
        # ==========================================
        # These penalties were way too high. We make them much smaller.

        # Penalize fast joint velocities (jerky motion)
        velocity_penalty = np.sum(np.square(joint_vel))
        reward += -0.01 * velocity_penalty  # <-- DECREASED from -0.3

        # penalize faster gripper movement
        gripper_speed = np.linalg.norm(gripper_vel)
        if gripper_speed > 0.5:
            reward += -0.05 * (gripper_speed - 0.5)  # <-- DECREASED from -0.5

        bottle_speed = np.linalg.norm(bottle_vel)
        if bottle_speed > 0.3:
            reward += -0.1 * (bottle_speed - 0.3)  # <-- DECREASED from -1.0

        # ==========================================
        # 5. CONTROL EFFORT
        # ==========================================
        control_penalty = np.sum(np.square(self.data.ctrl[:7]))
        reward += -0.001 * control_penalty  # <-- DECREASED from -0.05

        # ==========================================
        # 6. SUCCESS BONUS
        # ==========================================
        if dist_bottle_to_target < 0.05:
            reward += 200.0

            if bottle_speed < 0.1:
                reward += 50.0

        # ==========================================
        # 7. FAILURE PENALTIES
        # ==========================================
        if bottle_pos[2] < 0.85:
            reward += -50.0

        # if gripper_pos[2] > 1.2:
        #   reward += -10.0

        self.prev_dist_to_bottle = dist_gripper_to_bottle
        self.prev_dist_to_target = dist_bottle_to_target

        return reward

    def reset(self, seed=None, options=None):
        """Reset environment with curriculum learning support"""
        super().reset(seed=seed)

        # Reset simulation
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mujoco.mj_forward(self.model, self.data)

        # Randomize bottle position (curriculum: start easier)
        bottle_qpos_adr = self.model.jnt("bottle_joint").qposadr[0]

        # Start with bottle closer to target, gradually increase difficulty
        # You can adjust this based on training progress
        random_x = self.np_random.uniform(low=0.45, high=0.55)
        random_y = self.np_random.uniform(low=-0.05, high=0.05)

        self.data.qpos[bottle_qpos_adr: bottle_qpos_adr + 3] = [random_x, random_y, 0.88]

        # Reset tracking variables
        self.episode_length = 0
        self.prev_dist_to_target = None
        self.prev_dist_to_bottle = None

        # Forward simulation to update state
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):
        """Execute action with improved control"""

        # Apply action with scaling for smoother control
        self.data.ctrl[:7] = action
        self.data.ctrl[7] = 255  # Lock gripper closed

        # Run physics simulation
        # REDUCED from 75 to 50 for more responsive control
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Get results
        observation = self._get_obs()
        reward = self._get_reward()

        # Check termination conditions
        bottle_pos = self.data.body(self.bottle_body_id).xpos
        dist_bottle_to_target = np.linalg.norm(bottle_pos - self.target_pos)

        # Success condition
        terminated = (dist_bottle_to_target < 0.05)

        # Failure conditions
        bottle_fell = (bottle_pos[2] < 0.85)  # Fell off table
        if bottle_fell:
            terminated = True
            reward = -100  # Big penalty

        # Time limit
        self.episode_length += 1
        truncated = (self.episode_length >= self.max_episode_steps)

        info = {
            'dist_to_target': dist_bottle_to_target,
            'dist_to_bottle': np.linalg.norm(
                self.data.site(self.gripper_site_id).xpos - bottle_pos
            ),
            'success': terminated and not bottle_fell
        }

        return observation, reward, terminated, truncated, info

    # -----------------------------------------------
    # --- NEW RENDER METHOD ---
    # -----------------------------------------------
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
