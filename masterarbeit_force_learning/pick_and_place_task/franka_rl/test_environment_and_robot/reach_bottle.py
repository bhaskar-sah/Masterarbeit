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
        # mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)

        # self.init_qpos = self.data.qpos.copy()
        # self.init_qvel = self.data.qvel.copy()

        # Body IDs
        self.bottle_body_id = self.model.body("bottle").id
        self.hand_body_id = self.model.body("hand").id
        self.target_site_id = self.model.site("goal").id

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

        # Observation: robot joints + gripper position + target position
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),  # 7 qpos + 7 qvel + 3 target + 4 hand Quat (w, x, y, z)
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0

    def _get_obs(self):
        """Simple observation: robot state + target position"""
        # bottle_pos = self.data.xpos[self.bottle_body_id]
        # target_goal_pos = self.data.xpos[self.target_site_id]
        reach_target, bottle_pos, _ = self._get_target_pos()
        hand_pos = self.data.xpos[self.hand_body_id]

        # Calculate HORIZONTAL push direction
        # bottle_to_goal_vec = target_goal_pos - bottle_pos
        # bottle_to_goal_vec[2] = 0  # Zero out Z - horizontal only
        # bottle_to_goal_dist = np.linalg.norm(bottle_to_goal_vec)

        relative_vec = reach_target - hand_pos

        hand_quat = self.data.xquat[self.hand_body_id]

        return np.concatenate([
            self.data.qpos[7:14],  # Robot joints (7)
            self.data.qvel[6:13],  # Robot velocities (7)
            relative_vec,  # Target position (3)
            hand_quat
        ]).astype(np.float32)

    def _get_target_pos(self):
        bottle_pos = self.data.xpos[self.bottle_body_id]
        target_goal_pos = self.data.xpos[self.target_site_id]

        vec = target_goal_pos - bottle_pos
        vec[2] = 0 # ignore z
        dist = np.linalg.norm(vec)

        if dist < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = vec / dist

        reach_target = bottle_pos - (direction * 0.18) # instead of 0.12
        reach_target[2] = 0.99 # table (0.80) + 0.08 + 0.11
        return reach_target, bottle_pos, direction

    def _get_reward(self):
        reach_target, bottle_pos, push_dir = self._get_target_pos()
        hand_pos = self.data.xpos[self.hand_body_id]
        hand_mat = self.data.xmat[self.hand_body_id].reshape(3, 3)

        # === 1. DISTANCE TO TARGET (XY only) ===
        distance_xy = np.linalg.norm(hand_pos[:2] - reach_target[:2])
        reward_dist = 1.0 - np.tanh(5.0 * distance_xy)

        # === 2. HEIGHT: Stay at target Z level ===
        height_error = np.abs(hand_pos[2] - reach_target[2])
        reward_height = 1.0 - np.tanh(10.0 * height_error)

        # === 3. ORIENTATION A: Hand Z-axis should point DOWN [0, 0, -1] ===
        hand_z_axis = hand_mat[:, 2]
        desired_down = np.array([0.0, 0.0, -1.0])
        z_alignment = np.dot(hand_z_axis, desired_down)
        reward_z_down = (z_alignment + 1.0) / 2.0

        # === 4. ORIENTATION B: Try POSITIVE X axis instead ===
        hand_x = -hand_mat[:, 0]  # Try +X instead of -X

        # Project to horizontal
        hand_x_horiz = hand_x.copy()
        hand_x_horiz[2] = 0
        norm = np.linalg.norm(hand_x_horiz)
        if norm > 1e-6:
            hand_x_horiz /= norm
        else:
            hand_x_horiz = np.array([1.0, 0.0, 0.0])

        push_alignment = np.dot(hand_x_horiz, push_dir)
        reward_push_align = (push_alignment + 1.0) / 2.0

        # === 5. CONTROL PENALTY ===
        reward_ctrl = -0.01 * np.square(self.data.ctrl[:7]).sum()

        # === GATING ===
        proximity_gate = np.exp(-5.0 * distance_xy)

        # === TOTAL REWARD ===
        total_reward = (
                2.0 * reward_dist +
                1.5 * reward_height +
                1.0 * proximity_gate * reward_z_down +
                1.5 * proximity_gate * reward_push_align +  # Increased weight
                reward_ctrl
        )

        # === DEBUG PRINTS ===
        if self.episode_length % 50 == 0:
            # Also print all three axes to see which one we should use
            hand_y = hand_mat[:, 1]
            print(f"\n{'=' * 60}")
            print(f"Step: {self.episode_length}")
            print(f"{'=' * 60}")
            print(f"Hand pos:    [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
            print(f"Bottle pos:    [{bottle_pos[0]:.3f}, {bottle_pos[1]:.3f}, {bottle_pos[2]:.3f}]")
            print(f"Target pos:  [{reach_target[0]:.3f}, {reach_target[1]:.3f}, {reach_target[2]:.3f}]")
            print(f"-" * 60)
            print(f"Distance XY:      {distance_xy:.4f}")
            print(f"Height error:     {height_error:.4f}")
            print(f"Z-down alignment: {z_alignment:.4f}")
            print(f"Push alignment:   {push_alignment:.4f}")
            print(f"-" * 60)
            print(f"Push dir:    [{push_dir[0]:.3f}, {push_dir[1]:.3f}, {push_dir[2]:.3f}]")
            print(f"Hand +X:     [{hand_x[0]:.3f}, {hand_x[1]:.3f}, {hand_x[2]:.3f}]")
            print(f"Hand +Y:     [{hand_y[0]:.3f}, {hand_y[1]:.3f}, {hand_y[2]:.3f}]")
            print(f"Hand +Z:     [{hand_z_axis[0]:.3f}, {hand_z_axis[1]:.3f}, {hand_z_axis[2]:.3f}]")
            print(f"-" * 60)
            # Show dot products of all axes with push_dir
            dot_x = np.dot(hand_x[:2] / (np.linalg.norm(hand_x[:2]) + 1e-8), push_dir[:2])
            dot_y = np.dot(hand_y[:2] / (np.linalg.norm(hand_y[:2]) + 1e-8), push_dir[:2])
            dot_neg_x = np.dot(-hand_x[:2] / (np.linalg.norm(hand_x[:2]) + 1e-8), push_dir[:2])
            dot_neg_y = np.dot(-hand_y[:2] / (np.linalg.norm(hand_y[:2]) + 1e-8), push_dir[:2])
            print(f"Dot products with push_dir:")
            print(f"  +X: {dot_x:.3f}  |  -X: {dot_neg_x:.3f}")
            print(f"  +Y: {dot_y:.3f}  |  -Y: {dot_neg_y:.3f}")
            print(f"-" * 60)
            print(f"TOTAL:    {total_reward:.3f}")

        info = {"is_success": False}
        if distance_xy < 0.05 and height_error < 0.03 and z_alignment > 0.9 and push_alignment > 0.85:
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

        self.episode_length = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Small step size for smooth motion
        step_size = 0.005 # before 0.01
        self.current_ctrl = self.current_ctrl + (action * step_size)
        # Clip to hardware limits
        self.current_ctrl = np.clip(self.current_ctrl, self.act_low, self.act_high)

        # Send to MuJoCo
        self.data.ctrl[:7] = self.current_ctrl

        # Step physics multiple times for stability
        for _ in range(20):  # 20 is usually enough if timestep is 0.002
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, info = self._get_reward()

        self.episode_length += 1

        # Termination conditions
        terminated = False
        if info["is_success"]:
            terminated = True  # Stop if we reached the target!

        truncated = bool(self.episode_length >= 500)

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