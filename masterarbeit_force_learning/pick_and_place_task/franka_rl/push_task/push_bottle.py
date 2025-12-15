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
        xml_path = os.path.join(current_dir, "..", "test_environment_and_robot", "panda_robot.xml")

        # Optional: Convert to an absolute path to avoid potential issues with relative paths later
        xml_path = os.path.abspath(xml_path)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find XML file at: {xml_path}")

        print(f"Loading XML from: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.home_key_id = self.model.key("home").id

        # Body IDs
        self.bottle_body_id = self.model.body("bottle").id
        self.hand_body_id = self.model.body("hand").id
        self.goal_site_id = self.model.site("goal").id

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
            shape=(24,),  # 7 qpos + 7 qvel + 3 target + 4 hand Quat (w, x, y, z)
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0

        # Store initial bottle position and computed targets (set in reset)
        self.initial_bottle_pos = None
        self.reach_target = None
        self.push_dir = None

        # Store home hand quaternion for orientation matching
        self.home_hand_quat = None

        # Store initial distance for progress tracking
        self.initial_distance = None
        self.is_pushing = False

    def _get_obs(self):
        """Simple observation: robot state + target position"""
        hand_pos = self.data.xpos[self.hand_body_id]
        relative_vec = self.reach_target - hand_pos
        hand_quat = self.data.xquat[self.hand_body_id]

        return np.concatenate([
            self.data.qpos[7:14],  # Robot joints (7)
            self.data.qvel[6:13],  # Robot velocities (7)
            hand_pos,
            relative_vec,  # Target position (3)
            hand_quat  # Hand orientation (4)
        ]).astype(np.float32)

    def _get_target_pos(self):
        bottle_pos = self.data.xpos[self.bottle_body_id]
        # target_goal_pos = np.array([0.4, -0.2, 0.80])
        goal_pos = self.data.site_xpos[self.goal_site_id]

        # print(f"goal_pos: {goal_pos:.3f}")

        vec = goal_pos - bottle_pos
        # print(f"vector from bottle to goal: {vec:.3f}")
        vec[2] = 0  # ignore z
        dist = np.linalg.norm(vec)

        if dist < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = vec / dist

        # height of table 0.8m (80cm) + bottle 0.16m (16cm) = 0.96m (96cm)
        # height of hand 0.107m (10.7cm) + Finger 0.0584 (5.84cm) = 0.1654cm (16.54m)
        reach_target = bottle_pos - (direction * 0.18)
        reach_target[2] = 0.9654  # table (0.8m) + (hand+gripper) (0.1654m) = 0.9654m

        return reach_target, bottle_pos.copy(), direction, goal_pos, vec

    def _get_hand_x_axis(self):
        w, x, y, z = self.data.xquat[self.hand_body_id]
        # Formula for the first column (X-axis) of the rotation matrix
        hand_x_axis_x = 1 - 2 * (y * y + z * z)
        hand_x_axis_y = 2 * (x * y + w * z)
        hand_x_axis_z = 2 * (x * z - w * y)

        hand_x_axis = np.array([hand_x_axis_x, hand_x_axis_y, hand_x_axis_z])

        return hand_x_axis

    def _get_reward(self):
        hand_pos = self.data.xpos[self.hand_body_id]
        hand_quat = self.data.xquat[self.hand_body_id]
        _push_direction_vector = self._get_target_pos()

        _, _, _, goal_position, vector = self._get_target_pos()

        # Use pre-computed values (from initial bottle position)
        reach_target = self.reach_target

        # === 1. DISTANCE TO TARGET (3D) ===
        distance_3d = np.linalg.norm(hand_pos - reach_target)
        distance_xy = np.linalg.norm(hand_pos[:2] - reach_target[:2])
        height_error = np.abs(hand_pos[2] - reach_target[2])

        # === 2. PROGRESS REWARD (most important!) ===
        # Reward for getting CLOSER to target - this is the main driver
        # Negative distance means robot gets penalized for being far away
        ##########################################################
        # reward_distance = -distance_3d  # Simple negative distance

        # === 2. PROGRESS REWARD (most important!) ===
        # Exponential/Gaussian penalty for distance to target
        # Highly penalizes far distances, but flattens near zero,
        # forcing agent to learn precision.
        # Use np.exp(-(c * distance_3d**2)) - 1.0 (or just np.exp(-c * distance_3d))
        # This will be NEGATIVE and go to 0 as distance goes to 0.
        EXP_COEFF = 50.0  # Adjust this coefficient: higher C means steeper penalty closer to target
        # reward_distance = np.exp(-EXP_COEFF * distance_3d)
        reward_distance = np.exp(-EXP_COEFF * distance_3d)
        ##########################################################
        reward_pull = -2.0 * distance_3d
        ######################################
        reward_height = -5.0 * height_error

        # === 3. ORIENTATION: Match home quaternion ===
        home_quat = self.home_hand_quat
        quat_diff1 = np.linalg.norm(hand_quat - home_quat)
        quat_diff2 = np.linalg.norm(hand_quat + home_quat)
        quat_error = min(quat_diff1, quat_diff2)

        # Orientation bonus - but ONLY matters when close to target
        # This prevents robot from prioritizing orientation over movement
        proximity_gate = np.exp(-3.0 * distance_3d)  # Only kicks in when close
        reward_orientation = proximity_gate * (1.0 - np.tanh(5.0 * quat_error))

        hand_x_axis = self._get_hand_x_axis()
        # calculate alignment (Dot Product)
        # 1.0 = x-axis is parallel to push direction
        alignment_score = np.dot(hand_x_axis, self.push_dir)
        reward_alignment = max(0, -alignment_score)

        # === 4. CONTROL PENALTY (small) ===
        reward_ctrl = -0.0001 * np.square(self.data.ctrl[:7]).sum()

        # === 5. BONUS FOR REACHING TARGET ===
        reached_position = distance_3d < 0.02  # make it extremely precise to 2 cm rather than 0.08m
        # reached_with_orientation = distance_3d < 0.05 and quat_error < 0.1
        #reached_with_orientation = distance_3d < 0.02 and quat_error < 0.05
        # == 6. BONUS for aligning perfectly
        aligned_perfectly = alignment_score < - 0.9

        reached_and_aligned = reached_position and aligned_perfectly

        bonus = 0.0
        if reached_position:
            bonus += 1.5  # Small bonus for getting close
        # else:
        #     bonus += 0.0  # nothing for "close enough"

        if reached_and_aligned:
            bonus += 2.0  # Big bonus for reaching with correct orientation

        # === TOTAL REWARD ===
        total_reward = (
                9.0 * reward_distance +  # Primary: get closer (negative when far) -> now increase pull strength from 5.0 to 8.0
                3.0 * reward_pull +
                1.0 * reward_height +
                2.0 * reward_orientation +  # Secondary: orientation (gated by proximity)
                3.5 * reward_alignment +
                reward_ctrl +
                bonus
        )

        # === DEBUG PRINTS ===
        if self.episode_length % 50 == 0:
            print(f"\n{'=' * 60}")
            print(f"Step: {self.episode_length}")
            print(f"{'=' * 60}")
            print(f"Hand pos:           [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
            print(f"Reach target:       [{reach_target[0]:.3f}, {reach_target[1]:.3f}, {reach_target[2]:.3f}]")
            print(f"goal_position:      [{goal_position[0]:.3f}, {goal_position[1]:.3f}, {goal_position[2]:.3f}]")
            print(f"vector:             [{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}]")
            print(f"-" * 60)
            print(f"Distance 3D:      {distance_3d:.4f}")
            print(f"Distance XY:      {distance_xy:.4f}")
            print(f"Height error:     {height_error:.4f}")
            print(f"Quat error:       {quat_error:.4f}")
            print(f"Proximity gate:   {proximity_gate:.4f}")
            print(f"R_precision:    {9.0 * reward_distance:.3f}")
            print(f"R_pull:         {1.0 * reward_pull:.3f}")  # Now you can see exactly which one is working!
            print(f"-" * 60)
            print(f"Push Dir:          [{self.push_dir[0]:.3f}, {self.push_dir[1]:.3f}, {self.push_dir[2]:.3f}]")
            print(f"Hand X-Axis:       [{hand_x_axis[0]:.3f}, {hand_x_axis[1]:.3f}, {hand_x_axis[2]:.3f}]")
            print(f"X-ALIGNMENT SCORE: {alignment_score:.4f}")
            print(f"-" * 60)
            print(f"R_distance:    {9.0 * reward_distance:.3f}")
            print(f"R_alignment:    {4.0 * reward_alignment:.3f}")
            print(f"R_ctrl:        {reward_ctrl:.4f}")
            print(f"Bonus:         {bonus:.3f}")
            print(f"TOTAL:         {total_reward:.3f}")
            print(f"Hand quat:     [{hand_quat[0]:.3f}, {hand_quat[1]:.3f}, {hand_quat[2]:.3f}, {hand_quat[3]:.3f}]")
            print(f"Home quat:     [{home_quat[0]:.3f}, {home_quat[1]:.3f}, {home_quat[2]:.3f}, {home_quat[3]:.3f}]")

        info = {"is_success": False}
        if reached_and_aligned:
            total_reward += 10.0
            info["is_success"] = True
            print(f"\n*** SUCCESS! ***")

        return total_reward, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ##################
        self.is_pushing = False
        ##################
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

        self.current_ctrl = self.data.qpos[7:14].copy()

        # Settle physics
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # Update kinematics after settling
        mujoco.mj_forward(self.model, self.data)

        # CAPTURE HOME HAND QUATERNION
        self.home_hand_quat = self.data.xquat[self.hand_body_id].copy()

        # Compute reach target from initial bottle position
        self.reach_target, self.initial_bottle_pos, self.push_dir,_,_ = self._get_target_pos()

        # Store initial distance for progress tracking
        hand_pos = self.data.xpos[self.hand_body_id]
        self.initial_distance = np.linalg.norm(hand_pos - self.reach_target)

        self.episode_length = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Small step size for smooth motion
        step_size = 0.005
        self.current_ctrl = self.current_ctrl + (action * step_size)
        # Clip to hardware limits
        self.current_ctrl = np.clip(self.current_ctrl, self.act_low, self.act_high)

        # Send to MuJoCo
        self.data.ctrl[:7] = self.current_ctrl

        # Step physics multiple times for stability
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, info = self._get_reward()

        self.episode_length += 1

        # Termination conditions
        terminated = False
        if info["is_success"]:
            terminated = True

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