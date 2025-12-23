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
        xml_path = os.path.join(current_dir, "panda_robot_push.xml")

        # Optional: Convert to an absolute path to avoid potential issues with relative paths later
        xml_path = os.path.abspath(xml_path)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find XML file at: {xml_path}")

        print(f"Loading XML from: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Body IDs
        self.home_key_id = self.model.key("home").id
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

        # Observation space: robot joints + gripper position + target position
        # [qpos(7), qvel(7), hand_pos(3), bottle_pos(3), bottle_to_goal(3), hand_to_bottle(3)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(26,),  # 7 qpos + 7 qvel + 3 target + 4 hand Quat (w, x, y, z)
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0
        self.goal_pos = None
        self.contact_made = False

        # Get finger body IDs for contact detection
        self.left_finger_body_id = self.model.body("left_finger").id
        self.right_finger_body_id = self.model.body("right_finger").id

        # All bodies that count as "robot touching"
        self.robot_contact_bodies = {
            self.hand_body_id,
            self.left_finger_body_id,
            self.right_finger_body_id
        }

        print(
            f"Robot contact bodies: hand={self.hand_body_id}, left_finger={self.left_finger_body_id}, right_finger={self.right_finger_body_id}")

    def _get_obs(self):
        """Simple observation: robot state + target position"""
        # robot state
        qpos = self.data.qpos[7:14]
        qvel = self.data.qvel[6:13]

        # object positions
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]

        # relativ vectors
        # vector from bottle to goal (where should the bottle go?)
        bottle_to_goal = self.goal_pos - bottle_pos

        # vector from hand to bottle (where is the hand relative to object?)
        hand_to_bottle = bottle_pos - hand_pos

        return np.concatenate([
            qpos,  # Robot joints (7)
            qvel,  # Robot velocities (7)
            hand_pos,
            bottle_pos,
            bottle_to_goal,
            hand_to_bottle,  # Target position (3)
        ]).astype(np.float32)

    def _compute_push_direction(self):
        # Normalized vector from Bottle -> Goal (XY plane only).
        bottle_pos = self.data.xpos[self.bottle_body_id]

        vec = self.goal_pos - bottle_pos
        vec[2] = 0 # ignore Z
        dist = np.linalg.norm(vec)

        if dist < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = vec / dist

        return direction, dist

    # def _is_touching(self):
    #     # returns true if the mujoco physics engine detects a collision between the hand and the bottle
    #     for i in range(self.data.ncon):
    #         contact = self.data.contact[i]
    #
    #         geom1 = contact.geom1
    #         geom2 = contact.geom2
    #
    #         body1 = self.model.geom_bodyid[geom1]
    #         body2 = self.model.geom_bodyid[geom2]
    #
    #         match_1 = (body1 == self.hand_body_id and body2 == self.bottle_body_id)
    #         match_2 = (body1 == self.bottle_body_id and body2 == self.hand_body_id)
    #
    #         if match_1 or match_2:
    #             return True
    #
    #     return False

    def _is_touching(self):
        # Method 1: Physics contact detection (fingers)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            robot_touch = body1 in self.robot_contact_bodies or body2 in self.robot_contact_bodies
            bottle_touch = body1 == self.bottle_body_id or body2 == self.bottle_body_id

            if robot_touch and bottle_touch:
                return True

        # Method 2: Distance-based backup
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]

        # XY distance from hand to bottle center
        dist_xy = np.linalg.norm(hand_pos[:2] - bottle_pos[:2])

        # Height of hand above bottle base
        hand_height_above_bottle = hand_pos[2] - bottle_pos[2]

        # Contact if:
        # - XY distance < bottle_radius + finger_reach (about 7cm)
        # - Hand is at the right height (10-16cm above bottle base)
        bottle_radius = 0.03
        if dist_xy < (bottle_radius + 0.05) and 0.10 < hand_height_above_bottle < 0.16:
            return True

        return False

    def _get_reward(self):
        info = {"is_success": False}

        bottle_pos = self.data.xpos[self.bottle_body_id]
        hand_pos = self.data.xpos[self.hand_body_id]

        # ============ CONTACT CHECK ============
        is_touching = self._is_touching()
        if is_touching:
            self.contact_made = True

        # ============ GOAL DIRECTION ============
        # Direction from bottle to goal (normalized)
        bottle_to_goal = self.goal_pos[:2] - bottle_pos[:2]
        dist_bottle_goal = np.linalg.norm(bottle_to_goal)
        if dist_bottle_goal > 0.001:
            goal_dir = bottle_to_goal / dist_bottle_goal
        else:
            goal_dir = np.array([0, -1])  # Default push direction

        # ============ VELOCITIES ============
        hand_vel = self.data.cvel[self.hand_body_id][3:6]
        bottle_vel = self.data.cvel[self.bottle_body_id][3:6]

        # ============ REWARD CALCULATION ============

        if is_touching:
            # === CONTACT REWARD (Paper's key insight) ===

            # 1. θ: Angle between push direction and goal direction
            hand_vel_xy = hand_vel[:2]
            hand_speed_xy = np.linalg.norm(hand_vel_xy)
            if hand_speed_xy > 0.01:
                push_dir = hand_vel_xy / hand_speed_xy
                # cos(θ) - want this to be 1.0 (aligned)
                alignment = np.dot(push_dir, goal_dir)
                theta_penalty = -2.0 * (1.0 - alignment)  # 0 when aligned, -4 when opposite
            else:
                theta_penalty = 0.0

            # 2. d3: Lever arm (distance from hand to bottle center line)
            # In 2D: perpendicular distance from hand to the line through bottle CoM in goal direction
            hand_to_bottle = hand_pos[:2] - bottle_pos[:2]
            # Project onto perpendicular of goal_dir
            perp_dir = np.array([-goal_dir[1], goal_dir[0]])  # 90° rotation
            lever_arm = abs(np.dot(hand_to_bottle, perp_dir))
            lever_penalty = -30.0 * lever_arm  # λ = 30 from paper

            # 3. Progress reward
            reward_progress = np.exp(-3.0 * dist_bottle_goal)

            # Combined contact reward
            reward = theta_penalty + lever_penalty + 5.0 * reward_progress + 2.0

        else:
            # === NO CONTACT: Encourage approaching ===
            dist_hand_bottle = np.linalg.norm(hand_pos[:2] - bottle_pos[:2])
            reward = np.exp(-5.0 * dist_hand_bottle) - 2.0

        # ============ YOUR EXISTING CONSTRAINTS ============
        # Keep height and orientation penalties for 3D task

        bottle_radius = 0.03
        sweet_spot_height = 0.13
        sweet_spot_z = bottle_pos[2] + sweet_spot_height

        # Height penalty (critical for 3D - prevent tipping)
        height_diff = hand_pos[2] - sweet_spot_z
        if height_diff < -0.01:
            height_penalty = -50.0 * abs(height_diff)
        elif height_diff < 0:
            height_penalty = -10.0 * abs(height_diff)
        else:
            height_penalty = -0.5 * height_diff

        # Orientation penalty (hand pointing down)
        hand_z_axis = self.data.xmat[self.hand_body_id].reshape(3, 3)[:, 2]
        orientation_penalty = -3.0 * (1.0 + hand_z_axis[2])  # 0 when pointing down

        # Control cost
        reward_ctrl = -0.001 * np.square(self.data.ctrl[:7]).sum()

        # ============ TOTAL REWARD ============
        total_reward = reward + height_penalty + orientation_penalty + reward_ctrl

        # ============ DEBUG ============
        if self.episode_length % 20 == 0:
            print(f"\nStep: {self.episode_length}")
            print(f"  Touching: {is_touching}, Contact made: {self.contact_made}")
            print(f"  Dist to goal: {dist_bottle_goal:.4f}")
            if is_touching and hand_speed_xy > 0.01:
                print(f"  θ alignment: {alignment:.3f} (want 1.0)")
                print(f"  Lever arm d3: {lever_arm:.4f} (want 0.0)")
            print(f"  Height diff: {height_diff:.4f}")
            print(f"  Reward: {total_reward:.2f}")

        # ============ TERMINATION ============
        if dist_bottle_goal < 0.07:
            total_reward += 50.0
            info["is_success"] = True
            print(f"\n *** GOAL REACHED at step {self.episode_length}! ***")

        if hand_pos[2] < sweet_spot_z - 0.03:
            total_reward -= 20.0
            info["hand_too_low"] = True
            print(f"\n *** HAND TOO LOW! ***")

        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        bottle_up_z = bottle_mat[2, 2]
        if bottle_up_z < 0.5:
            total_reward -= 20.0
            info["bottle_fallen"] = True
            print(f"\n *** BOTTLE FALLEN! ***")

        return total_reward, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

        self.current_ctrl = self.data.qpos[7:14].copy()

        # Settle physics
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.goal_pos = self.data.site_xpos[self.goal_site_id].copy()

        # -- Reset contact tracking
        self.contact_made = False
        self.no_contact_frames = 0

        self.episode_length = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Small step size for smooth motion
        step_size = 0.03
        # Clip to hardware limits
        self.current_ctrl = np.clip(
            self.current_ctrl + (action * step_size),
            self.act_low,
            self.act_high
        )

        # Send to MuJoCo
        self.data.ctrl[:7] = self.current_ctrl

        # Step physics multiple times for stability
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, info = self._get_reward()

        self.episode_length += 1

        terminated = info.get("is_success", False) or\
                     info.get("bottle_fallen", False) or\
                     info.get("hand_too_low", False) or\
                     info.get("contact_lost", False) or \
                     info.get("bottle_flying", False) or \
                    info.get("bad_orientation", False)
        truncated = self.episode_length >= 400  # Shorter episodes for pushing

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