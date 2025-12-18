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

    def _is_touching(self):
        # returns true if the mujoco physics engine detects a collision between the hand and the bottle
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            match_1 = (body1 == self.hand_body_id and body2 == self.bottle_body_id)
            match_2 = (body1 == self.bottle_body_id and body2 == self.hand_body_id)

            if match_1 or match_2:
                return True

        return False


    def _get_reward(self):
        info = {"is_success": False}

        bottle_pos = self.data.xpos[self.bottle_body_id]
        hand_pos = self.data.xpos[self.hand_body_id]

        # sweet spot-> at bottle surface, opposite to push direction
        bottle_radius = 0.03
        sweet_spot_height = 0.13

        push_dir, dist_bottle_goal = self._compute_push_direction()

        sweet_spot = bottle_pos.copy()
        sweet_spot[:2] -= push_dir[:2] * bottle_radius
        sweet_spot[2] = bottle_pos[2] + sweet_spot_height

        ##### REWARDS #####
        # -- Distances --
        # -- 1. APPROACH XY: Get hand to sweet spot in XY plane --
        dist_xy = np.linalg.norm(hand_pos[:2] - sweet_spot[:2])
        dist_z = abs(hand_pos[2] - sweet_spot[2])
        # -- 2. Total distance (for success check) --
        dist_total = np.linalg.norm(hand_pos - sweet_spot)

        ##### Contact Check #####
        is_touching = self._is_touching()
        if is_touching:
            self.contact_made = True

        ##### Approach first (before contact) #####
        # -- 1. Approach xy --
        reward_approach_xy = -dist_xy

        # -- 2. APPROACH Z: Get hand to correct height --
        reward_approach_z = -dist_z

        # -- 3. Close Bonus --
        reward_close = np.exp(-10.0 * dist_total)

        # -- 4. ORIENTATION: Keep hand pointing down (Z-axis toward -Z world) --
        hand_z_axis = self.data.xmat[self.hand_body_id].reshape(3, 3)[:, 2]
        # hand_z_axis[2] should be -1 (pointing down), penalize if not
        orientation_z_reward = -1.0 - hand_z_axis[2]  # Negative when pointing down (good)

        # -- 5. ORIENTATION X: Hand X-axis aligned with push direction --
        hand_x_axis = self.data.xmat[self.hand_body_id].reshape(3, 3)[:, 0]
        alignment = abs(np.dot(hand_x_axis[:2], push_dir[:2]))  # Want +1.0
        orientation_x_reward = alignment - 1.0  # +1 when aligned, -1 when opposite

        # -- 6. HEIGHT PENALTY: Asymmetric - heavily penalize dropping below sweet spot --
        height_diff = hand_pos[2] - sweet_spot[2]
        if height_diff < -0.01:  # Below sweet spot
            height_penalty = -100.0 * abs(height_diff)  # Strong negative penalty # 10.0
        elif height_diff < 0:
            height_penalty = -20.0 * abs(height_diff)
        else:  # Above sweet spot
            height_penalty = -0.5 * height_diff  # Mild penalty # -0.5

        ##### CONTROL PENALTY #####
        reward_ctrl = -0.001 * np.square(self.data.ctrl[:7]).sum()

        ##### Contact #####
        # -- Contact Reward: Bonus for touching
        reward_contact = 2.0 if is_touching else 0.0

        # -- contact loss penalty: once contact is made, losing it is very BAD --
        contact_loss_penalty = 0.0
        if self.contact_made and not is_touching:
            contact_loss_penalty = -5.0

        # -- stay at bottle: once contact is made, stay close to bottle surface --
        reward_stay_close = 0.0
        if self.contact_made:
            # -- Distance from hand to bottle surface (should be ~0 when touching)
            dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_pos[:2] - bottle_radius)
            reward_stay_close = -10.0 * max(0, dist_to_bottle) # penalize being far from bottle

            # ============ DEBUG OUTPUT ============
            if self.episode_length % 3 == 0:
                print("\n" + "=" * 60)
                print(f"Step: {self.episode_length}")
                print("-" * 60)

                print(f"Hand pos:      [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
                print(f"Bottle pos:    [{bottle_pos[0]:.3f}, {bottle_pos[1]:.3f}, {bottle_pos[2]:.3f}]")
                print(f"Sweet spot:    [{sweet_spot[0]:.3f}, {sweet_spot[1]:.3f}, {sweet_spot[2]:.3f}]")
                print(f"Goal pos:      [{self.goal_pos[0]:.3f}, {self.goal_pos[1]:.3f}, {self.goal_pos[2]:.3f}]")

                print("-" * 60)

                print(f"Dist hand->sweet:  {dist_total:.4f} m")
                print(f"Dist bottle->goal: {dist_bottle_goal:.4f} m")
                print(f"Height diff:       {height_diff:.4f} m")

                print("-" * 60)

                print(f"X-axis align:   {alignment:.3f} (want +1.0)")
                print(f"Z-axis[2]:      {hand_z_axis[2]:.3f} (want -1.0)")
                print(f"Is touching:    {is_touching}")
                print(f"Contact made:   {self.contact_made}")

                print("-" * 60)

                print("REWARD BREAKDOWN:")
                print(f"  R_approach_xy:    {5.0 * reward_approach_xy:+.4f}")
                print(f"  R_approach_z:     {5.0 * reward_approach_z:+.4f}")
                print(f"  R_close:          {3.0 * reward_close:+.4f}")
                print(f"  R_orient_z:       {2.0 * orientation_z_reward:+.4f}")
                print(f"  R_orient_x:       {2.0 * orientation_x_reward:+.4f}")
                print(f"  R_height:         {height_penalty:+.4f}")
                print(f"  R_contact:        {reward_contact:+.4f}")
                print(f"  R_contact_loss:   {contact_loss_penalty:+.4f}")
                print(f"  R_stay_close:     {reward_stay_close:+.4f}")
                print(f"  R_ctrl:           {reward_ctrl:+.4f}")

                print("=" * 60)
            # ============ END DEBUG ============

        # === TOTAL REWARD ===
        total_reward = (
                # -- reward for approach --
                5.0 * reward_approach_xy +
                8.0 * reward_approach_z +
                3.0 * reward_close +
                2.0 * orientation_z_reward +
                2.0 * orientation_x_reward +
                height_penalty +
                # -- reward for contact --
                reward_contact +
                contact_loss_penalty +
                reward_stay_close +
                # -- control --
                reward_ctrl
        )

        # --- SUCCESS: Hand reached sweet spot ---
        if dist_total < 0.05:
            total_reward += 10.0
            info["is_success"] = True
            print(f"\n *** SWEET SPOT REACHED at step {self.episode_length}! ***")
            print(f"     Final dist: {dist_total:.4f} m")
            print(f"     X-axis alignment: {alignment:.3f}")

        # --- FAILURE: Hand too low ---
        if hand_pos[2] < sweet_spot[2] - 0.02:
            total_reward -= 20.0
            info["hand_too_low"] = True
            print(f"\n *** HAND TOO LOW at step {self.episode_length}! Z={hand_pos[2]:.3f} ***")

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
        step_size = 0.05
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
                     info.get("contact_lost", False)
        truncated = self.episode_length >= 300  # Shorter episodes for pushing

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