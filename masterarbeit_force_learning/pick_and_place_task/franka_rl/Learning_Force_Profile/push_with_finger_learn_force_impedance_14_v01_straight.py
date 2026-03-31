"""
Panda Push Environment - Lookahead Target Concept (Supervisor's Method)

Key Concept:
    - Push direction = Vector from bottle to LOOKAHEAD TARGET (3-4 points ahead)
    - When ON trajectory: lookahead direction ≈ tangent (forward motion)
    - When OFF trajectory: lookahead direction = correction + forward (automatic blend!)
    - Minimize angle θ between Force P and correction vector
    - No separate forward/correction weighting needed - geometry handles it!

Updates:
    - Blended push direction (tangent when on-track, correction when off-track)
    - Hand repositions to SIDE of bottle when deviation is large
    - Wrist aligns with push direction (hand position handles correction geometry)
    - Faster wrist response
    - 200 trajectory points for finer resolution
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os

from trajectory import TrajectoryManager
from contact import ContactManager
from push_controller import PushController
from reward import RewardManager


class PandaPushTrajectoryEnv(gym.Env):
    """
    Lookahead Target Concept (Supervisor's Method):
        - Push direction = direction to lookahead target (blended with tangent)
        - Hand repositions to side of bottle for correction
        - Wrist aligns with push direction
        - Automatically blends forward + correction based on geometry
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_type="straight"):
        super().__init__()

        # Load model
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "robot_panda_push_force.xml")
        xml_path = os.path.abspath(xml_path)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find XML file at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # IDs
        self.home_key_id = self.model.key("home").id
        self.bottle_body_id = self.model.body("bottle").id
        self.hand_body_id = self.model.body("hand").id
        self.goal_site_id = self.model.site("goal").id
        self.left_finger_body_id = self.model.body("left_finger").id
        self.right_finger_body_id = self.model.body("right_finger").id

        self.robot_contact_bodies = {
            self.hand_body_id,
            self.left_finger_body_id,
            self.right_finger_body_id
        }

        # Control
        self.actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = self.actuator_ranges[:, 0]
        self.act_high = self.actuator_ranges[:, 1]
        self.current_qpos_target = np.zeros(7)

        # ==================== PUSH PARAMETERS ====================
        self.base_forward_speed = 0.015  # Base push speed
        self.behind_distance = 0.04  # Distance hand stays behind bottle

        # ==================== LOOKAHEAD PARAMETERS ====================
        self.lookahead_points = 4  # Look 4 points ahead on trajectory

        # ==================== WRIST ROTATION ====================
        self.gripper_push_angle_at_home = -np.pi / 2  # -90 degrees
        self.max_wrist_rotation = 2.5 # Allow more rotation
        self.wrist_offset = 0.0
        self.base_wrist_pos = 0.0

        # ==================== STIFFNESS (What RL learns!) ====================
        self.K_min = 100.0
        self.K_max = 500.0
        self.current_K = np.array([300.0, 300.0])

        # ==================== TILT SAFETY ====================
        self.tilt_ok = 0.99
        self.tilt_slow = 0.98
        self.tilt_stop = 0.96

        # is_settling and settle_counter are owned by push_ctrl
        self.settle_required = 25

        # ==================== CARTESIAN CONTROL ====================
        self.target_z = 0.92
        self.z_gain = 10.0
        self.damping = 0.01

        # ==================== CONTACT ====================
        self.contact_manager = ContactManager(self.model, self.data, self.robot_contact_bodies, self.bottle_body_id)

        # ==================== TRAJECTORY ====================
        self.traj_manager = TrajectoryManager(goal_position=np.array([0.4, -0.2]), path_tolerance=0.05)
        self.goal_position = self.traj_manager.goal_position

        # ==================== PUSH CONTROLLER ====================
        # Note: current_K and is_settling are owned by push_ctrl.
        # The env references below are aliases so all existing code keeps working.
        self.push_ctrl = PushController(
            model=self.model,
            data=self.data,
            traj_manager=self.traj_manager,
            contact_manager=self.contact_manager,
            hand_body_id=self.hand_body_id,
            bottle_body_id=self.bottle_body_id,
            goal_position=self.goal_position,
            lookahead_points=self.lookahead_points,
            base_forward_speed=self.base_forward_speed,
            behind_distance=self.behind_distance,
            target_z=self.target_z,
            z_gain=self.z_gain,
            damping=self.damping,
            K_min=self.K_min,
            K_max=self.K_max,
            tilt_ok=self.tilt_ok,
            tilt_slow=self.tilt_slow,
            tilt_stop=self.tilt_stop,
            settle_required=self.settle_required,
        )
        # Alias push_ctrl's mutable state so env references stay in sync
        self.current_K = self.push_ctrl.current_K

        # ==================== REWARD ====================
        self.reward_manager = RewardManager(
            path_tolerance=self.traj_manager.path_tolerance,
            target_z=self.target_z,
        )

        # adaptie target_z
        self.bottle_start_y = 0.2

        # Phase
        self.in_approach = True

        # Action space: [forward_mod, lateral_mod, Kx, Ky]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space
        obs_dim = 39  # Added: lookahead_direction (2) + angle_error (1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State
        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0
        self.max_episode_length = 2500
        self.prev_progress = 0.0

        # Logging
        self.force_profile_log = []
        self.stiffness_profile_log = []
        self.deviation_log = []

        print(f"\n{'=' * 60}")
        print("LOOKAHEAD TARGET CONCEPT (SUPERVISOR'S METHOD)")
        print(f"{'=' * 60}")
        print("Push direction = Blended tangent + correction to LOOKAHEAD target")
        print("Hand repositions to SIDE of bottle for correction")
        print(f"Lookahead points: {self.lookahead_points}")
        print(f"K range: [{self.K_min}, {self.K_max}] N/m")
        print(f"Goal position: {self.goal_position}")
        print(f"{'=' * 60}")

    # ==================================================================
    #           CORE CONCEPT: Lookahead Target + Blended Direction
    # ==================================================================

    def _get_lookahead_target(self, bottle_xy):
        return self.push_ctrl.get_lookahead_target(bottle_xy)

    def _get_push_direction(self, bottle_xy):
        return self.push_ctrl.get_push_direction(bottle_xy)

    def _get_angle_error(self, bottle_xy):
        return self.push_ctrl.get_angle_error(bottle_xy)

    def _get_hand_target_position(self, bottle_xy):
        return self.push_ctrl.get_hand_target_position(bottle_xy)

    # ==================================================================
    #           VELOCITY COMPUTATION
    # ==================================================================

    def _compute_approach_velocity(self):
        return self.push_ctrl.compute_approach_velocity()

    def _compute_push_velocity(self, action):
        return self.push_ctrl.compute_push_velocity(action, self.episode_length, self.bottle_start_y)

    def _get_bottle_tilt(self):
        return self.push_ctrl.get_bottle_tilt()

    def _check_stable(self):
        return self.push_ctrl._check_stable()

    def _cartesian_to_joint_velocity(self, cart_vel):
        return self.push_ctrl.cartesian_to_joint_velocity(cart_vel)

    # def _compute_wrist_rotation(self, bottle_xy):
    #     """
    #     Align gripper's -X axis with PUSH DIRECTION (to lookahead target).
    #     This minimizes angle θ between force P and desired direction!
    #     """
    #     # Get push direction (to lookahead target)
    #     push_dir, _, _ = self._get_push_direction(bottle_xy)
    #
    #     # Angle of push direction in world frame
    #     push_angle = np.arctan2(push_dir[1], push_dir[0])
    #
    #     # Rotation needed to align gripper -X with push direction
    #     alignment_rotation = push_angle - self.gripper_push_angle_at_home
    #
    #     # Normalize to [-π, π]
    #     while alignment_rotation > np.pi:
    #         alignment_rotation -= 2 * np.pi
    #     while alignment_rotation < -np.pi:
    #         alignment_rotation += 2 * np.pi
    #
    #     # Clamp to joint limits
    #     target_rotation = np.clip(alignment_rotation, -self.max_wrist_rotation, self.max_wrist_rotation)
    #
    #     # Debug output
    #     if self.episode_length % 100 == 0:
    #         print(f"    Wrist: push_angle={np.degrees(push_angle):.1f}°, "
    #               f"rotation={np.degrees(target_rotation):.1f}°")
    #
    #     return target_rotation

    # ==================================================================
    #                      TRAJECTORY
    # ==================================================================



    # ==================================================================
    #                      CONTACT
    # ==================================================================

    def _get_contact_force(self):
        return self.contact_manager.get_contact_force()

    def _is_touching(self):
        return self.contact_manager.is_touching()

    # ==================================================================
    #                      OBSERVATION
    # ==================================================================

    def _get_obs(self):
        """
        Observation includes:
        - Robot state (qpos, qvel)
        - Positions (hand, bottle)
        - Push direction (to lookahead target)
        - Deviation from trajectory
        - Angle error θ between force and push direction
        - Contact info
        """
        qpos = self.data.qpos[7:14].astype(np.float32)
        qvel = self.data.qvel[6:13].astype(np.float32)
        hand_pos = self.data.xpos[self.hand_body_id].astype(np.float32)
        bottle_pos = self.data.xpos[self.bottle_body_id].astype(np.float32)
        bottle_xy = bottle_pos[:2]

        # Direction to bottle
        hand_to_bottle = bottle_pos[:2] - hand_pos[:2]
        dist_to_bottle = np.linalg.norm(hand_to_bottle)
        dir_to_bottle = hand_to_bottle / (dist_to_bottle + 1e-6)

        # Push direction (to lookahead target) - THE KEY OBSERVATION!
        push_dir, dist_to_target, _ = self._get_push_direction(bottle_xy)

        # Deviation from trajectory
        deviation_vec, deviation_mag = self.traj_manager._get_path_deviation(bottle_xy)

        # Angle error between force and push direction
        angle_error = self._get_angle_error(bottle_xy)

        # Progress
        progress = self.traj_manager._get_progress(bottle_xy)

        # Contact
        contact_force = self._get_contact_force()
        is_touching = np.array([1.0 if self._is_touching() else 0.0], dtype=np.float32)

        # Stiffness
        K_normalized = (self.current_K - self.K_min) / (self.K_max - self.K_min)
        settling_flag = np.array([1.0 if self.push_ctrl.is_settling else 0.0], dtype=np.float32)

        wrist_normalized = np.array([self.wrist_offset / self.max_wrist_rotation], dtype=np.float32)

        obs = np.concatenate([
            qpos,  # 7
            qvel,  # 7
            hand_pos,  # 3
            bottle_pos,  # 3
            dir_to_bottle,  # 2
            [dist_to_bottle],  # 1
            push_dir,  # 2  (NEW: direction to lookahead target)
            [dist_to_target],  # 1  (NEW: distance to lookahead target)
            deviation_vec,  # 2
            [deviation_mag],  # 1
            [angle_error],  # 1  (NEW: angle θ between force and push dir)
            [progress],  # 1
            contact_force,  # 3
            is_touching,  # 1
            K_normalized,  # 2
            settling_flag,  # 1
            wrist_normalized, #1
        ])  # Total: 39

        return obs.astype(np.float32)

    # ==================================================================
    #                      REWARD
    # ==================================================================

    def _get_reward(self):
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        hand_pos = self.data.xpos[self.hand_body_id]

        _, deviation_mag = self.traj_manager._get_path_deviation(bottle_xy)
        progress = self.traj_manager._get_progress(bottle_xy)
        tilt = self._get_bottle_tilt()
        is_touching = self._is_touching()
        force_mag = np.linalg.norm(self._get_contact_force())
        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)
        angle_error = self._get_angle_error(bottle_xy)

        # sync in_approach back to env after reward_manager may flip it
        result = self.reward_manager.compute(
            deviation_mag=deviation_mag,
            progress=progress,
            tilt=tilt,
            is_touching=is_touching,
            force_mag=force_mag,
            dist_to_bottle=dist_to_bottle,
            angle_error=angle_error,
            hand_z=hand_pos[2],
            K_avg=float(np.mean(self.current_K)),
            is_settling=self.push_ctrl.is_settling,
            episode_length=self.episode_length,
        )
        self.in_approach = self.reward_manager.in_approach
        self.prev_progress = self.reward_manager.prev_progress
        return result

    # ==================================================================
    #                      RESET / STEP
    # ==================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

        self.current_qpos_target = self.data.qpos[7:14].copy()
        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(100):
            self.data.ctrl[:7] = self.current_qpos_target
            mujoco.mj_step(self.model, self.data)

        bottle_start = self.data.xpos[self.bottle_body_id].copy()

        # adaptive target-z
        self.bottle_start_y = bottle_start[1]

        traj_type = self.traj_manager.trajectory_type
        if options and "trajectory_type" in options:
            traj_type = options["trajectory_type"]

        self.traj_manager.generate_trajectory(bottle_start[:2], traj_type)

        self.episode_length = 0
        self.push_ctrl.current_K[:] = [200.0, 200.0]
        self.push_ctrl.is_settling = False
        self.push_ctrl.settle_counter = 0
        self.reward_manager.reset()
        self.in_approach = self.reward_manager.in_approach
        self.prev_progress = self.reward_manager.prev_progress

        self.base_wrist_pos = self.current_qpos_target[6]
        self.wrist_offset = 0.0

        self.force_profile_log = []
        self.stiffness_profile_log = []
        self.deviation_log = []
        self.push_ctrl.angle_error_log = []

        print(f"\n{'=' * 50}")
        print(f"EPISODE: {traj_type}")
        print(f"Bottle start: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"Goal: ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})")
        print(f"Trajectory arc length: {self.traj_manager.total_arc_length:.3f}m")
        print(f"Lookahead points: {self.lookahead_points}")
        print(f"{'=' * 50}")

        return self._get_obs(), {}

    def step(self, action):
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        if self.in_approach:
            cart_vel, dist = self._compute_approach_velocity()
            target_wrist_rotation = 0.0
            if dist < 0.06 or self._is_touching():
                self.in_approach = False
                print(f"Step {self.episode_length}: → PUSH phase")
        else:
            cart_vel = self._compute_push_velocity(action)
            # target_wrist_rotation = self._compute_wrist_rotaiton()
            target_wrist_rotation = action[1] * self.max_wrist_rotation # RL controls wrist

        q_dot = self._cartesian_to_joint_velocity(cart_vel)

        dt = 0.02

        self.current_qpos_target[:6] = np.clip(
            self.current_qpos_target[:6] + q_dot[:6] * dt,
            self.act_low[:6], self.act_high[:6]
        )

        wrist_error = target_wrist_rotation - self.wrist_offset
        wrist_speed = 5.0 * wrist_error # before 2.0
        # wrist_speed = np.clip(wrist_speed, -0.5, 0.5)
        wrist_speed = np.clip(wrist_speed, -2.0, 2.0)

        self.wrist_offset += wrist_speed * dt
        self.wrist_offset = np.clip(self.wrist_offset, -self.max_wrist_rotation, self.max_wrist_rotation)

        self.current_qpos_target[6] = np.clip(
            self.base_wrist_pos + self.wrist_offset,
            self.act_low[6], self.act_high[6]
        )

        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Logging
        force = self._get_contact_force()
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        _, dev = self.traj_manager._get_path_deviation(bottle_xy)

        self.force_profile_log.append(force.copy())
        self.stiffness_profile_log.append(self.current_K.copy())
        self.deviation_log.append(dev)

        obs = self._get_obs()
        reward, info = self._get_reward()
        self.episode_length += 1

        terminated = bool(info.get("is_success") or info.get("bottle_fallen") or info.get("off_path"))
        truncated = bool(self.episode_length >= self.max_episode_length)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # Install the custom drawing callback
                self.viewer.user_scn.ngeom = 0  # reset custom geoms

            # Draw trajectory
            if self.traj_manager.trajectory is not None:
                self.viewer.user_scn.ngeom = 0  # clear previous frame's geoms
                for i in range(len(self.traj_manager.trajectory) - 1):
                    if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                        break

                    p1 = np.array([self.traj_manager.trajectory[i][0], self.traj_manager.trajectory[i][1], 0.801])
                    p2 = np.array([self.traj_manager.trajectory[i + 1][0], self.traj_manager.trajectory[i + 1][1], 0.801])

                    mujoco.mjv_initGeom(
                        self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=[0.003, 0, 0],  # radius
                        pos=(p1 + p2) / 2,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32)
                    )
                    # Orient capsule from p1 to p2
                    mujoco.mjv_connector(
                        self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        0.003,  # width
                        p1, p2
                    )
                    self.viewer.user_scn.ngeom += 1

            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None