import gymnasium as gym
from gymnasium import spaces
import mujoco
# import mujoco.viewer
import numpy as np
import os

# Imports from modules
from .config import *
from .trajectory import generate_trajectory, compute_arc_length
from .geometry import get_progress, get_path_deviation
from .contact import get_contact_force, is_touching
from .control import (
    compute_approach_velocity,
    compute_push_velocity,
    cartesian_to_joint_velocity
)
from .observation import get_observation
from .reward import compute_reward


class PandaPushTrajectoryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_type="straight"):
        super().__init__()

        # ==================== Load Mujoco model ====================
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "..", "robot_environment", "robot_environment.xml")
        xml_path = os.path.abspath(xml_path)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find XML file at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ==================== IDs ====================
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

        # ==================== Control ====================
        self.actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = self.actuator_ranges[:, 0]
        self.act_high = self.actuator_ranges[:, 1]
        self.current_qpos_target = np.zeros(7)

        # ==================== RL state ====================
        self.current_K = np.array([200.0, 200.0])
        self.prev_progress = 0.0
        self.episode_length = 0
        self.in_approach = True
        self.is_settling = False
        self.settle_counter = 0
        self.last_desired_push_dir = np.array([0.0, -1.0], dtype=np.float32)

        # ==================== Trajectory ====================
        self.trajectory_type = trajectory_type
        self.trajectory = None
        self.total_arc_length = 0.0

        # ==================== Misc ====================
        self.render_mode = render_mode
        self.viewer = None

        # ==================== Action space ====================
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ==================== Observation ====================
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # ============================================================
        # RESET
        # ============================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

        self.current_qpos_target = self.data.qpos[7:14].copy()
        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # Initialize trajectory
        bottle_start = self.data.xpos[self.bottle_body_id].copy()
        self.trajectory = generate_trajectory(
            bottle_start[:2],
            GOAL_POSITION,
            self.trajectory_type
        )
        self.total_arc_length = compute_arc_length(self.trajectory)

        # Reset state
        self.prev_progress = 0.0
        self.episode_length = 0
        self.in_approach = True
        self.is_settling = False
        self.settle_counter = 0

        return self._get_obs(), {}

        # ============================================================
        # STEP
        # ============================================================

    def step(self, action):
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        # ==================== Phase logic ====================
        if self.in_approach:
            cart_vel, dist = compute_approach_velocity(self)
            if dist < 0.06 or is_touching(self.model, self.data, self.bottle_body_id, self.robot_contact_bodies):
                self.in_approach = False
        else:
            cart_vel = compute_push_velocity(self, action)

        # ==================== Convert to joint velocities ====================
        q_dot = cartesian_to_joint_velocity(self, cart_vel)

        dt = 0.02
        self.current_qpos_target[:6] = np.clip(
            self.current_qpos_target[:6] + q_dot[:6] * dt,
            self.act_low[:6],
            self.act_high[:6]
        )

        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # ==================== Observation ====================
        obs = self._get_obs()

        # ==================== Reward ====================
        reward, info = compute_reward(self)

        self.episode_length += 1

        # ==================== Termination ====================
        terminated = bool(
            info.get("is_success") or
            info.get("bottle_fallen") or
            info.get("off_path")
        )

        truncated = self.episode_length >= MAX_EPISODE_LENGTH

        return obs, reward, terminated, truncated, info

        # ============================================================
        # OBSERVATION
        # ============================================================

    def _get_obs(self):
        return get_observation(self)

        # ============================================================
        # RENDER
        # ============================================================

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.user_scn.ngeom = 0

            if self.trajectory is not None:
                self.viewer.user_scn.ngeom = 0
                for i in range(len(self.trajectory) - 1):
                    if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                        break

                    p1 = np.array([self.trajectory[i][0], self.trajectory[i][1], 0.801])
                    p2 = np.array([self.trajectory[i + 1][0], self.trajectory[i + 1][1], 0.801])

                    mujoco.mjv_initGeom(
                        self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=[0.003, 0, 0],
                        pos=(p1 + p2) / 2,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32)
                    )
                    mujoco.mjv_connector(
                        self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        0.003,
                        p1, p2
                    )
                    self.viewer.user_scn.ngeom += 1

            self.viewer.sync()

        # ============================================================
        # CLOSE
        # ============================================================

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None