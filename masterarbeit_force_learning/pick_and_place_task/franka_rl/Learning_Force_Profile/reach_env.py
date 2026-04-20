"""
Reach Phase Environment for Panda Robot.

Phase 1 of the two-phase hierarchical approach:
- Goal: Learn to reach from home position to stable contact with bottle
- Action Space: 3D velocity [vx, vy, vz]
- Observation: Robot state + bottle position + target info
- Reward: Distance-based + contact bonus
- Terminates: When stable contact is achieved (post-condition for push phase)

TRAJECTORY-AGNOSTIC DESIGN:
This environment does NOT depend on which trajectory will be pushed.
The EE simply approaches the bottle from whatever side is natural
given the home position, makes contact at the correct height, and
holds. The push phase handles all trajectory alignment (wrist rotation)
after taking over from this post-condition.

Post-condition (universal, trajectory-independent):
- End-effector is in contact with bottle at correct height
- EE is oriented toward the bottle (facing it)
- Bottle is still upright (not knocked over)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

from reach_config import ReachConfig, get_reach_config
from contact import ContactManager
from renderer import TrajectoryRenderer


class PandaReachEnv(gym.Env):
    """
    Gymnasium environment for learning to reach a bottle.

    The agent learns to move the end-effector from home position
    to stable contact with the bottle, ready for pushing.

    This is trajectory-agnostic: the approach direction is determined
    by the hand-to-bottle vector, not by any trajectory. The push
    phase handles trajectory alignment after contact is established.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, config: ReachConfig = None):
        """
        Initialize the reach environment.

        Args:
            render_mode: "human" for visualization, None for training
            config: ReachConfig instance (uses default if None)
        """
        super().__init__()

        self.config = config or get_reach_config()
        self.render_mode = render_mode

        # Load MuJoCo model
        self._load_model()

        # Get body IDs
        self._init_body_ids()

        # Initialize managers
        self._init_managers()

        # Action space: [vx, vy, vz] in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config.action_dim,),
            dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.obs_dim,),
            dtype=np.float32
        )

        # Episode state
        self.episode_length = 0
        self.contact_hold_counter = 0
        self.p_des = None  # Desired position for velocity integration

        # Approach direction: computed once per episode in reset()
        # from hand-to-bottle vector (trajectory-independent)
        self.approach_dir_2d = None

        # Print initialization info
        self._print_init_info()

    def _load_model(self):
        """Load MuJoCo model from XML."""
        model_path = Path(__file__).parent / "robot_panda_push_force.xml"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Get home keyframe
        self.home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )

    def _init_body_ids(self):
        """Initialize body IDs for robot and objects."""
        self.hand_body_id = self.model.body("hand").id
        self.bottle_body_id = self.model.body("bottle").id
        self.gripper_site_id = self.model.site("gripper_center").id

        # Robot contact bodies (for contact detection)
        self.robot_contact_bodies = set()
        for name in ["hand", "left_finger", "right_finger"]:
            try:
                self.robot_contact_bodies.add(self.model.body(name).id)
            except KeyError:
                pass

    def _init_managers(self):
        """Initialize helper managers."""
        # Contact manager
        self.contact_manager = ContactManager(
            self.model,
            self.data,
            self.robot_contact_bodies,
            self.bottle_body_id
        )

        # Renderer
        self.renderer = TrajectoryRenderer(self.model, self.data, self.render_mode)

    def _print_init_info(self):
        """Print initialization information."""
        print("\n" + "=" * 60)
        print("REACH PHASE ENVIRONMENT (trajectory-agnostic)")
        print("=" * 60)
        print(f"Action space: [vx, vy, vz] (3D velocity)")
        print(f"Observation dim: {self.config.obs_dim}")
        print(f"Velocity limit: +/-{self.config.v_max} m/s")
        print(f"Target height (hand origin): {self.config.target_z} m")
        print(f"Bottle radius: {self.config.bottle_radius} m")
        print(f"Contact hold required: {self.config.contact_hold_steps} steps")
        print("=" * 60 + "\n")

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.

        Computes approach direction from initial hand-to-bottle vector.
        This is trajectory-independent: the EE approaches from wherever
        the home position is, toward the bottle.

        Returns:
            obs: Initial observation
            info: Empty dict
        """
        super().reset(seed=seed)

        # Reset MuJoCo to home keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

        # Let simulation settle with gravity compensation
        for _ in range(200):
            tau_gravity = self.data.qfrc_bias[6:13].copy()
            self.data.ctrl[:7] = tau_gravity
            mujoco.mj_step(self.model, self.data)

        # Reset episode state
        self.episode_length = 0
        self.contact_hold_counter = 0
        

        # Compute approach direction from hand to bottle (2D, XY plane)
        hand_pos = self.data.xpos[self.hand_body_id]
        self.p_des = np.array([hand_pos[0], hand_pos[1], self.config.target_z])
        bottle_pos = self.data.xpos[self.bottle_body_id]
        self.bottle_start_xy = bottle_pos[:2].copy()
        to_bottle = bottle_pos[:2] - hand_pos[:2]
        norm = np.linalg.norm(to_bottle)
        if norm > 1e-6:
            self.approach_dir_2d = (to_bottle / norm).astype(np.float32)
        else:
            self.approach_dir_2d = np.array([0.0, -1.0], dtype=np.float32)

        print(f"\n{'='*50}")
        print(f"REACH EPISODE START")
        print(f"Hand start: ({hand_pos[0]:.2f}, {hand_pos[1]:.2f}, {hand_pos[2]:.2f})")
        print(f"Bottle: ({bottle_pos[0]:.2f}, {bottle_pos[1]:.2f}, {bottle_pos[2]:.2f})")
        print(f"Approach dir: ({self.approach_dir_2d[0]:.3f}, "
              f"{self.approach_dir_2d[1]:.3f})")
        print(f"{'='*50}")

        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: [vx, vy, vz] in [-1, 1]

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Compute and apply torques
        tau = self._compute_torque(action)

        # Step simulation
        for _ in range(self.config.n_substeps):
            self.data.ctrl[:7] = tau
            mujoco.mj_step(self.model, self.data)

        # Get state info
        info = self._get_info()

        # Compute reward
        reward = self._compute_reward(action, info)

        # Update contact hold counter
        if info["is_touching"]:
            self.contact_hold_counter += 1
        else:
            self.contact_hold_counter = 0

        # Check termination
        terminated = False

        # Success: stable contact achieved
        if self.contact_hold_counter >= self.config.contact_hold_steps:
            info["reach_success"] = True
            reward += self.config.stable_contact_bonus
            terminated = True
            print(f"  REACH SUCCESS at step {self.episode_length}!")
            self._print_post_condition(info)

        # Failure: knocked over bottle
        if info["bottle_tilt"] < self.config.max_tilt_during_approach:
            info["bottle_knocked"] = True
            reward -= 50.0
            terminated = True
            print(f"  Bottle knocked at step {self.episode_length} "
                  f"(tilt={info['bottle_tilt']:.4f})")

        # Update episode length
        self.episode_length += 1
        truncated = self.episode_length >= self.config.max_episode_length

        if truncated and not terminated:
            print(f"  Episode truncated at step {self.episode_length}")

        # Debug print
        # if self.episode_length % 10 == 0:
        if self.episode_length % 1 == 0:
            self._print_debug(info)

        return self._get_observation(), reward, terminated, truncated, info

    def _compute_torque(self, action):
        """
        Compute joint torques from velocity action.

        Uses velocity integration + PD control in task space.
        Orientation controller dynamically points EE toward the bottle.
        """
        # Scale action to velocity
        v_des = np.array([
            action[0] * self.config.v_max,
            action[1] * self.config.v_max,
            # action[2] * self.config.v_max
            0.0
        ])

        # Initialize desired position
        if self.p_des is None:
            self.p_des = self._get_ee_position()

        # Integrate velocity to position
        self.p_des = self.p_des + v_des * self.config.dt * self.config.n_substeps

        # Hard floor: never command the hand below target height
        # This prevents the fingers from hitting the table
        self.p_des[2] = max(self.p_des[2], self.config.target_z)

        # Current state
        p_current = self._get_ee_position()
        v_current = self._get_ee_velocity()
        omega_current = self._get_ee_angular_velocity()
        R_current = self._get_ee_orientation()

        # Position PD control
        p_error = self.p_des - p_current
        v_error = v_des - v_current
        F_cmd = self.config.Kp * p_error + self.config.Kd * v_error

        # Limit force command
        F_cmd_mag = np.linalg.norm(F_cmd)
        if F_cmd_mag > 30.0:
            F_cmd = F_cmd / F_cmd_mag * 30.0

        # Orientation control: dynamically point EE toward bottle
        # This is trajectory-independent — the EE always faces the bottle
        bottle_pos = self.data.xpos[self.bottle_body_id][:2]
        hand_pos = self._get_ee_position()[:2]
        to_bottle = bottle_pos - hand_pos
        to_bottle_norm = np.linalg.norm(to_bottle)
        if to_bottle_norm > 0.01:
            face_dir_2d = to_bottle / to_bottle_norm
        else:
            face_dir_2d = self.approach_dir_2d

        R_desired = self._compute_desired_orientation(face_dir_2d)
        theta_error = self._compute_orientation_error(R_current, R_desired)
        omega_error = -omega_current
        tau_rot_cmd = self.config.Kp_rot * theta_error + self.config.Kd_rot * omega_error

        # Limit orientation torque
        tau_rot_mag = np.linalg.norm(tau_rot_cmd)
        if tau_rot_mag > 5.0:
            tau_rot_cmd = tau_rot_cmd / tau_rot_mag * 5.0

        # Combine into 6D wrench
        wrench_cmd = np.concatenate([F_cmd, tau_rot_cmd])

        # Jacobian transpose
        J_full = self._get_jacobian_full()
        tau_task = J_full.T @ wrench_cmd

        # Add gravity compensation
        tau_gravity = self.data.qfrc_bias[6:13].copy()
        tau = tau_task + tau_gravity

        # Clip to limits
        tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
        tau = np.clip(tau, -tau_max, tau_max)

        return tau.astype(np.float32)

    def _get_observation(self):
        """
        Build observation vector.

        Observation (32-dim):
            qpos(7) + qvel(7) + hand_pos(3) + hand_vel(3) +
            bottle_pos(3) + dir_to_target(3) + dist_to_target(1) +
            bottle_tilt(1) + is_touching(1) + ee_x_axis(3)
        """
        # Joint state
        qpos = self.data.qpos[7:14].astype(np.float32)
        qvel = self.data.qvel[6:13].astype(np.float32)

        # End-effector state
        hand_pos = self._get_ee_position().astype(np.float32)
        hand_vel = self._get_ee_velocity().astype(np.float32)

        # Bottle state
        bottle_pos = self.data.xpos[self.bottle_body_id].astype(np.float32)

        # Target position accounting for bottle radius
        target_pos = self._get_target_position()
        dir_to_target = target_pos - hand_pos
        dist_to_target = np.linalg.norm(dir_to_target)
        if dist_to_target > 0.001:
            dir_to_target = dir_to_target / dist_to_target
        else:
            dir_to_target = np.zeros(3)

        # Bottle tilt
        bottle_tilt = self._get_bottle_tilt()

        # Contact
        is_touching = np.array([1.0 if self.contact_manager.is_touching() else 0.0],
                               dtype=np.float32)

        # EE orientation (x-axis)
        R_ee = self._get_ee_orientation()
        ee_x_axis = R_ee[:, 0].astype(np.float32)

        obs = np.concatenate([
            qpos,                                   # 7
            qvel,                                   # 7
            hand_pos,                               # 3
            hand_vel,                               # 3
            bottle_pos,                             # 3
            dir_to_target.astype(np.float32),       # 3
            [dist_to_target],                       # 1
            [bottle_tilt],                          # 1
            is_touching,                            # 1
            ee_x_axis,                              # 3
        ])  # Total: 32

        return obs.astype(np.float32)

    def _get_target_position(self):
        """
        Get target contact position accounting for bottle geometry.

        The target is on the bottle's skin facing the approach direction,
        offset by behind_distance so the EE presses slightly into the
        bottle for stable contact.

        This is trajectory-independent: the approach direction comes from
        the hand-to-bottle vector computed at episode start.
        """
        bottle_pos = self.data.xpos[self.bottle_body_id]

        # Contact surface on the bottle's skin (side facing the hand)
        # approach_dir_2d points FROM hand TOWARD bottle, so the contact
        # surface on the near side is at bottle - approach_dir * radius
        contact_surface_x = bottle_pos[0] - self.approach_dir_2d[0] * self.config.bottle_radius
        contact_surface_y = bottle_pos[1] - self.approach_dir_2d[1] * self.config.bottle_radius

        # EE target is behind_distance further back from the surface
        target = np.array([
            contact_surface_x - self.approach_dir_2d[0] * self.config.behind_distance,
            contact_surface_y - self.approach_dir_2d[1] * self.config.behind_distance,
            self.config.target_z
        ])

        return target

    def _get_info(self):
        """Get current state information."""
        hand_pos = self._get_ee_position()
        bottle_pos = self.data.xpos[self.bottle_body_id]
        target_pos = self._get_target_position()

        dist_to_target = np.linalg.norm(hand_pos - target_pos)
        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_pos[:2])

        return {
            "dist_to_target": dist_to_target,
            "dist_to_bottle": dist_to_bottle,
            "is_touching": self.contact_manager.is_touching(),
            "bottle_tilt": self._get_bottle_tilt(),
            "contact_hold": self.contact_hold_counter,
            "hand_pos": hand_pos.copy(),
            "bottle_pos": bottle_pos.copy(),
        }

# ==============================================================
#       Reward Computation
# ==============================================================

    def _compute_reward(self, action, info):
        """
        Compute reward for reach task.

        Rewards:
        - Negative distance to target (encourage approaching)
        - Bonus for approaching velocity (when far)
        - Penalty for high velocity when close (gentle approach)
        - Height alignment bonus
        - Lateral alignment (keep hand centered on bottle)
        - Contact bonus with gentle touch bonus
        - Penalty for pushing bottle after contact
        - Penalty for velocity during contact
        - Tilt penalty
        - Small time penalty
        """
        reward = 0.0

        # Distance reward (negative distance)
        dist = info["dist_to_target"]
        reward -= self.config.w_distance * dist

        # Get velocity info
        hand_vel = self._get_ee_velocity()
        vel_magnitude = np.linalg.norm(hand_vel)
        target_pos = self._get_target_position()
        hand_pos = self._get_ee_position()
        to_target = target_pos - hand_pos
        to_target_norm = np.linalg.norm(to_target)

        # Distance thresholds for velocity control
        slow_down_distance = 0.08

        if to_target_norm > slow_down_distance:
            # FAR: Reward approaching velocity
            if to_target_norm > 0.01:
                height_error = abs(hand_pos[2] - self.config.target_z)
                if height_error < 0.02: # Only reward approach when within 2cm of target height
                    to_target_dir = to_target / to_target_norm
                    approach_vel = np.dot(hand_vel, to_target_dir)
                    reward += self.config.w_approach_velocity * max(approach_vel, 0)
        else:
            # CLOSE: Penalize high velocity (encourage gentle approach)
            closeness_factor = 1.0 - (to_target_norm / slow_down_distance)
            desired_max_vel = 0.08 * (1.0 - closeness_factor * 0.9)

            if vel_magnitude > desired_max_vel:
                excess_vel = vel_magnitude - desired_max_vel
                velocity_penalty = self.config.w_velocity_penalty * excess_vel * (1.0 + closeness_factor)
                reward -= velocity_penalty

        # Height alignment reward
        height_error = abs(hand_pos[2] - self.config.target_z)
        reward -= self.config.w_height * height_error

        # Lateral alignment: penalize offset perpendicular to approach direction
        # This keeps the hand centered on the bottle, not drifting sideways
        bottle_pos = info["bottle_pos"]
        offset = hand_pos[:2] - bottle_pos[:2]
        lateral_error = abs(
            offset[0] * (-self.approach_dir_2d[1]) +
            offset[1] * self.approach_dir_2d[0]
        )
        reward -= 60.0 * lateral_error

        # Contact bonus
        if info["is_touching"]:
            reward += self.config.contact_bonus * 0.1

            # Extra bonus for gentle contact (low velocity at contact)
            if vel_magnitude < 0.01:
                reward += 10.0

            # Penalty for pushing the bottle — it should stay in place
            bottle_displacement = np.linalg.norm(
                bottle_pos[:2] - self.bottle_start_xy
            )
            if bottle_displacement > 0.005:
                reward -= 200.0 * bottle_displacement

            # Penalty for any velocity while in contact (should be stationary)
            if vel_magnitude > 0.01:
                reward -= 100.0 * vel_magnitude

        # Tilt penalty
        tilt = info["bottle_tilt"]
        if tilt < 0.99:
            reward -= 100 * (0.99 - tilt)

        # Time penalty
        reward -= self.config.time_penalty

        return reward

    def _print_post_condition(self, info):
        """Print the post-condition state for verification against push pre-condition."""
        hand_pos = info["hand_pos"]
        bottle_pos = info["bottle_pos"]
        print(f"\n--- REACH POST-CONDITION (trajectory-independent) ---")
        print(f"  Hand pos:  ({hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f})")
        print(f"  Bottle:    ({bottle_pos[0]:.3f}, {bottle_pos[1]:.3f}, {bottle_pos[2]:.3f})")
        print(f"  Tilt:      {info['bottle_tilt']:.4f}")
        print(f"  Contact:   {info['is_touching']}")
        print(f"  Approach:  ({self.approach_dir_2d[0]:.3f}, "
              f"{self.approach_dir_2d[1]:.3f})")
        # --- THE CHEAT CODE: Extract the exact qpos from the environment! ---
        # The bottle uses the first 7 positions (3 translation + 4 quaternion)
        bottle_qpos = self.data.qpos[0:7]
        bottle_str = " ".join([f"{x:.6f}" for x in bottle_qpos])
        
        # The robot uses the next 7 positions (the 7 arm joints)
        robot_qpos = self.data.qpos[7:14]
        robot_str = " ".join([f"{x:.6f}" for x in robot_qpos])
        
        print(f"\n--- COPY THIS EXACT KEYFRAME INTO YOUR XML FOR PHASE 2 ---")
        print(f'<key name="push_start"')
        print(f'     qpos="{bottle_str} {robot_str} 0.04 0.04"') # 0.04 0.04 keeps fingers open
        print(f'     ctrl="0 0 0 0 0 0 0 0"/>')
        print(f"----------------------------------------------------\n")

    def _print_debug(self, info):
        """Print debug information."""
        hand_pos = info["hand_pos"]
        dist = info["dist_to_target"]
        tilt = info["bottle_tilt"]
        contact = "CONTACT" if info["is_touching"] else "no contact"
        hold = info["contact_hold"]

        print(f"Step {self.episode_length}: "
              f"hand=({hand_pos[0]:.2f}, {hand_pos[1]:.2f}, {hand_pos[2]:.2f}), "
              f"dist={dist:.3f}m, tilt={tilt:.4f}, {contact}, hold={hold}")

    # ================================================================
    # Helper methods for kinematics
    # ================================================================

    def _get_ee_position(self):
        """Get end-effector position."""
        return self.data.xpos[self.hand_body_id].copy()
        # return self.data.site_xpos[self.gripper_site_id].copy()

    def _get_ee_velocity(self):
        """Get end-effector linear velocity."""
        return self.data.cvel[self.hand_body_id, 3:6].copy()

    def _get_ee_angular_velocity(self):
        """Get end-effector angular velocity."""
        return self.data.cvel[self.hand_body_id, 0:3].copy()

    def _get_ee_orientation(self):
        """Get end-effector rotation matrix."""
        quat = self.data.xquat[self.hand_body_id]
        R = np.zeros((3, 3))
        mujoco.mju_quat2Mat(R.flatten(), quat)
        return R.reshape(3, 3)

    def _get_jacobian_full(self):
        """Get full 6x7 Jacobian (position + rotation)."""
        J_pos = np.zeros((3, self.model.nv))
        J_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, J_pos, J_rot, self.hand_body_id)
        J_full = np.vstack([J_pos[:, 6:13], J_rot[:, 6:13]])
        return J_full

    def _get_bottle_tilt(self):
        """Get bottle tilt (1.0 = upright, 0.0 = fallen)."""
        quat = self.data.xquat[self.bottle_body_id]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        z_axis = R[6:9]  # Third column of rotation matrix
        return float(z_axis[2])  # Dot product with world Z

    def _compute_desired_orientation(self, push_dir_2d):
        """
        Compute desired EE orientation to face the given direction.

        The EE x-axis is aligned with push_dir_2d so that the
        pushing face of the end-effector faces toward the bottle.

        Args:
            push_dir_2d: 2D unit vector [x, y] — direction to face
        """
        x_des = np.array([push_dir_2d[0], push_dir_2d[1], 0.0])
        x_des = x_des / (np.linalg.norm(x_des) + 1e-6)

        z_des = np.array([0.0, 0.0, -1.0])  # Pointing down
        y_des = np.cross(z_des, x_des)
        y_des = y_des / (np.linalg.norm(y_des) + 1e-6)

        R_des = np.column_stack([x_des, y_des, z_des])
        return R_des

    def _compute_orientation_error(self, R_current, R_desired):
        """Compute orientation error as rotation vector."""
        R_err = R_desired @ R_current.T

        trace = np.trace(R_err)
        trace = np.clip(trace, -1.0, 3.0)
        angle = np.arccos((trace - 1.0) / 2.0)

        if abs(angle) < 1e-6:
            return np.zeros(3)

        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ])
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis = axis / axis_norm

        return axis * angle

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self.renderer.render(None)  # No trajectory to draw in reach phase

    def close(self):
        """Clean up resources."""
        self.renderer.close()