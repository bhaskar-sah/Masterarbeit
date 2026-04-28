# env.py
"""
Panda Push Trajectory Environment with Learned Force Profile (Pure RL).

This environment implements Marko's force-velocity control approach where:
    - RL learns: velocity (vx, vy, vz) and force magnitude (f)
    - Controller: Converts these to joint torques via J^T × F_cmd + τ_gravity

Action space: [vx, vy, vz, f] - all in [-1, 1]
    - vx, vy, vz: Desired end-effector velocity (scaled to ±v_max)
    - f: Push force magnitude (scaled to [0, f_max])

The robot pushes a bottle along a predefined trajectory using
direct torque control with a PD + Force controller and gravity compensation.
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
from debug_utils import DebugPrinter, StepLogger

from config import EnvConfig, get_default_config
from trajectory import TrajectoryManager
from contact import ContactManager
from push_controller import PushController
from observation import ObservationBuilder
from reward import RewardComputer
from logger import EpisodeLogger
from renderer import TrajectoryRenderer


class PandaPushTrajectoryEnv(gym.Env):
    """
    Gymnasium environment for pushing a bottle along a trajectory.
    
    Uses torque control with learned velocity and force commands.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_type="straight", config=None):
        """
        Initialize the environment.

        Args:
            render_mode: "human" for visualization, None for training
            trajectory_type: "straight", "curved", or "s_curve"
            config: EnvConfig (optional, uses default if None)
        """
        super().__init__()

        # Configuration
        self.config = config if config is not None else get_default_config()
        self.trajectory_type = trajectory_type
        self.render_mode = render_mode

        # Load MuJoCo model
        self._load_model()

        # Get body IDs
        self._get_body_ids()

        # Initialize managers
        self._init_managers()

        # Action space: [vx, vy, wz, f]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
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

        self.debug_printer = DebugPrinter(self.config, print_every=50)
        self.step_logger = StepLogger("logs/training_run.csv")
        self.episode_num = 0
        self.total_steps = 0

    def _load_model(self):
        """Load MuJoCo model from XML."""
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "robot_panda_push_force.xml")
        xml_path = os.path.abspath(xml_path)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find XML file at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def _get_body_ids(self):
        """Get MuJoCo body IDs."""
        self.push_start_key_id = self.model.key("push_start").id
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

    def _init_managers(self):
        """Initialize all manager objects."""
        # Trajectory manager
        self.traj_manager = TrajectoryManager(
            goal_position=np.array(self.config.goal_position),
            path_tolerance=self.config.path_tolerance,
            lookahead_points=self.config.lookahead_points
        )

        # Contact manager
        self.contact_manager = ContactManager(
            self.model,
            self.data,
            self.robot_contact_bodies,
            self.bottle_body_id
        )

        # Push controller
        self.push_controller = PushController(
            self.model,
            self.data,
            self.traj_manager,
            self.contact_manager,
            self.hand_body_id,
            self.bottle_body_id,
            self.config
        )

        # Observation builder
        self.obs_builder = ObservationBuilder(
            self.model,
            self.data,
            self.config,
            self.hand_body_id,
            self.bottle_body_id,
            self.traj_manager,
            self.contact_manager,
            self.push_controller
        )

        # Reward computer
        self.reward_computer = RewardComputer(
            self.config,
            self.traj_manager,
            self.contact_manager,
            self.bottle_body_id,
            self.hand_body_id,
            self.data
        )

        # Episode logger
        self.logger = EpisodeLogger()
        self.push_controller.logger = self.logger

        # Renderer
        self.renderer = TrajectoryRenderer(self.model, self.data, self.render_mode)

    def _print_init_info(self):
        """Print initialization information."""
        print(f"\n{'=' * 60}")
        print("FORCE-VELOCITY CONTROL")
        print(f"{'=' * 60}")
        print(f"Action space: [vx, vy, vz, f]")
        print(f"Control: τ = J^T × (Kp·Δp + Kd·Δv + Kf·ΔF) + τ_gravity")
        print(f"Gains: Kp={self.config.Kp}, Kd={self.config.Kd}, Kf={self.config.Kf}")
        print(f"Limits: v_max={self.config.v_max} m/s, f_max={self.config.f_max} N")
        print(f"{'=' * 60}")

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed
            options: Optional dict with 'trajectory_type'

        Returns:
            observation: Initial observation
            info: Empty dict
        """
        super().reset(seed=seed)

        # Reset MuJoCo to push_start keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.push_start_key_id)
        mujoco.mj_forward(self.model, self.data)

        # Let simulation settle with gravity compensation
        # This is important for torque control!
        for _ in range(200):
            # Apply gravity compensation to hold position
            tau_gravity = self.data.qfrc_bias[6:13].copy()
            self.data.ctrl[:7] = tau_gravity
            mujoco.mj_step(self.model, self.data)

        # Get bottle start position
        bottle_start = self.data.xpos[self.bottle_body_id].copy()

        # Determine trajectory type
        traj_type = options["trajectory_type"] if options and "trajectory_type" in options else self.trajectory_type

        # Generate trajectory
        self.traj_manager.generate_trajectory(bottle_start[:2], traj_type)

        # Reset controller with current EE position
        ee_pos = self.data.xpos[self.hand_body_id].copy()
        self.push_controller.reset(ee_pos)

        # Reset reward computer
        self.reward_computer.reset()

        # Reset logger
        self.logger.reset()

        # Reset episode state
        self.episode_length = 0

        # Print episode info
        print(f"\n{'=' * 50}")
        print(f"EPISODE: {traj_type}")
        print(f"Bottle start: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"Hand start: ({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f})")
        print(f"Goal: ({self.config.goal_position[0]:.2f}, {self.config.goal_position[1]:.2f})")
        print(f"{'=' * 50}")

        self.episode_num += 1

        return self.obs_builder.get_observation(), {}

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: RL action [vx, vy, wz, f] in [-1, 1]

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode was cut short (time limit)
            info: Additional information
        """
        tau = self.push_controller.compute_torque(action)
        self.data.ctrl[:7] = tau

        for _ in range(self.config.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self.obs_builder.get_observation()
        
        # Pass pure push logic to reward computer
        reward, info = self.reward_computer.compute_reward()

        # Get positions and force for logging
        hand_pos = self.data.xpos[self.hand_body_id].copy()
        bottle_pos = self.data.xpos[self.bottle_body_id].copy()
        force = self.contact_manager.get_contact_force()

        # Enhanced debug print
        if self.episode_length % 5 == 0:
            self.debug_printer.print_step(
                self.episode_length, hand_pos, bottle_pos, action, reward, info,
                self.push_controller.last_push_dir,
                self.push_controller.p_des,
                self.push_controller.last_F_cmd,
                self.push_controller.last_F_des,
                self.push_controller.last_f_magnitude
            )

            # Log to CSV
            self.step_logger.log(
                self.total_steps, self.episode_length, self.episode_num,
                hand_pos, bottle_pos, self.push_controller.p_des,
                action, self.config,
                self.push_controller.last_push_dir, 
                self.push_controller.last_F_des,
                self.push_controller.last_F_cmd, 
                force,
                reward, info
            )

        self.logger.log_step(
            force=self.contact_manager.get_contact_force(),
            deviation=info.get("deviation", 0.0),
            velocity=np.array([action[0]*self.config.v_max, action[1]*self.config.v_max, 0]),
            action=action,
            tilt=info.get("bottle_tilt", 1.0)
        )

        # if self.episode_length % 50 == 0:
        #     self._print_debug_info(info, action, reward)


        self.episode_length += 1
        self.total_steps += 1
        terminated = bool(
            info.get("is_success") or 
            info.get("bottle_fallen") or 
            info.get("off_path") or
            info.get("fled")
        )
        truncated = bool(self.episode_length >= self.config.max_episode_length)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    # def _print_debug_info(self, info, action, reward):
    #     """Print detailed debug information every N steps."""
    #     progress = info.get("progress", 0.0)
    #     deviation = info.get("deviation", 0.0)
    #     tilt = info.get("bottle_tilt", 1.0)
    #     force_mag = info.get("force_magnitude", 0.0)
    #     is_touching = info.get("is_touching", False)
        
    #     # Get actual coordinates
    #     hand_pos = self.data.xpos[self.hand_body_id]
    #     bottle_pos = self.data.xpos[self.bottle_body_id]
        
    #     contact_str = "CONTACT" if is_touching else "NO CONTACT"
        
    #     print(f"Step {self.episode_length:4d} | {contact_str} | Step Reward: {reward:6.2f}")
    #     print(f"  Hand Pos:   ({hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f})")
    #     print(f"  Bottle Pos: ({bottle_pos[0]:.3f}, {bottle_pos[1]:.3f}, {bottle_pos[2]:.3f})")
    #     # Action is [vx, vy, wz, f]
    #     print(f"  NN Action:  vx={action[0]:5.2f}, vy={action[1]:5.2f}, wz={action[2]:5.2f}, f={action[3]:5.2f}")
    #     print(f"  Status:     Prog={progress*100:5.1f}%, Dev={deviation*100:4.1f}cm, Tilt={tilt:.4f}, Force={force_mag:4.1f}N")
    #     print("-" * 65)

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self.renderer.render(self.traj_manager.trajectory)

    def close(self):
        """Clean up resources."""
        self.renderer.close()
        self.step_logger.close()

    def get_episode_summary(self) -> dict:
        """Get summary of the episode for logging."""
        return self.logger.get_summary()