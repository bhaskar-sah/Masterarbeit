from dataclasses import dataclass, field
import numpy as np


@dataclass
class EnvConfig:
    """
    All hyperparameters for PandaPushTrajectoryEnv.
    Change values here instead of editing the environment code.
    """

    # ==================== TRAJECTORY ====================
    goal_position: np.ndarray = field(default_factory=lambda: np.array([0.4, -0.4]))
    path_tolerance: float = 0.05          # Max deviation before off_path penalty (m)

    # ==================== PUSH PARAMETERS ====================
    base_forward_speed: float = 0.015     # Base push speed (m/s)
    behind_distance: float = 0.04         # Distance hand stays behind bottle (m)

    # ==================== LOOKAHEAD ====================
    lookahead_points: int = 4             # Points ahead on trajectory to aim for

    # ==================== WRIST ROTATION ====================
    gripper_push_angle_at_home: float = -np.pi / 2   # -90 deg home wrist angle
    max_wrist_rotation: float = 2.5       # Max wrist joint offset (rad)

    # ==================== STIFFNESS (What RL learns!) ====================
    K_min: float = 100.0                  # Minimum Cartesian stiffness (N/m)
    K_max: float = 500.0                  # Maximum Cartesian stiffness (N/m)

    # ==================== TILT SAFETY ====================
    tilt_ok: float = 0.98 # 0.97 # was 0.99                 # Full speed above this tilt
    tilt_slow: float = 0.97 # 0.95 # was 0.98               # Half speed below this tilt
    tilt_stop: float = 0.95 # 0.93 # was 0.96              # Trigger settle mode below this tilt
    settle_required: int = 50 # 25             # Steps needed to confirm stability

    # ==================== CARTESIAN CONTROL ====================
    target_z: float = 0.92               # Desired hand height (m)
    z_gain: float = 10.0                  # Proportional gain for Z control
    damping: float = 0.01                 # Jacobian damping for IK

    # ==================== EPISODE ====================
    max_episode_length: int = 2500        # Steps before truncation

    # ==================== OBSERVATION ====================
    obs_dim: int = 39                     # Observation vector dimension
