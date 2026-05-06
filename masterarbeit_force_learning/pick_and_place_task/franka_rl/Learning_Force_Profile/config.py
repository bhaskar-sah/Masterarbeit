"""
config.py

CHANGES from previous version:
  - obs_dim: 45 -> 48 (added cos_force_alignment + future_push_dir)
  - All other reward weights and gains kept the same
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvConfig:
    # ============================================================
    # CONTROLLER GAINS
    # ============================================================
    Kp: float = 150.0 # was 100.0
    Kd: float = 30.0 # was 20.0
    Kf: float = 0.0 # was 0.5 # Force feedback gain: kept for backward compat, unused in pure motion control

    Kp_rot: float = 10.0
    Kd_rot: float = 2.0

    # Cap on p_error magnitude (m). Prevents position term from dominating
    # F_cmd when p_des races ahead of actual EE.
    p_error_max = 0.03 # was 0.03 # 0.03  # 3 cm

    # ============================================================
    # ACTION LIMITS
    # ============================================================
    v_max: float = 0.05
    w_max: float = 2.0
    # f_max: float = 20.0
    # F_FLOOR: float = 0.5 # was 2.0

    # ============================================================
    # TRAJECTORY
    # ============================================================
    goal_position: Tuple[float, float] = (0.4, -0.4)
    n_trajectory_points: int = 50
    lookahead_points: int = 4
    path_tolerance: float = 0.05

    # ============================================================
    # TIMING
    # ============================================================
    dt: float = 0.002
    n_substeps: int = 20
    max_episode_length: int = 2500
    control_dt: float = 0.04

    # ============================================================
    # TILT SAFETY
    # ============================================================
    tilt_threshold_ok: float = 0.98
    tilt_threshold_slow: float = 0.90

    # ============================================================
    # REWARD WEIGHTS — REBALANCED (from previous audit of reward.py)
    # ============================================================
    w_progress: float = 500.0
    w_deviation: float = 30.0           # was 100; now used quadratically
    max_deviation_reward: float = 5.0   # legacy, unused
    w_stability: float = 3.0
    w_contact: float = 0.0
    w_contact_loss: float = -10.0
    w_force_smoothness: float = 0.01
    w_force: float = 0.05               # legacy
    target_force: float = 5.0
    w_alignment: float = 0.5            # was 0.3
    w_position: float = 15.0 # was 1.0             # was 0.5

    time_penalty: float = 3.0 # was 0.5 # was 0.005

    success_bonus: float = 100.0
    failure_penalty: float = -10.0 # was -100.0
    off_path_penalty: float = -10.0     # was -50

    # ============================================================
    # OBSERVATION SPACE
    # ============================================================
    # Layout: see observation.py for full index reference
    # 7 + 7 + 3 + 3 + 3 + 2 + 1 + 2 + 1 + 2 + 1 + 1 + 3 + 1 + 1 + 3 + 2 + 1 + 1 + 1 + 2 = 48
    obs_dim: int = 48

    # ============================================================
    # ACTION SPACE: [vx, vy, wz]
    # ============================================================
    action_dim: int = 3


def get_default_config() -> EnvConfig:
    return EnvConfig()