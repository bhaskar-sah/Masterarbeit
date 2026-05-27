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
    # Kp: float = 150.0 # was 100.0
    # Kd: float = 100.0 # was 20.0 # maybe try with 100
    # Kf: float = 0.0 # was 0.5 # Force feedback gain: kept for backward compat, unused in pure motion control

    Kp_rot: float = 10.0
    Kd_rot: float = 15.0 # was 2.0

    # Cap on p_error magnitude (m). Prevents position term from dominating
    # F_cmd when p_des races ahead of actual EE.
    # p_error_max = 0.03 # was 0.03 # 0.03  # 3 cm

    # ============================================================
    # ACTION LIMITS
    # ============================================================
    v_max: float = 0.10 # was 0.05
    w_max: float = 1.0 # was 0.3
    f_max: float = 20.0 # was 20.0
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
    dt: float = 0.0005      # --> this was the crucial mistake
    n_substeps: int = 20
    max_episode_length: int = 2500
    control_dt: float = dt * n_substeps      # --> 0.0005 * 20 = 0.01 

    # ============================================================
    # TILT SAFETY
    # ============================================================
    tilt_threshold_ok: float = 0.98
    tilt_threshold_slow: float = 0.90

    # ============================================================
    # REWARD WEIGHTS — REBALANCED (from previous audit of reward.py)
    # ============================================================
    w_progress: float = 800.0 # was 500.0
    w_deviation: float = 100.0 # 50.0 # 100.0 # was 30.0          # was 100; now used quadratically
    max_deviation_reward: float = 5.0   # legacy, unused
    w_stability: float = 3.0
    w_contact: float = 0.0
    w_contact_loss: float = -10.0
    w_force_smoothness: float = 0.01
    w_force: float = 0.05               # legacy
    target_force: float = 5.0
    w_alignment: float = 0.5            # was 0.3
    w_position: float = 15.0 # was 1.0             # was 0.5
    w_velocity_penalty: float = 100.0 # was 10.0 # was 50.0 # Weight for the speed
    v_target_limit: float = 0.08 # 8 cm/s speed limit

    # NEW: Action Smoothing Penalties
    w_action_rate: float = 2.0    # Punishes jittery, oscillating commands (Bang-Bang fix)
    w_action_mag: float = 0.5     # Punishes using maximum force lazily
    w_action_perp: float = 1.0           # NEW — penalize lateral wasted force
    action_mag_buffer: float = 3.0       # NEW — N allowed over measured along-force

    time_penalty: float = 0.1 # was 0.05 #3.0 # was 3.0 # was 0.5 # was 0.005

    success_bonus: float = 100.0
    failure_penalty: float = -100.0 # was -10.0 # was -100.0
    off_path_penalty: float = -10.0 # was -50.0 # was -10.0     # was -50

    # ============================================================
    # OBSERVATION SPACE
    # ============================================================
    # Layout: see observation.py for full index reference
    # 7 + 7 + 3 + 3 + 3 + 2 + 2 + 1 + 2 + 1 + 2 + 1 + 1 + 3 + 1 + 1 + 3 + 2 + 1 + 1 + 1 + 2 = 50
    obs_dim: int = 50

    # ============================================================
    # ACTION SPACE: [Fx_ee, Fy_ee, wz]
    # ============================================================
    action_dim: int = 3

    # ============================================================
    # ACTION MODE
    # ============================================================
    # action_mode: str = "force"   # "velocity" or "force"


def get_default_config() -> EnvConfig:
    return EnvConfig()