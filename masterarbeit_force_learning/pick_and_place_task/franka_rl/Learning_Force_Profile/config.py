"""
config.py
 
Frame convention: the action and the directional observations use the path
frame {P} = [t_hat, b_hat, n_hat], built from the base frame {B}.
See push_controller.py (compute_path_frame_base) and observation.py.
 
  obs_dim    = 48
  action_dim = 3   ->  [Ft, Fb, Tz]  (path-frame planar force + yaw torque)
"""


import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvConfig:
    # ============================================================
    # CONTROLLER GAINS
    # ============================================================
    Kp_rot: float = 10.0
    Kd_rot: float = 15.0 # was 2.0

    # ============================================================
    # ACTION LIMITS
    # ============================================================
    v_max: float = 0.3 # was 0.05
    w_max: float = 1.0 # was 0.3
    f_max: float = 20.0 # was 20.0
    tau_rot_max: float = 4.0

    # ============================================================
    # TRAJECTORY
    # ============================================================
    goal_position: Tuple[float, float] = (0.4, -0.4)
    n_trajectory_points: int = 50
    lookahead_points: int = 4
    path_tolerance: float = 0.01

    # ============================================================
    # TIMING
    # ============================================================
    dt: float = 0.0005      # --> this was the crucial mistake
    n_substeps: int = 20
    max_episode_length: int = 2500 # was 2500
    control_dt: float = dt * n_substeps      # --> 0.0005 * 20 = 0.01 

    # ============================================================
    # TILT SAFETY
    # ============================================================
    tilt_threshold_ok: float = 0.98
    tilt_threshold_slow: float = 0.90

    # ============================================================
    # Height Wrench Based Controller
    # ============================================================
    Kfz: float = 500.0
    Dfz: float = 60.0

    # ============================================================
    # Motion Based Controller
    # ============================================================
    # Height controller
    Kvz: float = 5.0
    Dvz: float = 0.6

    # Anisotropic path gains for linear velocity
    kv_t: float = 80.0
    kv_b: float = 120.0
    kv_n: float = 200.0

    # Gain for angular velocity
    Dw: float= 10.0

    # ============================================================
    # REWARD WEIGHTS — REBALANCED (from previous audit of reward.py)
    # ============================================================
    w_progress: float = 100.0 # was 800 was 500.0 <- very high reward and
    w_deviation: float = 200.0 # 50.0 # 100.0 # was 30.0          # was 100; now used quadratically
    max_deviation_reward: float = 5.0   # legacy, unused
    w_stability: float = 3.0
    w_contact: float = 0.0
    w_contact_loss: float = -10.0 # -20.0
    w_force_smoothness: float = 0.01
    w_force: float = 0.05               # legacy
    target_force: float = 5.0
    w_alignment: float = 0.1            # was 0.3
    w_position: float = 15.0 # was 1.0             # was 0.5
    w_velocity_penalty: float = 10.0 # was 10.0 # was 50.0 # Weight for the speed
    v_target_limit: float = 0.2 # 20 cm/s speed limit
    v_minimum: float = 0.02 #2cm/s
    w_in_contact = 0.1 # was 1.0 # was 0.5


    time_penalty: float = 0.2 # was 0.05 #3.0 # was 3.0 # was 0.5 # was 0.005

    success_bonus: float = 100.0
    failure_penalty: float = -100.0 # was -10.0 # was -100.0
    off_path_penalty: float = -10.0 # was -50.0 # was -10.0     # was -50

    # ============================================================
    # OBSERVATION SPACE
    # ============================================================
    # Layout: see observation.py for full index reference
    # 7 + 7 + 3 + 3 + 3 + 2  + 1 + 2 + 1 + 2 + 1 + 1 + 3 + 1 + 1 + 3 + 2 + 1 + 1 + 1 + 2 = 48
    obs_dim: int = 48

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