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
    v_max: float = 0.10 # was 0.05
    w_max: float = 1.0 # was 0.3
    f_max: float = 20.0 # was 20.0
    tau_rot_max: float = 4.0

    # ============================================================
    # MOTION (IMPEDANCE) CONTROLLER GAINS  --  compute_torque_from_motion
    # Starting points only -- tune on the first episode:
    #   sluggish / can't move bottle -> raise kv ;  twitchy / unstable -> lower kv
    # ============================================================
    kp_normal: float = 5.0   # height error -> normal velocity command
    kd_normal: float = 1.0   # normal-velocity damping
    kv: float = 200.0        # linear impedance gain  (N per m/s of velocity error)
    kw: float = 20.0         # angular impedance gain (N.m per rad/s of error)

    # ============================================================
    # TRAJECTORY
    # ============================================================
    goal_position: Tuple[float, float] = (0.4, -0.4)
    n_trajectory_points: int = 50
    lookahead_points: int = 4
    path_tolerance: float = 0.03

    # ============================================================
    # TIMING
    # ============================================================
    dt: float = 0.0005      # --> this was the crucial mistake
    n_substeps: int = 20
    max_episode_length: int = 3500 # was 2500
    control_dt: float = dt * n_substeps      # --> 0.0005 * 20 = 0.01 

    # ============================================================
    # TILT SAFETY
    # ============================================================
    tilt_threshold_ok: float = 0.98
    tilt_threshold_slow: float = 0.90

    # ============================================================
    # Height 
    # ============================================================
    Kz: float = 500.0
    Dz: float = 60.0
    target_z: float = 0.84

    # ============================================================
    # REWARD WEIGHTS — REBALANCED (from previous audit of reward.py)
    # ============================================================
    w_progress: float = 120.0 # was 800 was 500.0 <- very high reward and 
    w_deviation: float = 100.0 # 50.0 # 100.0 # was 30.0          # was 100; now used quadratically
    max_deviation_reward: float = 5.0   # legacy, unused
    w_stability: float = 3.0
    w_contact: float = 0.0
    w_contact_loss: float = -10.0 # -20.0
    w_force_smoothness: float = 0.01
    w_force: float = 0.05               # legacy
    target_force: float = 5.0
    w_alignment: float = 0.1 # --> 0.5            # was 0.3
    w_position: float = 15.0 # was 1.0             # was 0.5
    w_velocity_penalty: float = 100.0 # was 10.0 # was 50.0 # Weight for the speed
    v_target_limit: float = 0.08 # 8 cm/s speed limit
    w_in_contact = 0.0 # was --> 1.0 # was 1.0 # was 0.5


    time_penalty: float = 0.1 # was 0.05 #3.0 # was 3.0 # was 0.5 # was 0.005

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
    # DOMAIN RANDOMIZATION  (per-episode bottle dynamics)
    # NOTE FOR ME: task_table sliding friction is 0.2 in the XML. MuJoCo combines
    # contact friction by element-wise MAX at equal geom priority, so keep
    # fric_min ABOVE 0.2 or the table dominates and friction stops varying.
    # Set the ranges around your real bottle; nominal here is mass 0.5, fric 0.8.
    # ============================================================
    randomize_dynamics: bool = True
    mass_min: float = 0.2
    mass_max: float = 0.8
    fric_min: float = 0.3
    fric_max: float = 1.0


def get_default_config() -> EnvConfig:
    return EnvConfig()