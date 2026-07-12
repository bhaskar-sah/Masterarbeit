"""
Configuration for Reach Phase Environment.

Phase 1 of the two-phase approach:
- Goal: Learn to reach from home position to contact with bottle
- Action: 3D velocity [vx, vy, vz]
- Terminates: When stable contact is achieved

The post-condition (EE in contact with bottle at correct height/angle)
becomes the pre-condition for the Push phase.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class ReachConfig:
    """Configuration dataclass for the reach environment."""

    # ==================================================================
    #                     CONTROLLER GAINS
    # ==================================================================
    # Position control gains for torque control
    Kp: float = 150.0           # Position stiffness (N/m)
    Kd: float = 25.0            # Position damping (Ns/m)

    # Orientation control gains
    Kp_rot: float = 15.0        # Orientation stiffness (Nm/rad)
    Kd_rot: float = 3.0         # Orientation damping (Nms/rad)

    # ==================================================================
    #                     ACTION LIMITS
    # ==================================================================
    # Velocity limits (m/s) - RL outputs in [-1, 1], scaled to [-v_max, v_max]
    v_max: float = 0.08         # Slightly slower for precise reaching

    # ==================================================================
    #                     GEOMETRY
    # ==================================================================
    # Target height for the HAND BODY ORIGIN (m)
    #
    # The finger contact surface is ~0.10m below the hand body origin:
    #   - Fingers offset from hand: 0.0584m (in hand local z)
    #   - Fingertip pad center:     0.0445m (in finger local z)
    #   - Total offset:             ~0.10m
    #
    # Bottle collision cylinder center: z = 0.805 + 0.06 = 0.865
    # For fingers to contact at bottle center height:
    #   target_z = 0.865 + 0.10 = 0.965
    #
    # Note: xpos[hand_body_id] reports this origin, NOT the fingertip.
    target_z: float = 0.96

    # Bottle collision radius (m) — from XML: size="0.039 0.05"
    # Used to compute the contact surface position on the bottle skin
    bottle_radius: float = 0.039

    # How far behind the contact surface the EE target is placed (m)
    # Small offset so the EE presses slightly into the bottle for
    # stable contact rather than barely touching the surface
    behind_distance: float = 0.0

    # Contact threshold distance (m) - when to consider "reached"
    contact_distance_threshold: float = 0.01

    # ==================================================================
    #                     TIMING
    # ==================================================================
    # Simulation timestep (s) - should match XML
    dt: float = 0.002

    # Number of simulation steps per control step
    n_substeps: int = 20

    # Maximum episode length (control steps)
    max_episode_length: int = 500  # Shorter episodes for reach task

    # ==================================================================
    #                     CONTACT VERIFICATION
    # ==================================================================
    # Number of steps contact must be maintained to consider success
    contact_hold_steps: int = 10

    # Maximum tilt allowed during approach (prevent knocking bottle)
    max_tilt_during_approach: float = 0.95  # cos(18°) ≈ 0.95

    # ==================================================================
    #                     REWARD WEIGHTS
    # ==================================================================
    # Distance reward (negative distance to target)
    w_distance: float = 10.0

    # Velocity toward target reward (when far from bottle)
    w_approach_velocity: float = 5.0

    # Velocity penalty (when close to bottle, penalize excess speed
    # to encourage gentle contact and prevent knocking)
    w_velocity_penalty: float = 5.0

    # Height alignment reward
    w_height: float = 20.0

    # Orientation alignment reward
    w_orientation: float = 2.0

    # Contact reward (big bonus for achieving contact)
    contact_bonus: float = 50.0

    # Stable contact bonus (for maintaining contact)
    stable_contact_bonus: float = 100.0

    # Penalty for hitting bottle too hard (causing tilt)
    w_tilt_penalty: float = 20.0

    # Time penalty (encourage efficiency)
    time_penalty: float = 0.01

    # ==================================================================
    #                     OBSERVATION SPACE
    # ==================================================================
    # Observation for reach phase:
    # qpos(7) + qvel(7) + hand_pos(3) + hand_vel(3) +
    # bottle_pos(3) + dir_to_target(3) + dist_to_target(1) +
    # bottle_tilt(1) + is_touching(1) + ee_orientation(3)
    # Total: 7+7+3+3+3+3+1+1+1+3 = 32
    obs_dim: int = 32

    # ==================================================================
    #                     ACTION SPACE
    # ==================================================================
    # Action dimension: [vx, vy, vz] - 3D velocity
    action_dim: int = 3


def get_reach_config() -> ReachConfig:
    """Return default reach configuration."""
    return ReachConfig()