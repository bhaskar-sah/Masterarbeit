import numpy as np

from .config import OBS_DIM, K_MIN, K_MAX, GOAL_POSITION
from .geometry import get_path_tangent, get_path_deviation, get_progress
from .contact import get_contact_force, is_touching
from .control import get_bottle_tilt, get_angle_error


def get_observation(env):
    """
    Build the full observation vector.

    Observation layout:
        qpos(7),
        qvel(7),
        hand_pos(3),
        bottle_pos(3),
        dir_to_bottle(2),
        dist_to_bottle(1),
        tangent_dir(2),
        deviation_vec(2),
        deviation_mag(1),
        progress(1),
        contact_force(3),
        is_touching(1),
        K_normalized(2),
        settling_flag(1),
        tilt(1),
        angle_error(1),
        dist_to_goal(1)

    Total = 39
    """
    qpos = env.data.qpos[7:14].astype(np.float32)
    qvel = env.data.qvel[6:13].astype(np.float32)

    hand_pos = env.data.xpos[env.hand_body_id].astype(np.float32)
    bottle_pos = env.data.xpos[env.bottle_body_id].astype(np.float32)
    bottle_xy = bottle_pos[:2]

    # --------------------------------------------------------
    # Hand-to-bottle relation
    # --------------------------------------------------------
    hand_to_bottle = bottle_pos[:2] - hand_pos[:2]
    dist_to_bottle = np.linalg.norm(hand_to_bottle)
    dir_to_bottle = hand_to_bottle / (dist_to_bottle + 1e-6)

    # --------------------------------------------------------
    # Path-related quantities
    # --------------------------------------------------------
    tangent_dir = get_path_tangent(bottle_xy, env.trajectory)
    deviation_vec, deviation_mag = get_path_deviation(bottle_xy, env.trajectory)
    progress = get_progress(bottle_xy, env.trajectory, env.total_arc_length)

    # --------------------------------------------------------
    # Contact-related quantities
    # --------------------------------------------------------
    contact_force = get_contact_force(
        env.model, env.data, env.bottle_body_id, env.robot_contact_bodies
    )

    touching = is_touching(
        env.model, env.data, env.bottle_body_id, env.robot_contact_bodies
    )
    is_touching_arr = np.array([1.0 if touching else 0.0], dtype=np.float32)

    # --------------------------------------------------------
    # Stiffness / internal state
    # --------------------------------------------------------
    K_normalized = (env.current_K - K_MIN) / (K_MAX - K_MIN)
    K_normalized = K_normalized.astype(np.float32)

    settling_flag = np.array(
        [1.0 if env.is_settling else 0.0], dtype=np.float32
    )

    # --------------------------------------------------------
    # Stability / task metrics
    # --------------------------------------------------------
    tilt = np.array([get_bottle_tilt(env)], dtype=np.float32)
    angle_error = np.array([get_angle_error(env)], dtype=np.float32)

    dist_to_goal = np.array(
        [np.linalg.norm(GOAL_POSITION - bottle_xy)],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Concatenate everything
    # --------------------------------------------------------
    obs = np.concatenate([
        qpos,                      # 7
        qvel,                      # 7
        hand_pos,                  # 3
        bottle_pos,                # 3
        dir_to_bottle.astype(np.float32),   # 2
        np.array([dist_to_bottle], dtype=np.float32),  # 1
        tangent_dir.astype(np.float32),      # 2
        deviation_vec.astype(np.float32),    # 2
        np.array([deviation_mag], dtype=np.float32),   # 1
        np.array([progress], dtype=np.float32),        # 1
        contact_force.astype(np.float32),    # 3
        is_touching_arr,            # 1
        K_normalized,               # 2
        settling_flag,              # 1
        tilt,                       # 1
        angle_error,                # 1
        dist_to_goal                # 1
    ]).astype(np.float32)

    if obs.shape != (OBS_DIM,):
        raise ValueError(
            f"Observation has wrong shape: {obs.shape}, expected ({OBS_DIM},)"
        )

    return obs