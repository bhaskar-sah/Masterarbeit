import numpy as np

from .config import (
    TARGET_Z,
    PATH_TOLERANCE,
    W_PROGRESS,
    W_DEVIATION,
    W_ANGLE,
    W_TILT,
    CONTACT_BONUS,
    NO_CONTACT_PENALTY,
    TIME_PENALTY,
    APPROACH_DIST_WEIGHT,
    APPROACH_Z_WEIGHT,
    APPROACH_CONTACT_BONUS,
    SUCCESS_BONUS,
    BOTTLE_FALL_PENALTY,
    OFF_PATH_PENALTY,
)
from .geometry import get_path_deviation, get_progress
from .contact import get_contact_force, is_touching
from .control import get_bottle_tilt, get_angle_error


def compute_reward(env):
    """
    Smooth interpretable reward:

        r = w_p * progress_delta
            - w_d * deviation
            - w_a * angle_error
            - w_tilt * (1 - tilt)
            + contact_reward
            - time_penalty

    Approach phase uses a simpler shaping reward.
    Terminal rewards/penalties remain discrete.
    """
    info = {"is_success": False}

    bottle_xy = env.data.xpos[env.bottle_body_id][:2]
    hand_pos = env.data.xpos[env.hand_body_id]

    _, deviation_mag = get_path_deviation(bottle_xy, env.trajectory)
    progress = get_progress(bottle_xy, env.trajectory, env.total_arc_length)
    tilt = get_bottle_tilt(env)

    touching = is_touching(
        env.model, env.data, env.bottle_body_id, env.robot_contact_bodies
    )

    force = get_contact_force(
        env.model, env.data, env.bottle_body_id, env.robot_contact_bodies
    )
    force_mag = float(np.linalg.norm(force))

    dist_to_bottle = float(np.linalg.norm(hand_pos[:2] - bottle_xy))
    angle_error = get_angle_error(env)

    # ========================================================
    # APPROACH PHASE REWARD
    # ========================================================
    if env.in_approach:
        total_reward = (
            -APPROACH_DIST_WEIGHT * dist_to_bottle
            -APPROACH_Z_WEIGHT * abs(hand_pos[2] - TARGET_Z)
        )

        if touching or dist_to_bottle < 0.06:
            total_reward += APPROACH_CONTACT_BONUS
            env.in_approach = False

        if env.episode_length % 50 == 0:
            print(f"Step {env.episode_length} [APPROACH]: dist={dist_to_bottle:.3f}m")

    # ========================================================
    # PUSH PHASE REWARD
    # ========================================================
    else:
        progress_delta = max(progress - env.prev_progress, 0.0)

        r_progress = W_PROGRESS * progress_delta
        r_deviation = -W_DEVIATION * deviation_mag
        r_angle = -W_ANGLE * angle_error
        r_tilt = -W_TILT * (1.0 - tilt)

        if touching:
            r_contact = CONTACT_BONUS
        else:
            r_contact = -NO_CONTACT_PENALTY

        total_reward = (
            r_progress
            + r_deviation
            + r_angle
            + r_tilt
            + r_contact
            - TIME_PENALTY
        )

        if env.episode_length % 50 == 0:
            K_avg = float(np.mean(env.current_K))
            mode = "SETTLE" if env.is_settling else "PUSH"
            contact_status = "CONTACT" if touching else "NO CONTACT!"
            print(
                f"Step {env.episode_length} [{mode}]: "
                f"prog={progress:.1%}, dev={deviation_mag * 100:.1f}cm, "
                f"θ={np.degrees(angle_error):.1f}°, K={K_avg:.0f}, F={force_mag:.1f}N, "
                f"tilt={tilt:.4f}, {contact_status}"
            )

    # ========================================================
    # TERMINAL REWARDS / PENALTIES
    # ========================================================
    if progress > 0.95 and deviation_mag < PATH_TOLERANCE:
        total_reward += SUCCESS_BONUS
        info["is_success"] = True
        print(f"SUCCESS at step {env.episode_length}!")

    if tilt < 0.5:
        total_reward -= BOTTLE_FALL_PENALTY
        info["bottle_fallen"] = True

    if deviation_mag > 0.15:
        total_reward -= OFF_PATH_PENALTY
        info["off_path"] = True

    env.prev_progress = progress

    # ========================================================
    # LOGGING INFO
    # ========================================================
    info["progress"] = progress
    info["deviation"] = deviation_mag
    info["angle_error"] = angle_error
    info["force_magnitude"] = force_mag
    info["bottle_tilt"] = tilt
    info["stiffness"] = float(np.mean(env.current_K))

    return float(total_reward), info