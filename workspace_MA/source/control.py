import numpy as np
import mujoco

from .config import (
    BASE_FORWARD_SPEED,
    BEHIND_DISTANCE,
    LATERAL_PUSH_SCALE,
    LATERAL_HAND_OFFSET_SCALE,
    TARGET_Z,
    Z_GAIN,
    DAMPING,
    K_MIN,
    K_MAX,
    TILT_OK,
    TILT_SLOW,
    TILT_STOP,
)
from .geometry import get_path_tangent, get_path_deviation
from .contact import get_contact_force, is_touching


# ============================================================
# BASIC BOTTLE / STABILITY FUNCTIONS
# ============================================================

def get_bottle_tilt(env):
    """
    Returns the z-axis alignment of the bottle body.
    Closer to 1.0 means upright.
    """
    bottle_mat = env.data.xmat[env.bottle_body_id].reshape(3, 3)
    return float(bottle_mat[2, 2])


def check_stable(env):
    """
    Check whether bottle is sufficiently upright and not rotating too much.
    """
    tilt = get_bottle_tilt(env)

    # Keeping your original logic here
    angvel = np.linalg.norm(env.data.qvel[3:6])

    return bool(tilt > 0.97 and angvel < 0.1)


# ============================================================
# GEOMETRY-BASED PUSHING HELPERS
# ============================================================

def get_rl_push_direction(env, bottle_xy, lateral_mod):
    """
    Nominal push direction = path tangent.
    RL adds lateral correction toward the path using deviation direction.
    """
    tangent_dir = get_path_tangent(bottle_xy, env.trajectory)
    deviation_vec, deviation_mag = get_path_deviation(bottle_xy, env.trajectory)

    if deviation_mag > 1e-6:
        correction_dir = deviation_vec / deviation_mag  # bottle -> path
    else:
        correction_dir = np.zeros(2, dtype=np.float32)

    raw_dir = tangent_dir + LATERAL_PUSH_SCALE * float(lateral_mod) * correction_dir
    norm = np.linalg.norm(raw_dir)

    if norm > 1e-6:
        return (raw_dir / norm).astype(np.float32)

    return tangent_dir.astype(np.float32)


def get_hand_target_position(env, bottle_xy, lateral_mod):
    """
    Nominal hand target:
    - behind bottle along path tangent
    - RL adds lateral hand offset toward/away from path correction direction
    """
    tangent_dir = get_path_tangent(bottle_xy, env.trajectory)
    deviation_vec, deviation_mag = get_path_deviation(bottle_xy, env.trajectory)

    hand_xy = bottle_xy - tangent_dir * BEHIND_DISTANCE

    if deviation_mag > 1e-6:
        correction_dir = deviation_vec / deviation_mag
    else:
        correction_dir = np.zeros(2, dtype=np.float32)

    hand_xy = hand_xy + float(lateral_mod) * LATERAL_HAND_OFFSET_SCALE * correction_dir

    return np.array([hand_xy[0], hand_xy[1], TARGET_Z], dtype=np.float32)


def get_angle_error(env):
    """
    Angle between actual contact force direction and desired push direction.
    """
    contact_force = get_contact_force(
        env.model, env.data, env.bottle_body_id, env.robot_contact_bodies
    )
    force_mag = np.linalg.norm(contact_force[:2])

    if force_mag > 0.5:
        force_dir = -contact_force[:2] / force_mag
        desired_dir = env.last_desired_push_dir
        dot_product = np.clip(np.dot(force_dir, desired_dir), -1.0, 1.0)
        angle_error = np.arccos(dot_product)
    else:
        angle_error = 0.0

    return float(angle_error)


# ============================================================
# CARTESIAN / JOINT VELOCITY MAPPING
# ============================================================

def cartesian_to_joint_velocity(env, cart_vel):
    """
    Convert desired Cartesian hand velocity to joint velocity using damped pseudoinverse.
    """
    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))

    mujoco.mj_jacBody(env.model, env.data, jacp, jacr, env.hand_body_id)

    # Keep same indexing as your original implementation
    J = jacp[:, 6:13]

    JJT = J @ J.T
    J_pinv = J.T @ np.linalg.inv(JJT + DAMPING ** 2 * np.eye(3))

    q_dot = J_pinv @ cart_vel
    return q_dot.astype(np.float32)


# ============================================================
# APPROACH PHASE
# ============================================================

def compute_approach_velocity(env):
    """
    Move hand behind the bottle before starting the push phase.
    """
    hand_pos = env.data.xpos[env.hand_body_id]
    bottle_xy = env.data.xpos[env.bottle_body_id][:2]

    target_pos = get_hand_target_position(env, bottle_xy, lateral_mod=0.0)
    error = target_pos - hand_pos
    dist = np.linalg.norm(error[:2])

    v_desired = error * 2.0

    v_mag = np.linalg.norm(v_desired[:2])
    if v_mag > 0.04:
        v_desired[:2] = v_desired[:2] / v_mag * 0.04

    v_desired[2] = Z_GAIN * (TARGET_Z - hand_pos[2])

    return v_desired.astype(np.float32), float(dist)


# ============================================================
# PUSH PHASE
# ============================================================

def compute_push_velocity(env, action):
    """
    Main RL-guided pushing logic.
    Action = [forward_mod, lateral_mod, Kx_norm, Ky_norm]
    """
    hand_pos = env.data.xpos[env.hand_body_id]
    bottle_xy = env.data.xpos[env.bottle_body_id][:2]
    tilt = get_bottle_tilt(env)

    touching = is_touching(
        env.model, env.data, env.bottle_body_id, env.robot_contact_bodies
    )

    # Adaptive Z reference
    bottle_y = env.data.xpos[env.bottle_body_id][1]
    distance_from_start = abs(bottle_y - env.bottle_start_y)
    current_target_z = TARGET_Z + 0.02 * distance_from_start
    current_target_z = max(current_target_z, 0.88)

    # Parse action
    forward_mod = float(action[0])
    lateral_mod = float(action[1])

    env.current_K[0] = K_MIN + float(action[2]) * (K_MAX - K_MIN)
    env.current_K[1] = K_MIN + float(action[3]) * (K_MAX - K_MIN)
    K_avg = float(np.mean(env.current_K))

    # --------------------------------------------------------
    # Tilt safety / settling mode
    # --------------------------------------------------------
    if env.is_settling:
        env.settle_counter += 1

        if check_stable(env):
            env.is_settling = False
            env.settle_counter = 0
        elif env.settle_counter >= 200:
            env.is_settling = False
            env.settle_counter = 0

        if env.is_settling:
            v_desired = np.zeros(3, dtype=np.float32)
            v_desired[2] = Z_GAIN * (current_target_z - hand_pos[2])
            return v_desired

    if tilt < TILT_STOP:
        env.is_settling = True
        env.settle_counter = 0

    if tilt > TILT_OK:
        speed_mult = 1.0
    elif tilt > TILT_SLOW:
        speed_mult = 0.5
    else:
        speed_mult = 0.2

    v_desired = np.zeros(3, dtype=np.float32)

    # --------------------------------------------------------
    # Contact recovery
    # --------------------------------------------------------
    if not touching:
        bottle_direction = bottle_xy - hand_pos[:2]
        bottle_dist = np.linalg.norm(bottle_direction)

        if bottle_dist > 0.01:
            bottle_dir_normalized = bottle_direction / bottle_dist
            recovery_speed = min(0.08, bottle_dist * 3.0)

            v_desired[0] = bottle_dir_normalized[0] * recovery_speed
            v_desired[1] = bottle_dir_normalized[1] * recovery_speed
            v_desired[2] = Z_GAIN * (current_target_z - hand_pos[2])

            if env.episode_length % 50 == 0:
                print(f"    CONTACT LOST! Recovery: dist={bottle_dist * 100:.1f}cm")

            return v_desired

    # --------------------------------------------------------
    # RL-driven push direction
    # --------------------------------------------------------
    desired_push_dir = get_rl_push_direction(env, bottle_xy, lateral_mod)
    env.last_desired_push_dir = desired_push_dir.copy()

    base_speed = BASE_FORWARD_SPEED * (1.0 + forward_mod * 0.3) * speed_mult
    k_factor = 0.8 + 0.4 * (K_avg / K_MAX)
    push_speed = base_speed * k_factor
    v_push = desired_push_dir * push_speed

    # --------------------------------------------------------
    # Hand tracking with RL lateral offset
    # --------------------------------------------------------
    target_hand_pos = get_hand_target_position(env, bottle_xy, lateral_mod)
    pos_error = target_hand_pos[:2] - hand_pos[:2]

    v_tracking = pos_error * 3.0
    v_tracking = np.clip(v_tracking, -0.03, 0.03)

    # Combine pushing and tracking
    v_desired[0] = v_push[0] + v_tracking[0]
    v_desired[1] = v_push[1] + v_tracking[1]

    # Limit total XY speed
    v_mag = np.linalg.norm(v_desired[:2])
    if v_mag > 0.05:
        v_desired[:2] = v_desired[:2] / v_mag * 0.05

    v_desired[2] = Z_GAIN * (current_target_z - hand_pos[2])

    angle_error = get_angle_error(env)

    if hasattr(env, "angle_error_log"):
        env.angle_error_log.append(angle_error)

    if env.episode_length % 100 == 0:
        _, deviation_mag = get_path_deviation(bottle_xy, env.trajectory)
        angle_deg = np.degrees(angle_error)
        print(
            f"    RL push_dir=[{desired_push_dir[0]:.2f}, {desired_push_dir[1]:.2f}], "
            f"lat_mod={lateral_mod:.2f}, dev={deviation_mag * 100:.1f}cm, θ={angle_deg:.1f}°"
        )

    return v_desired.astype(np.float32)