import numpy as np


# ============================================================
# CLOSEST POINT ON TRAJECTORY
# ============================================================

def get_closest_point_on_trajectory(pos_xy, trajectory):
    """
    Find closest point on trajectory and corresponding arc length.

    Returns:
        idx          : index of closest point
        closest_pt   : closest point (2D)
        arc_len      : arc length up to that point
    """
    if trajectory is None or len(trajectory) == 0:
        return 0, pos_xy.copy(), 0.0

    distances = np.linalg.norm(trajectory - pos_xy, axis=1)
    idx = int(np.argmin(distances))
    closest_pt = trajectory[idx].copy()

    if idx > 0:
        diffs = np.diff(trajectory[:idx + 1], axis=0)
        arc_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        arc_len = 0.0

    return idx, closest_pt, arc_len


# ============================================================
# PATH TANGENT
# ============================================================

def get_path_tangent(bottle_xy, trajectory):
    """
    Compute local tangent direction of trajectory at closest point.
    """
    if trajectory is None or len(trajectory) < 2:
        return np.array([0.0, -1.0], dtype=np.float32)

    idx, _, _ = get_closest_point_on_trajectory(bottle_xy, trajectory)

    if idx < len(trajectory) - 1:
        tangent = trajectory[idx + 1] - trajectory[idx]
    else:
        tangent = trajectory[idx] - trajectory[idx - 1]

    norm = np.linalg.norm(tangent)

    if norm > 1e-6:
        return (tangent / norm).astype(np.float32)

    return np.array([0.0, -1.0], dtype=np.float32)


# ============================================================
# PATH DEVIATION
# ============================================================

def get_path_deviation(bottle_xy, trajectory):
    """
    Compute deviation vector and magnitude from trajectory.
    """
    _, closest_pt, _ = get_closest_point_on_trajectory(bottle_xy, trajectory)

    deviation_vec = closest_pt - bottle_xy
    deviation_mag = float(np.linalg.norm(deviation_vec))

    return deviation_vec.astype(np.float32), deviation_mag


# ============================================================
# PROGRESS ALONG TRAJECTORY
# ============================================================

def get_progress(bottle_xy, trajectory, total_arc_length):
    """
    Compute normalized progress along trajectory [0, 1].
    """
    if total_arc_length < 1e-6:
        return 0.0

    _, _, arc_len = get_closest_point_on_trajectory(bottle_xy, trajectory)

    progress = arc_len / total_arc_length
    return float(np.clip(progress, 0.0, 1.0))


# ============================================================
# LOOKAHEAD TARGET (OPTIONAL BUT USEFUL)
# ============================================================

def get_lookahead_target(bottle_xy, trajectory, lookahead_points, goal_position):
    """
    Get future point on trajectory (used for predictive pushing).
    """
    if trajectory is None or len(trajectory) < 2:
        return goal_position.copy(), 0

    idx, _, _ = get_closest_point_on_trajectory(bottle_xy, trajectory)

    target_idx = min(idx + lookahead_points, len(trajectory) - 1)
    target_point = trajectory[target_idx].copy()

    # If near end → go directly to goal
    if target_idx >= len(trajectory) - 2:
        target_point = goal_position.copy()

    return target_point.astype(np.float32), target_idx