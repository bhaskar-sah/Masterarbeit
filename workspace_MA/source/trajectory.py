import numpy as np


def generate_trajectory(bottle_start_xy, goal_position, traj_type="straight", n_points=50):
    """
    Generate a 2D reference trajectory from bottle start position to goal.

    Parameters
    ----------
    bottle_start_xy : np.ndarray
        Initial bottle position in xy plane, shape (2,)
    goal_position : np.ndarray
        Goal position in xy plane, shape (2,)
    traj_type : str
        Type of trajectory: "straight", "curved", or "s_curve"
    n_points : int
        Number of trajectory sample points

    Returns
    -------
    np.ndarray
        Trajectory array of shape (n_points, 2)
    """
    start = bottle_start_xy.copy()
    goal = goal_position.copy()

    if traj_type == "straight":
        t = np.linspace(0.0, 1.0, n_points)
        traj = np.outer(1 - t, start) + np.outer(t, goal)

    elif traj_type == "curved":
        t = np.linspace(0.0, 1.0, n_points)

        # Quadratic Bézier-like path
        mid_point = np.array([0.55, 0.0], dtype=np.float32)

        x = ((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid_point[0] + t ** 2 * goal[0])
        y = ((1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid_point[1] + t ** 2 * goal[1])

        traj = np.stack([x, y], axis=1)

    elif traj_type == "s_curve":
        t = np.linspace(0.0, 1.0, n_points)

        wiggle_strength = np.sin(np.pi * t)
        x = start[0] + 0.04 * np.sin(2 * np.pi * t) * wiggle_strength
        y = start[1] + t * (goal[1] - start[1])

        traj = np.stack([x, y], axis=1)

    else:
        # fallback to straight trajectory
        t = np.linspace(0.0, 1.0, n_points)
        traj = np.outer(1 - t, start) + np.outer(t, goal)

    return traj.astype(np.float32)


def compute_arc_length(trajectory):
    """
    Compute total arc length of a 2D trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory array of shape (N, 2)

    Returns
    -------
    float
        Total arc length
    """
    if trajectory is None or len(trajectory) < 2:
        return 0.0

    diffs = np.diff(trajectory, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(segment_lengths))