import numpy as np

class TrajectoryManager:
    """
    Manages path generation and geometric trajectory calculations for the Panda push env.
    """
    def __init__(self, goal_position=np.array([0.4, -0.2]), path_tolerance=0.05):
        self.trajectory = None
        self.total_arc_length = 0.0
        self.goal_position = goal_position
        self.path_tolerance = path_tolerance
        self.trajectory_type = "straight"

    def generate_trajectory(self, bottle_start_xy, traj_type="straight"):
        """Generate trajectory from bottle start position to fixed goal"""
        start = bottle_start_xy.copy()
        n_points = 50
        goal = self.goal_position.copy()

        if traj_type == "straight":
            t = np.linspace(0, 1, n_points)
            self.trajectory = np.outer(1 - t, start) + np.outer(t, goal)

        elif traj_type == "curved":
            t = np.linspace(0, 1, n_points)
            mid_point = np.array([0.55, 0.0])
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid_point[0] + t ** 2 * goal[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid_point[1] + t ** 2 * goal[1]
            self.trajectory = np.stack([x,y], axis=1)

        elif traj_type == "s_curve":
            t = np.linspace(0, 1, n_points)
            wiggle_strength = np.sin(np.pi * t)  # Peaks at middle, zero at ends
            # x = start[0] + 0.08 * np.sin(2 * np.pi * t) * wiggle_strength
            x = start[0] + 0.04 * np.sin(2 * np.pi * t) * wiggle_strength
            y = start[1] + t * (goal[1] - start[1])
            self.trajectory = np.stack([x, y], axis=1)

        else:
            t = np.linspace(0, 1, n_points)
            self.trajectory = np.outer(1 - t, start) + np.outer(t, goal)

        self.trajectory_type = traj_type
        self.trajectory = self.trajectory.astype(np.float32)
        self._compute_arc_length()
        return self.trajectory


    def _compute_arc_length(self):
        """
        Compute total arc length of trajectory
        """
        if self.trajectory is None or len(self.trajectory) < 2:
            self.total_arc_length = 0.0
            return
        diffs = np.diff(self.trajectory, axis=0)
        self.total_arc_length = np.sum(np.linalg.norm(diffs, axis=1))


    def _get_closest_point_on_trajectory(self, pos_xy):
        """
        Find closest point on trajectory to given position
        """
        if self.trajectory is None:
            return 0, pos_xy.copy(), 0.0
        distances = np.linalg.norm(self.trajectory - pos_xy, axis=1)
        idx = np.argmin(distances)
        closest_pt = self.trajectory[idx].copy()
        arc_len = np.sum(np.linalg.norm(np.diff(self.trajectory[:idx + 1], axis=0), axis=1)) if idx > 0 else 0.0
        return idx, closest_pt, arc_len


    def _get_path_deviation(self, bottle_xy):
        """
        Get deviation vector from bottle to trajectory.
        """
        _, closest_pt, _ = self._get_closest_point_on_trajectory(bottle_xy)
        deviation_vec = closest_pt - bottle_xy
        return deviation_vec.astype(np.float32), float(np.linalg.norm(deviation_vec))


    def _get_path_tangent(self, bottle_xy):
        """
        Get tangent vector at bottle's closest point on trajectory
        """
        if self.trajectory is None or len(self.trajectory) < 2:
            return np.array([0.0, -1.0], dtype=np.float32)
        idx, _, _ = self._get_closest_point_on_trajectory(bottle_xy)
        if idx < len(self.trajectory) - 1:
            tangent = self.trajectory[idx + 1] - self.trajectory[idx]
        else:
            tangent = self.trajectory[idx] - self.trajectory[idx - 1]
        norm = np.linalg.norm(tangent)
        return (tangent / norm).astype(np.float32) if norm > 1e-6 else np.array([0.0, -1.0], dtype=np.float32)


    def _get_progress(self, bottle_xy):
        """
        Get progress along trajectory (0.0 = start, 1.0 = goal)
        """
        if self.total_arc_length < 1e-6:
            return 0.0
        _, _, arc_len = self._get_closest_point_on_trajectory(bottle_xy)
        return float(np.clip(arc_len / self.total_arc_length, 0.0, 1.0))