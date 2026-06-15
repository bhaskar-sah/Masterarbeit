"""
trajectory.py

Trajectory Manager for Panda Push Environment.

The new addition to this file is bascially is that it now uses central difference  instead of forward difference.
This produces a smoother, less jittery tangent estimate, which matters for curved and s-curve trajectories where adjacent forward differences
can change direction sharply between segments.

forward difference: TANGENT = traj[idx+1] - traj[idx]
central difference: TANGENT = traj[idx+1] - traj[idx-1]

The central difference averages two adjacent segments, giving a tangent that better approximates the local direction of the underlying smooth
curve. For straight trajectories the result is identical to forward difference (up to scaling, which doesn't matter after normalization).

Generates and manages reference trajectories (straight, curved, s-curve).
Provides trajectory-related computations for observation and reward.
"""

import numpy as np
 
 
class TrajectoryManager:
    """Manages trajectory generation and trajectory-related computations."""
 
    def __init__(self, goal_position, path_tolerance=0.05, lookahead_points=4):
        self.goal_position = np.array(goal_position)
        self.path_tolerance = path_tolerance
        self.lookahead_points = lookahead_points
 
        self.trajectory = None
        self.trajectory_type = "straight"
        self.total_arc_length = 0.0

 # TODO: trajectory generation should be in 3D not just planar 2D
    def generate_trajectory(self, bottle_start_xy, traj_type=None):
        """Generate trajectory from bottle start to goal."""
        if traj_type is not None:
            self.trajectory_type = traj_type
 
        start = np.array(bottle_start_xy)
        goal = self.goal_position.copy()
        n_points = 50
 
        if self.trajectory_type == "straight":
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, goal)
            self.trajectory = traj.astype(np.float32)
 
        elif self.trajectory_type == "curved":
            t = np.linspace(0, 1, n_points)
            mid_point = np.array([0.55, 0.0])
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid_point[0] + t ** 2 * goal[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid_point[1] + t ** 2 * goal[1]
            self.trajectory = np.stack([x, y], axis=1).astype(np.float32)

        elif self.trajectory_type == "curved_opposite":
            # Quadratic Bezier curve pulling to the LEFT
            t = np.linspace(0, 1, n_points)
            
            # Straight line is at x=0.40. Original curve goes +0.15 to x=0.55.
            # Opposite curve goes -0.15 to x=0.25.
            mid_point = np.array([0.25, 0.0]) 
            
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * mid_point[0] + t ** 2 * goal[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * mid_point[1] + t ** 2 * goal[1]
            self.trajectory = np.stack([x, y], axis=1).astype(np.float32)
 
        elif self.trajectory_type == "s_curve":
            t = np.linspace(0, 1, n_points)
            wiggle_strength = np.sin(np.pi * t)
            x = start[0] + 0.08 * np.sin(2 * np.pi * t) * wiggle_strength
            y = start[1] + t * (goal[1] - start[1])
            self.trajectory = np.stack([x, y], axis=1).astype(np.float32)
 
        else:
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, goal)
            self.trajectory = traj.astype(np.float32)
 
        self._compute_arc_length()
 
    def _compute_arc_length(self):
        if self.trajectory is None or len(self.trajectory) < 2:
            self.total_arc_length = 0.0
            return
        diffs = np.diff(self.trajectory, axis=0)
        self.total_arc_length = np.sum(np.linalg.norm(diffs, axis=1))

#TODO: Changed to give continues closest point and arc_length by projection on path
    # commented previous calculation below
    def get_closest_point_on_trajectory(self, pos_xy):
        """
        Fast version:
        - uses argmin to find closest vertex
        - only checks adjacent segments for projection

        Returns:
            best_idx      -> segment index
            closest_point -> projected closest point
            arc_length    -> continuous arc-length
        """

        traj = self.trajectory

        if traj is None or len(traj) < 2:
            return 0, pos_xy.copy(), 0.0

        # ---- 1. find closest trajectory point (vectorized, fast) ----
        distances = np.linalg.norm(traj - pos_xy, axis=1)
        idx = int(np.argmin(distances))

        # ---- 2. select candidate segments ----
        candidates = []

        if idx > 0:
            candidates.append(idx - 1)

        if idx < len(traj) - 1:
            candidates.append(idx)

        # ---- 3. project onto candidate segments ----
        best_dist = float("inf")
        best_point = None
        best_idx = 0
        best_t = 0.0

        for i in candidates:
            p0 = traj[i]
            p1 = traj[i + 1]

            v = p1 - p0
            seg_len = np.linalg.norm(v)

            if seg_len < 1e-8:
                continue

            v_unit = v / seg_len

            w = pos_xy - p0
            t = np.dot(w, v_unit)

            t_clamped = np.clip(t, 0.0, seg_len)

            proj = p0 + t_clamped * v_unit
            dist = np.linalg.norm(pos_xy - proj)

            if dist < best_dist:
                best_dist = dist
                best_point = proj
                best_idx = i
                best_t = t_clamped

        # ---- 4. compute arc length ----
        arc_length = np.sum(np.linalg.norm(np.diff(self.trajectory[:best_idx + 1], axis=0), axis=1)) + best_t

        return best_idx, best_point, arc_length

    # def get_closest_point_on_trajectory(self, pos_xy):
    #     """Returns (index, closest_point, arc_length_to_point)."""
    #     if self.trajectory is None:
    #         return 0, pos_xy.copy(), 0.0
    #
    #     distances = np.linalg.norm(self.trajectory - pos_xy, axis=1)
    #     idx = int(np.argmin(distances))
    #     closest_pt = self.trajectory[idx].copy()
    #
    #     if idx > 0:
    #         arc_len = np.sum(np.linalg.norm(np.diff(self.trajectory[:idx + 1], axis=0), axis=1))
    #     else:
    #         arc_len = 0.0
    #
    #     return idx, closest_pt, arc_len
 
    def get_path_deviation(self, bottle_xy):
        """Returns (deviation_vec, deviation_magnitude). Vec points bottle->trajectory."""
        _, closest_pt, _ = self.get_closest_point_on_trajectory(bottle_xy)
        deviation_vec = closest_pt - bottle_xy
        return deviation_vec.astype(np.float32), float(np.linalg.norm(deviation_vec))
 
    def get_path_tangent(self, bottle_xy):
        """
        Get tangent vector at bottle's closest trajectory point.
 
        Uses CENTRAL DIFFERENCE (smoother for curved trajectories):
            tangent = traj[idx+1] - traj[idx-1]
 
        Falls back to one-sided differences at endpoints.
        """
        if self.trajectory is None or len(self.trajectory) < 2:
            return np.array([0.0, 0.0], dtype=np.float32)
 
        idx, _, _ = self.get_closest_point_on_trajectory(bottle_xy)
        n = len(self.trajectory)
 
        if idx == 0:
            # Forward difference at start
            tangent = self.trajectory[1] - self.trajectory[0]
        elif idx >= n - 1:
            # Backward difference at end
            tangent = self.trajectory[n - 1] - self.trajectory[n - 2]
        else:
            # Central difference everywhere else
            tangent = self.trajectory[idx + 1] - self.trajectory[idx - 1]
 
        norm = np.linalg.norm(tangent)
        if norm > 1e-6:
            return (tangent / norm).astype(np.float32)
        else:
            return np.array([0.0, 0.0], dtype=np.float32)
 
    def get_progress(self, bottle_xy):
        """Returns progress in [0, 1]."""
        if self.total_arc_length < 1e-6:
            return 0.0
        _, _, arc_len = self.get_closest_point_on_trajectory(bottle_xy)
        return float(np.clip(arc_len / self.total_arc_length, 0.0, 1.0))
 
    def get_lookahead_target(self, bottle_xy):
        """
        Get target point ahead of bottle's closest trajectory point.
        Returns (target_point, target_index).
        """
        if self.trajectory is None or len(self.trajectory) < 2:
            return self.goal_position.copy(), 0
 
        idx, _, _ = self.get_closest_point_on_trajectory(bottle_xy)
        target_idx = min(idx + self.lookahead_points, len(self.trajectory) - 1)
        target_point = self.trajectory[target_idx].copy()
 
        # Near end: snap to goal directly
        if target_idx >= len(self.trajectory) - 2:
            target_point = self.goal_position.copy()
 
        return target_point, target_idx
 
    def get_push_direction(self, bottle_xy):
        """
        Get blended push direction.
 
        On trajectory   -> push along tangent
        Off trajectory  -> push toward lookahead target (correction)
        Smooth blend prevents oscillation.
 
        Returns (push_direction, distance_to_target, target_point).
        """
        target_point, _ = self.get_lookahead_target(bottle_xy)
        _, deviation_mag = self.get_path_deviation(bottle_xy)
 
        # Vector from bottle to lookahead target
        to_target = target_point - bottle_xy
        distance = float(np.linalg.norm(to_target))
 
        if distance > 1e-6:
            correction_dir = to_target / distance
        else:
            correction_dir = self.get_path_tangent(bottle_xy)
 
        tangent_dir = self.get_path_tangent(bottle_xy)
 
        # Blend: 0 (pure tangent) when dev<0.5cm, 1 (pure correction) when dev>2.5cm
        blend = float(np.clip((deviation_mag - 0.005) / 0.02, 0.0, 1.0))
 
        push_direction = (1.0 - blend) * tangent_dir + blend * correction_dir
 
        norm = np.linalg.norm(push_direction)
        if norm > 1e-6:
            push_direction = push_direction / norm
        else:
            push_direction = correction_dir
 
        return push_direction.astype(np.float32), distance, target_point