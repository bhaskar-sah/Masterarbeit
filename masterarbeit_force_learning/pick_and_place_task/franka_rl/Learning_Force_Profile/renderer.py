# renderer.py
"""
Trajectory Renderer for Panda Push Environment.

Handles MuJoCo passive viewer and trajectory visualization.
"""

import numpy as np
import mujoco
import mujoco.viewer


class TrajectoryRenderer:
    """
    Handles MuJoCo passive viewer and trajectory visualization.
    Draws the planned trajectory as a red capsule strip on the floor plane.
    """

    TRAJECTORY_Z = 0.801
    CAPSULE_RADIUS = 0.003
    TRAJ_COLOR = np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32)

    def __init__(self, model, data, render_mode: str):
        self.model = model
        self.data = data
        self.render_mode = render_mode
        self.viewer = None

    def render(self, trajectory: np.ndarray = None):
        """
        Launch viewer on first call, then draw trajectory and sync.
        """
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.user_scn.ngeom = 0

        self._draw_trajectory(trajectory)
        self.viewer.sync()

    def _draw_trajectory(self, trajectory: np.ndarray = None):
        """Draw trajectory as a strip of red capsules."""
        if trajectory is None:
            return

        self.viewer.user_scn.ngeom = 0

        for i in range(len(trajectory) - 1):
            if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom:
                break

            p1 = np.array([trajectory[i][0], trajectory[i][1], self.TRAJECTORY_Z])
            p2 = np.array([trajectory[i + 1][0], trajectory[i + 1][1], self.TRAJECTORY_Z])

            geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[self.CAPSULE_RADIUS, 0, 0],
                pos=(p1 + p2) / 2,
                mat=np.eye(3).flatten(),
                rgba=self.TRAJ_COLOR,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                self.CAPSULE_RADIUS,
                p1, p2,
            )
            self.viewer.user_scn.ngeom += 1

    def close(self):
        """Close the viewer if open."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None