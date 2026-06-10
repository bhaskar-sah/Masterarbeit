"""
logger.py
Episode Logger for task-space FORCE control.

Per-step episode logs:
    - force_profile      : WORLD-frame contact force (robot -> bottle) at each step
    - cmd_force_profile  : commanded planar force [Ft, Fb, 0] in N, path frame {P}
    - deviation          : path deviation magnitude (m) at each step
    - tilt_profile       : bottle upright component (z-cap . world-z)
    - action_profile     : raw RL action [Ft_norm, Fb_norm, Tz_norm] in [-1, 1]

NOTE: log_step()'s argument is still named `velocity=` for backward
compatibility with env.py's call site, but it carries COMMANDED FORCE
(path frame, Newtons), not a velocity. The stored profile and every output
column / summary key are labelled as force.
"""

import numpy as np
import csv
import os


class EpisodeLogger:
    """Per-episode logs for task-space force control."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all logs at the start of a new episode."""
        self.force_profile: list = []
        self.cmd_force_profile: list = []   # commanded planar force [Ft, Fb, 0] (N), path frame {P}
        self.deviation: list = []
        self.action_profile: list = []
        self.tilt_profile: list = []

    def log_step(self, force: np.ndarray, stiffness: np.ndarray = None,
                 deviation: float = 0.0, velocity: np.ndarray = None,
                 action: np.ndarray = None, tilt: float = 1.0):
        """
        Log data for one step.

        Args:
            force:    WORLD-frame contact force [fx, fy, fz] (robot -> bottle)
            stiffness:(legacy, unused) accepted for backward compatibility
            deviation:path deviation magnitude (m)
            velocity: (LEGACY NAME) commanded planar force [Ft, Fb, 0] in N,
                      expressed in the path frame {P}. NOT a velocity.
            action:   raw RL action [Ft_norm, Fb_norm, Tz_norm] in [-1, 1]
            tilt:     bottle upright component (1 = upright)
        """
        self.force_profile.append(force.copy() if force is not None else np.zeros(3))
        self.deviation.append(float(deviation))
        self.tilt_profile.append(float(tilt))

        if velocity is not None:
            self.cmd_force_profile.append(velocity.copy())

        if action is not None:
            self.action_profile.append(action.copy())

    def log_angle_error(self, angle_error: float):
        """Legacy method - kept for compatibility but not used."""
        pass

    def get_summary(self) -> dict:
        """Return per-episode statistics as a flat dict."""
        summary = {}

        if self.force_profile:
            forces = np.array(self.force_profile)
            force_mags = np.linalg.norm(forces, axis=1)
            summary["force_mean"] = float(np.mean(force_mags))
            summary["force_max"] = float(np.max(force_mags))
            summary["force_std"] = float(np.std(force_mags))

        # Commanded planar force magnitude (path frame), in Newtons.
        # (Previously mislabelled as "velocity_*".)
        if self.cmd_force_profile:
            cf = np.array(self.cmd_force_profile)
            cf_mag = np.linalg.norm(cf[:, :2], axis=1)
            summary["force_cmd_mean"] = float(np.mean(cf_mag))
            summary["force_cmd_max"] = float(np.max(cf_mag))
            summary["force_cmd_std"] = float(np.std(cf_mag))

        if self.deviation:
            devs = np.array(self.deviation)
            summary["deviation_mean_cm"] = float(np.mean(devs) * 100)
            summary["deviation_max_cm"] = float(np.max(devs) * 100)

        if self.tilt_profile:
            tilts = np.array(self.tilt_profile)
            summary["tilt_min"] = float(np.min(tilts))
            summary["tilt_mean"] = float(np.mean(tilts))

        summary["total_steps"] = len(self.force_profile)

        return summary

    def save_to_csv(self, filepath: str):
        """Save step-by-step logs to a CSV file."""
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        steps = len(self.force_profile)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header. F_contact_* are the WORLD-frame contact force (robot -> bottle).
            header = ["step", "F_contact_x", "F_contact_y", "F_contact_z",
                      "F_contact_mag", "deviation_m", "tilt"]
            if self.cmd_force_profile:
                # commanded force, path frame {P}: [t, b, n(=0)]
                header.extend(["F_cmd_t", "F_cmd_b", "F_cmd_n", "F_cmd_mag"])
            if self.action_profile:
                # raw action: [tangent, binormal, yaw], normalised to [-1, 1]
                header.extend(["a_ft", "a_fb", "a_tau_z"])
            writer.writerow(header)

            for i in range(steps):
                force = self.force_profile[i]
                force_mag = float(np.linalg.norm(force))
                dev = self.deviation[i]
                tilt = self.tilt_profile[i] if i < len(self.tilt_profile) else 1.0

                row = [
                    i,
                    round(float(force[0]), 5),
                    round(float(force[1]), 5),
                    round(float(force[2]), 5),
                    round(force_mag, 5),
                    round(float(dev), 5),
                    round(float(tilt), 5),
                ]

                if self.cmd_force_profile and i < len(self.cmd_force_profile):
                    cf = np.asarray(self.cmd_force_profile[i], dtype=float)
                    cf_mag = float(np.linalg.norm(cf[:2]))
                    row.extend([
                        round(float(cf[0]), 5),
                        round(float(cf[1]), 5),
                        round(float(cf[2]), 5),
                        round(cf_mag, 5),
                    ])

                if self.action_profile and i < len(self.action_profile):
                    action = self.action_profile[i]
                    row.extend([
                        round(float(action[0]), 5),
                        round(float(action[1]), 5),
                        round(float(action[2]), 5),
                    ])

                writer.writerow(row)