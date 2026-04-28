# logger.py
"""
Episode Logger for Force-Velocity Control.

Logs:
    - force_profile: Contact force at each step
    - velocity_profile: Commanded velocity at each step
    - deviation: Path deviation magnitude at each step
"""

import numpy as np
import csv
import os


class EpisodeLogger:
    """
    Manages per-step episode logs for force-velocity control.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all logs at the start of a new episode."""
        self.force_profile: list = []
        self.stiffness_profile: list = []  # Keep for compatibility, will be zeros
        self.deviation: list = []
        self.velocity_profile: list = []
        self.action_profile: list = []
        self.tilt_profile: list = []

    def log_step(self, force: np.ndarray, stiffness: np.ndarray = None,
                 deviation: float = 0.0, velocity: np.ndarray = None,
                 action: np.ndarray = None, tilt: float = 1.0):
        """
        Log data for one step.

        Args:
            force: Contact force [fx, fy, fz]
            stiffness: (Legacy) Stiffness values (not used, kept for compatibility)
            deviation: Path deviation magnitude
            velocity: Commanded velocity [vx, vy, vz] (optional)
            action: Raw RL action [a0, a1, a2, a3] (optional)
            tilt: Bottle tilt value
        """
        self.force_profile.append(force.copy() if force is not None else np.zeros(3))
        self.deviation.append(float(deviation))
        self.tilt_profile.append(float(tilt))

        if stiffness is not None:
            self.stiffness_profile.append(stiffness.copy())
        else:
            self.stiffness_profile.append(np.zeros(2))

        if velocity is not None:
            self.velocity_profile.append(velocity.copy())

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

        if self.velocity_profile:
            velocities = np.array(self.velocity_profile)
            vel_mags = np.linalg.norm(velocities, axis=1)
            summary["velocity_mean"] = float(np.mean(vel_mags))
            summary["velocity_max"] = float(np.max(vel_mags))

        if self.deviation:
            devs = np.array(self.deviation)
            summary["deviation_mean_cm"] = float(np.mean(devs) * 100)
            summary["deviation_max_cm"] = float(np.max(devs) * 100)

        if self.tilt_profile:
            tilts = np.array(self.tilt_profile)
            summary["tilt_min"] = float(np.min(tilts))
            summary["tilt_mean"] = float(np.mean(tilts))

        if self.action_profile:
            actions = np.array(self.action_profile)
            summary["force_cmd_mean"] = float(np.mean(actions[:, 3]))
            summary["force_cmd_std"] = float(np.std(actions[:, 3]))

        summary["total_steps"] = len(self.force_profile)

        return summary

    def save_to_csv(self, filepath: str):
        """Save step-by-step logs to a CSV file."""
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        steps = len(self.force_profile)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["step", "Fx", "Fy", "Fz", "F_mag", "deviation_m", "tilt"]
            if self.velocity_profile:
                header.extend(["vx", "vy", "vz", "v_mag"])
            if self.action_profile:
                header.extend(["a0_vx", "a1_vy", "a2_vz", "a3_f"])
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

                if self.velocity_profile and i < len(self.velocity_profile):
                    vel = self.velocity_profile[i]
                    vel_mag = float(np.linalg.norm(vel))
                    row.extend([
                        round(float(vel[0]), 5),
                        round(float(vel[1]), 5),
                        round(float(vel[2]), 5),
                        round(vel_mag, 5),
                    ])

                if self.action_profile and i < len(self.action_profile):
                    action = self.action_profile[i]
                    row.extend([
                        round(float(action[0]), 5),
                        round(float(action[1]), 5),
                        round(float(action[2]), 5),
                        round(float(action[3]), 5),
                    ])

                writer.writerow(row)