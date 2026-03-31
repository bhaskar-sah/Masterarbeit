import numpy as np
import csv
import os


class EpisodeLogger:
    """
    Owns and manages all per-step episode logs.

    Logs:
        force_profile    : contact force (3D) at each step
        stiffness_profile: K values (2D) at each step
        deviation        : path deviation magnitude (scalar) at each step
        angle_error      : force/push angle error (scalar) at each step
    """

    def __init__(self):
        self.force_profile: list = []
        self.stiffness_profile: list = []
        self.deviation: list = []
        self.angle_error: list = []

    def reset(self):
        """Clear all logs at the start of a new episode."""
        self.force_profile = []
        self.stiffness_profile = []
        self.deviation = []
        self.angle_error = []

    def log_step(self, force: np.ndarray, stiffness: np.ndarray, deviation: float):
        """Call once per env step to record force, stiffness, and deviation."""
        self.force_profile.append(force.copy())
        self.stiffness_profile.append(stiffness.copy())
        self.deviation.append(float(deviation))

    def log_angle_error(self, angle_error: float):
        """Call from PushController each push step to record angle error."""
        self.angle_error.append(float(angle_error))

    def get_summary(self) -> dict:
        """Return per-episode statistics as a flat dict."""
        summary = {}

        if self.force_profile:
            forces = np.array(self.force_profile)
            force_mags = np.linalg.norm(forces, axis=1)
            summary["force_mean"] = float(np.mean(force_mags))
            summary["force_max"] = float(np.max(force_mags))
            summary["force_std"] = float(np.std(force_mags))

        if self.stiffness_profile:
            stiffness = np.array(self.stiffness_profile)
            K_avg = stiffness.mean(axis=1)
            summary["K_mean"] = float(np.mean(K_avg))
            summary["K_min_seen"] = float(np.min(stiffness))
            summary["K_max_seen"] = float(np.max(stiffness))

        if self.deviation:
            devs = np.array(self.deviation)
            summary["deviation_mean_cm"] = float(np.mean(devs) * 100)
            summary["deviation_max_cm"] = float(np.max(devs) * 100)

        if self.angle_error:
            angles_deg = np.degrees(self.angle_error)
            summary["angle_error_mean_deg"] = float(np.mean(angles_deg))
            summary["angle_error_max_deg"] = float(np.max(angles_deg))

        summary["total_steps"] = len(self.force_profile)
        return summary

    def save_to_csv(self, filepath: str):
        """
        Save step-by-step logs to a CSV file.
        Each row: step, Fx, Fy, Fz, |F|, Kx, Ky, deviation, angle_error
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

        steps = len(self.force_profile)
        # Pad angle_error if push phase started after approach
        angle_errors = self.angle_error + [float("nan")] * (steps - len(self.angle_error))

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "Fx", "Fy", "Fz", "F_mag", "Kx", "Ky", "deviation_m", "angle_error_rad"])
            for i in range(steps):
                force = self.force_profile[i]
                K = self.stiffness_profile[i]
                dev = self.deviation[i]
                ae = angle_errors[i]
                writer.writerow([
                    i,
                    round(float(force[0]), 5),
                    round(float(force[1]), 5),
                    round(float(force[2]), 5),
                    round(float(np.linalg.norm(force)), 5),
                    round(float(K[0]), 2),
                    round(float(K[1]), 2),
                    round(float(dev), 5),
                    round(float(ae), 5) if not np.isnan(ae) else "",
                ])
