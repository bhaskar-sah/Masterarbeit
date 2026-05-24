# debug_utils.py
"""
Debug Utilities for Panda Push Training.

Provides:
1. Enhanced console debug printing
2. CSV step logger for post-training analysis
"""

import numpy as np
import csv
import os


class DebugPrinter:
    """Enhanced console debug output during training."""

    def __init__(self, config, print_every=50):
        self.config = config
        self.print_every = print_every

    def print_step(self, step, hand_pos, bottle_pos, action, reward,
                   info, push_dir, p_des, F_cmd):
        """
        Print detailed debug info for one step.
        """
        if step % self.print_every != 0:
            return

        progress = info.get("progress", 0.0)
        deviation = info.get("deviation", 0.0)
        tilt = info.get("bottle_tilt", 1.0)
        force_mag = info.get("force_magnitude", 0.0)
        is_touching = info.get("is_touching", False)

        contact_str = "CONTACT" if is_touching else "NO CONTACT"

        # Scaled actions (Now only 3D)
        vx_scaled = action[0] * self.config.v_max
        vy_scaled = action[1] * self.config.v_max
        wz_scaled = action[2] * self.config.w_max

        # Hand-to-bottle vector
        h2b = bottle_pos[:2] - hand_pos[:2]
        h2b_dist = np.linalg.norm(h2b)

        # Position error magnitude
        p_error = np.linalg.norm(p_des - hand_pos) if p_des is not None else 0.0

        # F_cmd magnitude
        F_cmd_mag = np.linalg.norm(F_cmd) if F_cmd is not None else 0.0

        print(f"\nStep {step:4d} | {contact_str} | Reward: {reward:7.2f}")
        print(f"  ── Positions ──")
        print(f"    Hand:    ({hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f})")
        print(f"    Bottle:  ({bottle_pos[0]:.3f}, {bottle_pos[1]:.3f}, {bottle_pos[2]:.3f})")
        print(f"    p_des:   ({p_des[0]:.3f}, {p_des[1]:.3f}, {p_des[2]:.3f})" if p_des is not None else "    p_des:   None")
        print(f"    H→B dist: {h2b_dist*100:.1f}cm")
        print(f"  ── Actions (raw → scaled) ──")
        print(f"    vx: {action[0]:+5.2f} → {vx_scaled*1000:+6.2f} mm/s")
        print(f"    vy: {action[1]:+5.2f} → {vy_scaled*1000:+6.2f} mm/s")
        print(f"    wz: {action[2]:+5.2f} → {wz_scaled:+6.2f} rad/s")
        print(f"  ── Forces ──")
        print(f"    Push dir:    ({push_dir[0]:+.3f}, {push_dir[1]:+.3f})")
        print(f"    F_cmd total: {F_cmd_mag:.2f} N")
        print(f"    F_measured:  {force_mag:.2f} N")
        print(f"    p_error:     {p_error*1000:.2f} mm")
        print(f"  ── Task ──")
        print(f"    Progress: {progress*100:5.1f}%  |  Dev: {deviation*100:4.1f}cm  |  Tilt: {tilt:.4f}")
        print(f"  ── Rewards ──")
        print(f"    r_prog: {info.get('r_progress', 0):+6.3f}  "
              f"r_dev: {info.get('r_deviation', 0):+6.3f}  "
              f"r_stab: {info.get('r_stability', 0):+6.3f}  "
              f"r_contact: {info.get('r_contact', 0):+6.3f}  "
              f"r_align: {info.get('r_alignment', 0):+6.3f}  "
              f"r_pos: {info.get('r_position', 0):+6.3f}  "
              f"r_vel: {info.get('r_velocity',0):+6.6f}")
        print(f"{'─' * 70}")


class StepLogger:
    """
    Logs per-step data to CSV for post-training analysis.
    """

    def __init__(self, filepath, flush_every=1000):
        self.filepath = filepath
        self.flush_every = flush_every
        self.step_count = 0

        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.file = open(filepath, "w", newline="")
        self.writer = csv.writer(self.file)

        # Write header
        self.writer.writerow([
            "global_step", "episode_step", "episode_num",
            # Positions
            "hand_x", "hand_y", "hand_z",
            "bottle_x", "bottle_y", "bottle_z",
            "p_des_x", "p_des_y", "p_des_z",
            "hand_to_bottle_dist",
            # Raw actions (3D)
            "a_vx", "a_vy", "a_wz",
            # Scaled actions
            "vx_mps", "vy_mps", "wz_rps",
            # ACTUAL VELOCITY
            "v_curr_x", "v_curr_y", "v_curr_z",
            # Forces
            "push_dir_x", "push_dir_y",
            "F_cmd_mag", "F_measured_mag",
            "F_meas_x", "F_meas_y", "F_meas_z",
            # Task state
            "progress", "deviation_m", "tilt",
            "is_touching",
            # Rewards
            "reward_total",
            "r_progress", "r_deviation", "r_stability",
            "r_contact", "r_alignment", "r_position", "r_velocity",
            # Position error
            "p_error_mag",
        ])

    def log(self, global_step, episode_step, episode_num,
            hand_pos, bottle_pos, p_des,
            action, config,
            push_dir, F_cmd, F_measured,
            reward, info, v_current):
        """Log one step of data."""

        # Scaled actions
        vx_scaled = action[0] * config.v_max
        vy_scaled = action[1] * config.v_max
        wz_scaled = action[2] * config.w_max

        h2b_dist = np.linalg.norm(bottle_pos[:2] - hand_pos[:2])
        F_cmd_mag = np.linalg.norm(F_cmd) if F_cmd is not None else 0.0
        F_meas_mag = np.linalg.norm(F_measured)
        p_error = np.linalg.norm(p_des - hand_pos) if p_des is not None else 0.0

        self.writer.writerow([
            global_step, episode_step, episode_num,
            # Positions
            f"{hand_pos[0]:.5f}", f"{hand_pos[1]:.5f}", f"{hand_pos[2]:.5f}",
            f"{bottle_pos[0]:.5f}", f"{bottle_pos[1]:.5f}", f"{bottle_pos[2]:.5f}",
            f"{p_des[0]:.5f}" if p_des is not None else "",
            f"{p_des[1]:.5f}" if p_des is not None else "",
            f"{p_des[2]:.5f}" if p_des is not None else "",
            f"{h2b_dist:.5f}",
            # Raw actions
            f"{action[0]:.4f}", f"{action[1]:.4f}", f"{action[2]:.4f}",
            # Scaled actions
            f"{vx_scaled:.5f}", f"{vy_scaled:.5f}", f"{wz_scaled:.4f}",
            # ACTUAL VELOCITY
            f"{v_current[0]:.5f}", f"{v_current[1]:.5f}", f"{v_current[2]:.5f}",
            # Forces
            f"{push_dir[0]:.4f}", f"{push_dir[1]:.4f}",
            f"{F_cmd_mag:.4f}", f"{F_meas_mag:.4f}",
            f"{F_measured[0]:.4f}", f"{F_measured[1]:.4f}", f"{F_measured[2]:.4f}",
            # Task state
            f"{info.get('progress', 0):.5f}",
            f"{info.get('deviation', 0):.5f}",
            f"{info.get('bottle_tilt', 1.0):.5f}",
            1 if info.get("is_touching", False) else 0,
            # Rewards
            f"{reward:.5f}",
            f"{info.get('r_progress', 0):.5f}",
            f"{info.get('r_deviation', 0):.5f}",
            f"{info.get('r_stability', 0):.5f}",
            f"{info.get('r_contact', 0):.5f}",
            f"{info.get('r_alignment', 0):.5f}",
            f"{info.get('r_position', 0):.5f}",
            f"{info.get('r_velocity', 0):.5f}",
            # Position error
            f"{p_error:.5f}",
        ])

        self.step_count += 1
        if self.step_count % self.flush_every == 0:
            self.file.flush()

    def close(self):
        """Close the CSV file."""
        if self.file:
            self.file.flush()
            self.file.close()
            print(f"Step log saved to: {self.filepath}")