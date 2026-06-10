# debug_utils.py
"""
Debug Utilities for Panda Push Training (task-space FORCE control).

Provides:
    1. DebugPrinter - enhanced console debug output during training
    2. StepLogger   - per-step CSV logger for post-training analysis

Frame convention (see push_controller.py):
    {B} base frame at link0   - v_curr_* and *_base_* columns are base-frame
    {P} path frame [t, b, n]  - the action acts here; F_cmd_path_* are along [t, b, n]
    World frame               - ee_*, bottle_*, and F_contact_* are world-frame
In the current scene the base has identity orientation, so base- and
world-frame vectors are numerically identical; the labels record intent.
"""

import numpy as np
import csv
import os


class DebugPrinter:
    """Enhanced console debug output during training."""

    def __init__(self, config, print_every=50):
        self.config = config
        self.print_every = print_every

    def print_step(self, step, ee_pos, bottle_pos, action, reward,
                   info, push_dir, p_des, F_cmd):
        """Print detailed debug info for one step."""
        if step % self.print_every != 0:
            return

        progress = info.get("progress", 0.0)
        deviation = info.get("deviation", 0.0)
        tilt = info.get("bottle_tilt", 1.0)
        force_mag = info.get("force_magnitude", 0.0)
        is_touching = info.get("is_touching", False)

        contact_str = "CONTACT" if is_touching else "NO CONTACT"

        # Scaled actions (path frame {P})
        ft_scaled = action[0] * self.config.f_max           # along t_hat
        fb_scaled = action[1] * self.config.f_max           # along b_hat
        tau_z_scaled = action[2] * self.config.tau_rot_max  # yaw about n_hat

        # ee-to-bottle (both world frame here)
        h2b = bottle_pos[:2] - ee_pos[:2]
        h2b_dist = np.linalg.norm(h2b)

        F_cmd_mag = np.linalg.norm(F_cmd) if F_cmd is not None else 0.0

        print(f"\nStep {step:4d} | {contact_str} | Reward: {reward:7.2f}")
        print(f"  -- Positions (world) --")
        print(f"    ee:      ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
        print(f"    Bottle:  ({bottle_pos[0]:.3f}, {bottle_pos[1]:.3f}, {bottle_pos[2]:.3f})")
        print(f"    ee->B dist: {h2b_dist*100:.1f}cm")
        print(f"  -- Actions (path frame [t, b, n]) --")
        print(f"    Ft: {action[0]:+5.2f} -> {ft_scaled:+6.2f} N    (along tangent)")
        print(f"    Fb: {action[1]:+5.2f} -> {fb_scaled:+6.2f} N    (along binormal)")
        print(f"    Tz: {action[2]:+5.2f} -> {tau_z_scaled:+6.2f} N.m  (yaw)")
        print(f"  -- Forces --")
        print(f"    Push tangent (base): ({push_dir[0]:+.3f}, {push_dir[1]:+.3f})")
        print(f"    F_cmd total (base):  {F_cmd_mag:.2f} N")
        print(f"    F_measured (contact):{force_mag:.2f} N")
        print(f"  -- Task --")
        print(f"    Progress: {progress*100:5.1f}%  |  Dev: {deviation*100:4.1f}cm  |  Tilt: {tilt:.4f}")
        print(f"  -- Rewards --")
        print(f"    r_prog: {info.get('r_progress', 0):+6.3f}  "
              f"r_dev: {info.get('r_deviation', 0):+6.3f}  "
              f"r_stab: {info.get('r_stability', 0):+6.3f}  "
              f"r_contact: {info.get('r_contact', 0):+6.3f}  "
              f"r_align: {info.get('r_alignment', 0):+6.3f}  "
              f"r_vel: {info.get('r_velocity', 0):+6.6f}  "
              f"t_pen: {info.get('time_penalty', 0):+6.3f}")
        print("-" * 70)


class StepLogger:
    """Logs per-step data to CSV for post-training analysis."""

    def __init__(self, filepath, flush_every=1000):
        self.filepath = filepath
        self.flush_every = flush_every
        self.step_count = 0

        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.file = open(filepath, "w", newline="")
        self.writer = csv.writer(self.file)

        # Header.
        #   ee_*            : WORLD frame, gripper_center site
        #   bottle_*        : WORLD frame
        #   v_curr_*        : base frame {B} (= world here)
        #   push_dir_*      : path tangent t_hat (base frame)
        #   F_cmd_path_*    : commanded wrench in path frame {P} = [t, b, n]
        #                     (the n component of force is the PD height controller)
        #   *_base_*        : base frame {B}
        #   F_contact_*     : WORLD frame contact force (robot -> bottle)
        self.writer.writerow([
            "global_step", "episode_step", "episode_num",
            # Positions (world)
            "ee_x", "ee_y", "ee_z",
            "bottle_x", "bottle_y", "bottle_z",
            "ee_to_bottle_dist",
            # Raw actions (3D)
            "a_ft", "a_fb", "a_tau_z",
            # Scaled actions
            "ft_N", "fb_N", "tau_z_Nm",
            # Measured EE velocity (base)
            "v_curr_x", "v_curr_y", "v_curr_z",
            # Forces & torques
            "push_dir_x", "push_dir_y",
            "F_cmd_path_t", "F_cmd_path_b", "F_cmd_path_n",
            "tau_cmd_path_t", "tau_cmd_path_b", "tau_cmd_path_n",
            "F_cmd_base_x", "F_cmd_base_y", "F_cmd_base_z",
            "tau_cmd_base_x", "tau_cmd_base_y", "tau_cmd_base_z",
            "F_meas_base_x", "F_meas_base_y", "F_meas_base_z",
            "tau_meas_base_x", "tau_meas_base_y", "tau_meas_base_z",
            "F_contact_world_x", "F_contact_world_y", "F_contact_world_z",
            "roll", "pitch", "yaw",
            # Task state
            "progress", "deviation_m", "tilt",
            "is_touching",
            # Rewards
            "reward_total",
            "r_progress", "r_deviation", "r_stability",
            "r_contact", "r_alignment", "r_velocity", "time_penalty",
        ])

    def log(self, global_step, episode_step, episode_num,
            ee_pos, bottle_pos, p_des,
            action, config,
            push_dir, F_cmd_path, tau_cmd_path, F_cmd_base, tau_cmd_base,
            F_meas_base, tau_meas_base,
            rpy, F_contact,
            reward, info, v_current):
        """
        Log one step of data.

        Frames: ee_pos / bottle_pos / F_contact are WORLD; v_current and *_base
        are base frame {B}; F_cmd_path / tau_cmd_path are path frame {P} = [t, b, n].
        (F_cmd_path / tau_cmd_path arrive as push_controller.last_F_cmd_ee /
        last_tau_cmd_ee, which already hold the path-frame command.)
        """
        # Scaled actions
        ft_scaled = action[0] * config.f_max
        fb_scaled = action[1] * config.f_max
        tau_z_scaled = action[2] * config.tau_rot_max

        h2b_dist = np.linalg.norm(bottle_pos[:2] - ee_pos[:2])

        self.writer.writerow([
            global_step, episode_step, episode_num,
            # Positions (world)
            f"{ee_pos[0]:.5f}", f"{ee_pos[1]:.5f}", f"{ee_pos[2]:.5f}",
            f"{bottle_pos[0]:.5f}", f"{bottle_pos[1]:.5f}", f"{bottle_pos[2]:.5f}",
            f"{h2b_dist:.5f}",
            # Raw actions
            f"{action[0]:.4f}", f"{action[1]:.4f}", f"{action[2]:.4f}",
            # Scaled actions
            f"{ft_scaled:.5f}", f"{fb_scaled:.5f}", f"{tau_z_scaled:.4f}",
            # Measured EE velocity (base)
            f"{v_current[0]:.5f}", f"{v_current[1]:.5f}", f"{v_current[2]:.5f}",
            # Forces & torques
            f"{push_dir[0]:.4f}", f"{push_dir[1]:.4f}",
            f"{F_cmd_path[0]:.4f}", f"{F_cmd_path[1]:.4f}", f"{F_cmd_path[2]:.4f}",
            f"{tau_cmd_path[0]:.4f}", f"{tau_cmd_path[1]:.4f}", f"{tau_cmd_path[2]:.4f}",
            f"{F_cmd_base[0]:.4f}", f"{F_cmd_base[1]:.4f}", f"{F_cmd_base[2]:.4f}",
            f"{tau_cmd_base[0]:.4f}", f"{tau_cmd_base[1]:.4f}", f"{tau_cmd_base[2]:.4f}",
            f"{F_meas_base[0]:.4f}", f"{F_meas_base[1]:.4f}", f"{F_meas_base[2]:.4f}",
            f"{tau_meas_base[0]:.4f}", f"{tau_meas_base[1]:.4f}", f"{tau_meas_base[2]:.4f}",
            f"{F_contact[0]:.4f}", f"{F_contact[1]:.4f}", f"{F_contact[2]:.4f}",
            f"{rpy[0]:.4f}", f"{rpy[1]:.4f}", f"{rpy[2]:.4f}",
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
            f"{info.get('r_velocity', 0):.5f}",
            f"{info.get('time_penalty', 0):.5f}",
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