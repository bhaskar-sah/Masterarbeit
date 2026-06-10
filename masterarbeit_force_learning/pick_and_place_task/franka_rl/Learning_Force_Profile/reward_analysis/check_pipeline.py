################################################################################################
# UPDATED: Reward components now match current reward.py + Marko's Verification Check
#   Active: r_progress, r_deviation, r_stability, r_contact, r_velocity
#   Removed: r_alignment, r_position, r_orientation
#   Added: R_path extraction and mathematical transformation proof.
################################################################################################
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import matplotlib.ticker as ticker
import datetime

# =========================================================================
# PATH FIX: Tell Python to look one folder up to find env.py
# =========================================================================
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from stable_baselines3 import PPO
from env import PandaPushTrajectoryEnv

def run_episode_and_collect(env, model, trajectory_type="straight"):
    """Run one episode and collect detailed logs."""

    obs, _ = env.reset(options={"trajectory_type": trajectory_type})
    t_start = env.data.time

    logs = {
        "step": [], "time": [], "progress": [], "deviation": [], "tilt": [], "is_touching": [],
        "action_vx": [], "action_vy": [], "action_wz": [],
        "v_des_mag": [], "v_curr_mag": [], "wz_des": [], "wz_curr": [],
        "F_cmd_mag": [], "F_meas_mag": [],
        "F_cmd_x": [], "F_cmd_y": [], "F_cmd_z": [],
        "F_meas_x": [], "F_meas_y": [], "F_meas_z": [],
        
        # Reward components
        "r_progress": [], "r_deviation": [], "r_stability": [],
        "r_contact": [], "r_alignment": [], "r_velocity": [], 
        "time_penalty": [], "r_total": [],
        "bottle_x": [], "bottle_y": [], "hand_x": [], "hand_y": [],
        
        # ================= NEW ALL LOGS =================
        "F_cmd_ee_x": [], "F_cmd_ee_y": [], "F_cmd_ee_z": [],
        "tau_cmd_ee_x": [], "tau_cmd_ee_y": [], "tau_cmd_ee_z": [],
        "F_cmd_base_x": [], "F_cmd_base_y": [], "F_cmd_base_z": [],
        "tau_cmd_base_x": [], "tau_cmd_base_y": [], "tau_cmd_base_z": [],
        "F_meas_base_x": [], "F_meas_base_y": [], "F_meas_base_z": [],
        "tau_meas_base_x": [], "tau_meas_base_y": [], "tau_meas_base_z": [],
        "roll": [], "pitch": [], "yaw": [],
        "R_path": [],
        "b_hat_x":[],"b_hat_y":[],"b_hat_z":[],
         "t_hat_x": [], "t_hat_y":[], "t_hat_z":[] # Added for Marko's math check
    }

    trajectory = env.traj_manager.trajectory.copy() if env.traj_manager.trajectory is not None else None
    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        sim_time = env.data.time - t_start

        bottle_xy = env.data.xpos[env.bottle_body_id][:2].copy()
        hand_pos = env.data.xpos[env.hand_body_id].copy()

        v_curr = env.push_controller.get_ee_velocity()
        w_curr = env.push_controller.get_ee_angular_velocity()
        v_curr_mag = np.linalg.norm(v_curr[:2])
        wz_curr = w_curr[2]

        # Use total force sent to robot for F_cmd
        F_cmd = env.push_controller.last_F_cmd_base_total if hasattr(env.push_controller, 'last_F_cmd_base_total') else env.push_controller.last_F_cmd
        F_meas = env.contact_manager.get_contact_force()
        F_cmd_mag = np.linalg.norm(F_cmd[:2])
        F_meas_mag = np.linalg.norm(F_meas[:2])

        # New data extraction
        F_cmd_ee = env.push_controller.last_F_cmd_ee
        tau_cmd_ee = env.push_controller.last_tau_cmd_ee
        F_cmd_base = env.push_controller.last_F_cmd_base_pure
        tau_cmd_base = env.push_controller.last_tau_cmd_base
        F_meas_base, tau_meas_base = env.push_controller.get_measured_wrench()
        rpy = env.push_controller.get_rpy()
        R_path = env.push_controller.last_R_path.copy() if hasattr(env.push_controller, 'last_R_path') else np.eye(3)

        # 
        t_hat = env.push_controller.last_t_hat
        b_hat = env.push_controller.last_b_hat

        # ================= APPEND =================
        logs["step"].append(step)
        logs["time"].append(sim_time)
        logs["progress"].append(info.get("progress", 0.0) * 100)
        logs["deviation"].append(info.get("deviation", 0.0) * 100)
        logs["tilt"].append(info.get("bottle_tilt", 1.0))
        logs["is_touching"].append(1.0 if info.get("is_touching", False) else 0.0)
        logs["action_vx"].append(action[0])
        logs["action_vy"].append(action[1])
        logs["action_wz"].append(action[2])
        logs["v_curr_mag"].append(v_curr_mag * 100)
        logs["wz_curr"].append(wz_curr)
        logs["F_cmd_mag"].append(F_cmd_mag)
        logs["F_meas_mag"].append(F_meas_mag)
        logs["F_cmd_x"].append(F_cmd[0])
        logs["F_cmd_y"].append(F_cmd[1])
        logs["F_cmd_z"].append(F_cmd[2])
        logs["F_meas_x"].append(F_meas[0])
        logs["F_meas_y"].append(F_meas[1])
        logs["F_meas_z"].append(F_meas[2])

        logs["r_progress"].append(info.get("r_progress", 0.0))
        logs["r_deviation"].append(info.get("r_deviation", 0.0))
        logs["r_stability"].append(info.get("r_stability", 0.0))
        logs["r_contact"].append(info.get("r_contact", 0.0))
        logs["r_alignment"].append(info.get("r_alignment", 0.0))
        logs["r_velocity"].append(info.get("r_velocity", 0.0))
        logs["time_penalty"].append(info.get("time_penalty", 0.0))
        logs["r_total"].append(reward)

        logs["bottle_x"].append(float(bottle_xy[0]))
        logs["bottle_y"].append(float(bottle_xy[1]))
        logs["hand_x"].append(float(hand_pos[0]))
        logs["hand_y"].append(float(hand_pos[1]))

        # ================= APPEND ALL NEW =================
        logs["F_cmd_ee_x"].append(F_cmd_ee[0])
        logs["F_cmd_ee_y"].append(F_cmd_ee[1])
        logs["F_cmd_ee_z"].append(F_cmd_ee[2])
        logs["tau_cmd_ee_x"].append(tau_cmd_ee[0])
        logs["tau_cmd_ee_y"].append(tau_cmd_ee[1])
        logs["tau_cmd_ee_z"].append(tau_cmd_ee[2])
        
        logs["F_cmd_base_x"].append(F_cmd_base[0])
        logs["F_cmd_base_y"].append(F_cmd_base[1])
        logs["F_cmd_base_z"].append(F_cmd_base[2])
        logs["tau_cmd_base_x"].append(tau_cmd_base[0])
        logs["tau_cmd_base_y"].append(tau_cmd_base[1])
        logs["tau_cmd_base_z"].append(tau_cmd_base[2])
        
        logs["F_meas_base_x"].append(F_meas_base[0])
        logs["F_meas_base_y"].append(F_meas_base[1])
        logs["F_meas_base_z"].append(F_meas_base[2])
        logs["tau_meas_base_x"].append(tau_meas_base[0])
        logs["tau_meas_base_y"].append(tau_meas_base[1])
        logs["tau_meas_base_z"].append(tau_meas_base[2])
        
        logs["roll"].append(rpy[0])
        logs["pitch"].append(rpy[1])
        logs["yaw"].append(rpy[2])
        logs["R_path"].append(R_path)

        logs["b_hat_x"].append(b_hat[0])
        logs["b_hat_y"].append(b_hat[1])
        logs["b_hat_z"].append(b_hat[2])
        logs["t_hat_x"].append(t_hat[0])
        logs["t_hat_y"].append(t_hat[1])
        logs["t_hat_z"].append(t_hat[2])



        step += 1

    for key in logs:
        if key != "R_path":
            logs[key] = np.array(logs[key])

    print(f"  Episode duration (real sim time): {logs['time'][-1]:.2f} s")
    print(f"  Total control steps: {len(logs['time'])}")
    print(f"  Mean control_dt: {(logs['time'][-1] / max(len(logs['time']) - 1, 1)):.4f} s")

    # =======================================================
    # MARKO'S MATHEMATICAL VERIFICATION CHECK
    # =======================================================
    math_errors = []
    for i in range(len(logs["time"])):
        f_ee = np.array([logs["F_cmd_ee_x"][i], logs["F_cmd_ee_y"][i], logs["F_cmd_ee_z"][i]])
        f_base = np.array([logs["F_cmd_base_x"][i], logs["F_cmd_base_y"][i], logs["F_cmd_base_z"][i]])
        r_mat = logs["R_path"][i]
        
        # Calculate expected base force
        f_base_calculated = r_mat @ f_ee
        error = np.linalg.norm(f_base - f_base_calculated)
        math_errors.append(error)
        
    max_error = np.max(math_errors)
    print("\n" + "="*60)
    print("PIPELINE MATHEMATICAL VERIFICATION (MARKO'S CHECK)")
    print("="*60)
    print(f"Maximum Transformation Error: {max_error:.8f} Newtons")
    if max_error < 1e-5:
        print("STATUS: PASSED. F_cmd_base perfectly matches R_path @ F_cmd_ee.")
    else:
        print("STATUS: FAILED. There is a mathematical mismatch in the controller.")
    print("="*60 + "\n")

    return logs, trajectory, info


def plot_rewards(logs, trajectory, info, env, save_dir=None):
    """Create separate diagnostic plots, one per figure."""
    time_axis = logs["time"]
    t_end = float(time_axis[-1])
    t_zoom = min(1.0, t_end * 0.1)
    t_end_plot = np.ceil(t_end * 2) / 2

    figures = []
    suptitle = (
        f"Final Progress: {logs['progress'][-1]:.1f}% | "
        f"Final Dev: {logs['deviation'][-1]:.1f}cm | "
        f"Success: {info.get('is_success', False)}"
    )

    # ==================== 1. Individual Rewards ====================
    reward_specs = [
        ("r_progress",  "Progress (+)",         "green"),
        ("r_deviation", "Deviation (-)",        "red"),
        ("r_stability", "Stability (-)",        "orange"),
        ("r_contact",   "Contact (-)",          "brown"),
        ("r_alignment", "Alignment (+)",        "cyan"),
        ("r_velocity",  "Velocity Penalty (-)", "purple"),
        ("time_penalty","Time Penalty (-)",     "gray"),
        ("r_total",     "Total Reward",         "black"),
    ]

    for key, title, color in reward_specs:
        fig_indiv, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'hspace': 0.4})

        ax_full.plot(time_axis, logs[key], color=color, linewidth=1.5)
        ax_full.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_full.set_xlim(0.0, t_end_plot)
        ax_full.set_title(f"{title} - Full")
        ax_full.grid(True, alpha=0.3)

        ax_zoom.plot(time_axis, logs[key], color=color, linewidth=1.5)
        ax_zoom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_zoom.set_xlim(0.0, t_zoom)
        ax_zoom.set_title(f"{title} - Zoom 0.0 - {t_zoom:.2f}s")
        ax_zoom.grid(True, alpha=0.3)

        fig_indiv.supxlabel("Simulation Time (seconds)", fontsize=12)
        fig_indiv.supylabel("Reward Value", fontsize=12)
        fig_indiv.suptitle(f"{title} Profile | {suptitle}", fontsize=14, fontweight='bold')
        figures.append((f"reward_{key}", fig_indiv))

    # ==================== 1b. Combined Rewards (All-in-One Plot) ====================
    fig1b, (ax1b_full, ax1b_zoom) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.4})

    ax1b_full_twin = ax1b_full.twinx()
    ax1b_zoom_twin = ax1b_zoom.twinx()

    for key, title, color in reward_specs:
        if key == "r_total":
            ax1b_full_twin.plot(time_axis, logs[key], color=color, linewidth=2, linestyle='--', label=title)
            ax1b_zoom_twin.plot(time_axis, logs[key], color=color, linewidth=2, linestyle='--', label=title)
        else:
            ax1b_full.plot(time_axis, logs[key], color=color, linewidth=1.5, label=title)
            ax1b_zoom.plot(time_axis, logs[key], color=color, linewidth=1.5, label=title)

    ax1b_full.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1b_full.set_xlim(0.0, t_end_plot)
    ax1b_full.set_title("Combined Reward Components Overlap - Full")
    ax1b_full.grid(True, alpha=0.3)
    ax1b_full.set_ylabel("Component Values")
    ax1b_full_twin.set_ylabel("Total Reward", color='black')

    ax1b_zoom.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1b_zoom.set_xlim(0.0, t_zoom)
    ax1b_zoom.set_title(f"Combined Reward Components Overlap - Zoom 0.0 - {t_zoom:.2f}s")
    ax1b_zoom.grid(True, alpha=0.3)
    ax1b_zoom.set_ylabel("Component Values")
    ax1b_zoom_twin.set_ylabel("Total Reward", color='black')

    lines, labels = ax1b_full.get_legend_handles_labels()
    lines2, labels2 = ax1b_full_twin.get_legend_handles_labels()
    ax1b_full.legend(lines + lines2, labels + labels2, loc='center left', bbox_to_anchor=(1.05, 0.5))

    lines_z, labels_z = ax1b_zoom.get_legend_handles_labels()
    lines2_z, labels2_z = ax1b_zoom_twin.get_legend_handles_labels()
    ax1b_zoom.legend(lines_z + lines2_z, labels_z + labels2_z, loc='center left', bbox_to_anchor=(1.05, 0.5))

    fig1b.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig1b.suptitle(f"Combined Rewards Overview | {suptitle}", fontsize=14, fontweight='bold')
    fig1b.tight_layout(rect=[0, 0, 0.85, 1]) 
    figures.append(("rewards_combined", fig1b))

    # ==================== 2. Raw Actions ====================
    fig2, axes2 = plt.subplots(6, 1, figsize=(16, 25), gridspec_kw={'hspace': 1.2})
    
    action_specs = [
        ("action_vx", "Fx Command (Forward/Backward Force)", "blue"),
        ("action_vy", "Fy Command (Lateral Force)", "green"),
        ("action_wz", "Tz Command (Yaw Torque)", "purple"),
    ]
    for i, (key, title, color) in enumerate(action_specs):
        ax_full = axes2[i * 2]
        ax_full.plot(time_axis, logs[key], color=color, linewidth=1.2)
        ax_full.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_full.set_xlim(0.0, t_end_plot)
        ax_full.set_ylim(-1.2, 1.2)
        ax_full.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_full.set_title(f"{title} - Full")
        ax_full.grid(True, alpha=0.3)

        ax_zoom = axes2[i * 2 + 1]
        ax_zoom.plot(time_axis, logs[key], color=color, linewidth=1.2)
        ax_zoom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_zoom.set_xlim(0.0, t_zoom)
        ax_zoom.set_ylim(-1.2, 1.2)
        ax_zoom.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_zoom.set_title(f"{title} - Zoom 0.0 - {t_zoom:.2f}s")
        ax_zoom.grid(True, alpha=0.3)

    fig2.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig2.supylabel("Action Command", fontsize=12)
    fig2.suptitle("Raw Neural Network Actions [-1, 1]", fontsize=14, fontweight='bold')
    figures.append(("actions", fig2))

    # ==================== 3. Forces (X, Y, Z Components) ====================
    fig3, axes3 = plt.subplots(6, 1, figsize=(16, 22), gridspec_kw={'hspace': 0.8})
    force_specs = [
        ("F_cmd_x", "F_meas_x", "X-Axis Force (Forward/Back)"),
        ("F_cmd_y", "F_meas_y", "Y-Axis Force (Left/Right)"),
        ("F_cmd_z", "F_meas_z", "Z-Axis Force (Up/Down)"),
    ]
    for i, (cmd_key, meas_key, title) in enumerate(force_specs):
        ax_full = axes3[i * 2]
        ax_full.plot(time_axis, logs[cmd_key], label="F_cmd (Command)", color="red", linewidth=1.2)
        # Multiply measured force by -1 to mirror it appropriately as discussed
        ax_full.plot(time_axis, -1.0 * logs[meas_key], label="F_meas (Actual)", color="blue", linewidth=1.2, alpha=0.7)
        ax_full.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_full.set_xlim(0.0, t_end_plot)
        ax_full.set_title(f"{title} - Full")
        ax_full.legend(loc='upper right', fontsize=9)
        ax_full.grid(True, alpha=0.3)

        ax_zoom = axes3[i * 2 + 1]
        ax_zoom.plot(time_axis, logs[cmd_key], color="red", linewidth=1.5)
        ax_zoom.plot(time_axis, -1.0 * logs[meas_key], color="blue", linewidth=1.5, alpha=0.7)
        ax_zoom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_zoom.set_xlim(0.0, t_zoom)
        ax_zoom.set_title(f"{title} - Zoom: 0.0 - {t_zoom:.2f}s")
        ax_zoom.grid(True, alpha=0.3)

    fig3.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig3.supylabel("Force (Newtons)", fontsize=12)
    fig3.suptitle("Force Components Profile (X, Y, Z)", fontsize=14, fontweight='bold')
    figures.append(("forces_xyz", fig3))

    # ==================== 3b. Total Planar Force Magnitude (XY) ====================
    fig3b, (ax3b, ax3bz) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'hspace': 0.4})
 
    def plot_force_mag_on(ax):
        ax.plot(time_axis, logs["F_cmd_mag"],  label="||F_cmd||  (Commanded, XY)",
                color="red",  linewidth=1.5)
        ax.plot(time_axis, logs["F_meas_mag"], label="||F_meas|| (Contact reaction, XY)",
                color="blue", linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
 
    plot_force_mag_on(ax3b)
    ax3b.set_xlim(0.0, t_end_plot)
    ax3b.set_title("Total Planar Force Magnitude (XY plane) - Full")
    ax3b.legend(loc='upper right', fontsize=9)
 
    plot_force_mag_on(ax3bz)
    ax3bz.set_xlim(0.0, t_zoom)
    ax3bz.set_title(f"Total Planar Force Magnitude (XY plane) - Zoom: 0.0 - {t_zoom:.2f}s")
    ax3bz.legend(loc='upper right', fontsize=9)
 
    fig3b.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig3b.supylabel("Force Magnitude (Newtons)", fontsize=12)
    fig3b.suptitle("Total Force Profile (Commanded vs Measured)", fontsize=14, fontweight='bold')
    figures.append(("forces_total", fig3b))

    # ==================== 4. Linear Velocity ====================
    fig4, (ax4, ax4z) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'hspace': 0.4})
    def plot_linvel_on(ax):
        ax.plot(time_axis, logs["v_curr_mag"], label="Actual v (cm/s)",  color="green", linewidth=1.5)
        ax.axhline(y=env.config.v_target_limit * 100, color='red', linestyle=':', label="Penalty Threshold")

    plot_linvel_on(ax4)
    ax4.set_title("Actual Velocity magnitude (Eucl. Norm) - Full")
    ax4.set_xlim(0.0, t_end_plot)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plot_linvel_on(ax4z)
    ax4z.set_xlim(0.0, t_zoom)
    ax4z.set_title(f"Actual Velocity magnitude (Eucl. Norm) - Zoom: 0.0 - {t_zoom:.2f}s")
    ax4z.legend(loc='upper right', fontsize=9)
    ax4z.grid(True, alpha=0.3)

    fig4.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig4.supylabel("Velocity (cm/s)", fontsize=12)
    fig4.suptitle("Planar Velocity Tracking (Mag)", fontsize=14, fontweight='bold')
    figures.append(("linear_velocity", fig4))

    # ==================== 5. Angular Velocity ====================
    fig5, axes5 = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'hspace': 0.6})
    
    axes5[0].plot(time_axis, logs["wz_curr"], color="purple", linewidth=1.5, label="Actual wz")
    axes5[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes5[0].set_xlim(0.0, t_end_plot)
    axes5[0].set_title("Actual wz - Full")
    axes5[0].grid(True, alpha=0.3)
    axes5[0].legend()

    axes5[1].plot(time_axis, logs["wz_curr"], color="purple", linewidth=1.5, label="Actual wz")
    axes5[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes5[1].set_xlim(0.0, t_zoom)
    axes5[1].set_title(f"Actual wz - Zoom 0.0 - {t_zoom:.2f}s")
    axes5[1].grid(True, alpha=0.3)
    axes5[1].legend()

    fig5.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig5.supylabel("Angular Velocity (rad/s)", fontsize=12)
    fig5.suptitle("Wrist Yaw Velocity Tracking", fontsize=14, fontweight='bold')
    figures.append(("angular_velocity", fig5))

    # ==================== 6. Task Progress & Deviation ====================
    fig6, ax6 = plt.subplots(figsize=(16, 5))
    ax6.plot(time_axis, logs["progress"], label="Progress (%)", color="green", linewidth=2)
    ax6b = ax6.twinx()
    ax6b.plot(time_axis, logs["deviation"], label="Deviation (cm)", color="red", linewidth=2)
    ax6b.axhline(y=5.0, color='red', linestyle=':', alpha=0.5, label="Deviation Tolerance (5cm)")
    ax6.set_xlim(0.0, t_end_plot)
    ax6.set_title("Task Performance: Progress vs Deviation")
    ax6.set_xlabel("Simulation Time (seconds)")
    ax6.set_ylabel("Progress (%)", color="green")
    ax6b.set_ylabel("Deviation (cm)", color="red")
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)
    ax6.grid(True, alpha=0.3)
    figures.append(("progress_deviation", fig6))

    # ==================== 7. Base Forces ====================
    fig8, axes8 = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.6})
    force_base_specs = [
        ("F_cmd_base_x", "F_meas_base_x", "Base Force X (Forward/Back)"),
        ("F_cmd_base_y", "F_meas_base_y", "Base Force Y (Left/Right)"),
        ("F_cmd_base_z", "F_meas_base_z", "Base Force Z (Up/Down)"),
    ]
    for i, (cmd, meas, title) in enumerate(force_base_specs):
        axes8[i].plot(time_axis, logs[cmd], label="F_cmd_base (Pure Rotation)", color="red", linewidth=1.5)
        # Note: Inverted measured force for visual alignment
        axes8[i].plot(time_axis, -1.0 * logs[meas], label="F_meas_base (Actual Motor Output)", color="blue", alpha=0.7, linewidth=1.5)
        axes8[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes8[i].set_xlim(0.0, t_end_plot)
        axes8[i].set_title(title)
        axes8[i].legend(loc='upper right')
        axes8[i].grid(True, alpha=0.3)
    fig8.suptitle("Base Forces (Command vs Measurement)", fontsize=14, fontweight='bold')
    figures.append(("base_forces", fig8))

    # ==================== 8. Base Torques ====================
    fig9, axes9 = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.6})
    tau_base_specs = [
        ("tau_cmd_base_x", "tau_meas_base_x", "Base Torque X (Roll)"),
        ("tau_cmd_base_y", "tau_meas_base_y", "Base Torque Y (Pitch)"),
        ("tau_cmd_base_z", "tau_meas_base_z", "Base Torque Z (Yaw)"),
    ]
    for i, (cmd, meas, title) in enumerate(tau_base_specs):
        axes9[i].plot(time_axis, logs[cmd], label="tau_cmd_base", color="red", linewidth=1.5)
        axes9[i].plot(time_axis, -1.0 * logs[meas], label="tau_meas_base (Actual Motor Output)", color="blue", alpha=0.7, linewidth=1.5)
        axes9[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes9[i].set_xlim(0.0, t_end_plot)
        axes9[i].set_title(title)
        axes9[i].legend(loc='upper right')
        axes9[i].grid(True, alpha=0.3)
    fig9.suptitle("Base Torques (Command vs Measurement)", fontsize=14, fontweight='bold')
    figures.append(("base_torques", fig9))

    # ==================== 9. EE vs Base Commands ====================
    fig10, axes10 = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'hspace': 0.4})
    axes10[0].plot(time_axis, logs["F_cmd_ee_x"], label="EE Local Fx", color="purple")
    axes10[0].plot(time_axis, logs["F_cmd_base_x"], label="Base World Fx", color="red", alpha=0.7)
    axes10[0].set_title("Force Command: Local End-Effector vs World Base (X-Axis)")
    axes10[0].legend()
    axes10[0].set_xlim(0.0, t_end_plot)
    axes10[0].grid(True, alpha=0.3)

    axes10[1].plot(time_axis, logs["tau_cmd_ee_z"], label="EE Local Tau_z (Yaw)", color="purple")
    axes10[1].plot(time_axis, logs["tau_cmd_base_z"], label="Base World Tau_z", color="red", alpha=0.7)
    axes10[1].set_title("Torque Command: Local End-Effector vs World Base (Z-Axis)")
    axes10[1].legend()
    axes10[1].set_xlim(0.0, t_end_plot)
    axes10[1].grid(True, alpha=0.3)
    fig10.suptitle("Frame Rotation Verification", fontsize=14, fontweight='bold')
    figures.append(("frame_commands", fig10))

    # ==================== 10. RPY Angles ====================
    fig11, axes11 = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.6})
    rpy_specs = [
        ("roll", "Roll (Degrees)", "orange"),
        ("pitch", "Pitch (Degrees)", "green"),
        ("yaw", "Yaw (Degrees)", "purple"),
    ]
    for i, (key, title, color) in enumerate(rpy_specs):
        axes11[i].plot(time_axis, np.degrees(logs[key]), color=color, linewidth=1.5)
        axes11[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes11[i].set_xlim(0.0, t_end_plot)
        axes11[i].set_title(title)
        axes11[i].grid(True, alpha=0.3)
    fig11.suptitle("End-Effector Orientation (RPY from Rotation Matrix)", fontsize=14, fontweight='bold')
    figures.append(("rpy", fig11))

    # ==================== 11. Bird's Eye View (MUST BE LAST) ====================
    fig7, ax7 = plt.subplots(figsize=(12, 10))
    if trajectory is not None:
        ax7.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=3,
                 label="Desired Trajectory", zorder=1)
        n_arrows = 6
        n_points = len(trajectory)
        indices = np.linspace(0, n_points - 10, n_arrows, dtype=int)
        for idx in indices:
            dx = trajectory[idx + 5, 0] - trajectory[idx, 0]
            dy = trajectory[idx + 5, 1] - trajectory[idx, 1]
            ax7.annotate(
                '',
                xy=(trajectory[idx, 0] + dx * 0.3, trajectory[idx, 1] + dy * 0.3),
                xytext=(trajectory[idx, 0], trajectory[idx, 1]),
                arrowprops=dict(
                    arrowstyle='-|>,head_length=1.2,head_width=0.8',
                    color='red', lw=2, alpha=0.9,
                ),
                zorder=2,
            )
 
    ax7.plot(logs["bottle_x"], logs["bottle_y"], 'b-', linewidth=2,
             label="Actual Bottle Path", zorder=3)
 
    step_size = max(1, len(logs["bottle_x"]) // 30)
    ax7.quiver(
        logs["bottle_x"][::step_size],
        logs["bottle_y"][::step_size],
        logs["F_meas_x"][::step_size],
        logs["F_meas_y"][::step_size],
        color='purple', angles='xy', scale_units='xy',
        scale=150, width=0.005, alpha=0.6,
        label="Measured Force Vectors (robot -> bottle)",
        zorder=4
    )
 
    ax7.plot(logs["hand_x"], logs["hand_y"], 'g-', linewidth=1, alpha=0.5,
             label="Hand Path", zorder=1)
    ax7.scatter(logs["bottle_x"][0],  logs["bottle_y"][0],
                c='black', s=120, marker='o', zorder=4, label="Start")
    ax7.scatter(logs["bottle_x"][-1], logs["bottle_y"][-1],
                c='green', s=120, marker='o', zorder=4, label="End")
 
    mid_idx = len(logs["bottle_x"]) // 2
    if mid_idx + 5 < len(logs["bottle_x"]):
        dx = logs["bottle_x"][mid_idx + 5] - logs["bottle_x"][mid_idx]
        dy = logs["bottle_y"][mid_idx + 5] - logs["bottle_y"][mid_idx]
        ax7.annotate(
            '',
            xy=(logs["bottle_x"][mid_idx] + dx * 3, logs["bottle_y"][mid_idx] + dy * 3),
            xytext=(logs["bottle_x"][mid_idx], logs["bottle_y"][mid_idx]),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2.5),
            zorder=5,
        )
 
    ax7.set_title("Bird's Eye View: X-Y Trajectory Map")
    ax7.set_xlabel("X Position (m)")
    ax7.set_ylabel("Y Position (m)")
    ax7.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        frameon=True,
    )
    ax7.set_aspect('equal')
    
    # We append the Bird's Eye view LAST so the time-formatting loop ignores it!
    figures.append(("trajectory", fig7))

    # 
    fig12, axes12 = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.6})
    normal_specs = [
        ("t_hat_x", "orange"),
        ("t_hat_y", "green"),
        ("t_hat_z", "purple"),
    ]
    for i, (key, color) in enumerate(normal_specs):
        axes12[i].plot(time_axis, np.degrees(logs[key]), color=color, linewidth=1.5)
        axes12[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes12[i].set_xlim(0.0, t_end_plot)
        axes12[i].grid(True, alpha=0.3)
    fig12.suptitle("plotting normal", fontsize=14, fontweight='bold')
    figures.append(("normals", fig12))
 
    # ==================== TICK FORMATTING (data-driven) ====================
    # All figures except the trajectory (bird's-eye view) get time-axis tick formatting.
    for _, fig in figures[:-1]:
        for ax in fig.axes:
            xlim = ax.get_xlim()
            if xlim[1] - xlim[0] <= t_zoom + 0.05:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
            else:
                step = max(0.5, round(t_end_plot / 10, 1))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax.tick_params(axis='x', labelsize=8, rotation=45)
 
    for _, fig in figures:
        fig.tight_layout()

    # Save or show
    if save_dir:
        print("\nSaving plots...")
        for name, fig in figures:
            # Create the exact file path inside the new timestamped folder
            fig_path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved: {name}.png")
        print("All plots saved successfully!")
    else:
        plt.show()

    # plt.show()
    return figures

def main():
    parser = argparse.ArgumentParser(description="Plot diagnostic dashboard for pure velocity control.")
    parser.add_argument("--model_dir", type=str, required=True, help="Name of the model directory (e.g. model_name/phase_folder)")
    parser.add_argument("--trajectory_type", type=str, default="s_curve",
                        choices=["straight", "curved", "curved_opposite", "s_curve"])
    parser.add_argument("--save", action="store_true", help="Save the plot to the reward_results directory")
    args = parser.parse_args()

    env = PandaPushTrajectoryEnv(render_mode=None, trajectory_type=args.trajectory_type)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_path = os.path.join(parent_dir, "saved_models_new", args.model_dir, "best_model.zip")

    print(f"\nLoading best model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Could not find best_model.zip at {model_path}")
        exit()

    model = PPO.load(model_path, env=env)
    print(f"Running episode with trajectory: {args.trajectory_type}...")
    logs, trajectory, info = run_episode_and_collect(env, model, args.trajectory_type)

    print(f"\nEpisode complete! Generating plots...")
    print(f"  Progress: {logs['progress'][-1]:.1f}%")
    print(f"  Final deviation: {logs['deviation'][-1]:.1f}cm")

    save_dir = None
    if args.save:
        # Get exact current time formatted as YYYY-MM-DD_HH-MM-SS
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Clean the model name for folder creation
        clean_model_name = args.model_dir.replace("/", "_").replace("\\", "_")
        
        # Create the master results directory
        analysis_dir = os.path.join(current_dir, "reward_results")
        
        # Create the specific timestamped subfolder for THIS run
        folder_name = f"Run_{clean_model_name}_{args.trajectory_type}_{timestamp}"
        save_dir = os.path.join(analysis_dir, folder_name)
        
        # Make the directory
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nCreated new directory for results:\n-> {save_dir}")

    # Pass save_dir instead of save_path
    plot_rewards(logs, trajectory, info, env, save_dir=save_dir)
    env.close()

if __name__ == "__main__":
    main()