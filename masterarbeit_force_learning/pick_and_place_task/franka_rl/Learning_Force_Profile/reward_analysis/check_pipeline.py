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
        "action_0": [], "action_1": [], "action_2": [],
        "ee_vel_world_mag": [], "ee_ang_vel_world_z": [],
        
        # Reward components
        "r_progress": [], "r_deviation": [], "r_stability": [],
        "r_contact": [], "r_alignment": [], "r_velocity": [], 
        "time_penalty": [], "r_total": [],
        "bottle_x": [], "bottle_y": [], "bottle_z": [],
        "ee_pos_x": [], "ee_pos_y": [], "ee_pos_z": [],
        
        # ================= Forces =================
        "F_path_cmd_x": [], "F_path_cmd_y": [], "F_path_cmd_z": [],
        "tau_path_cmd_x": [], "tau_path_cmd_y": [], "tau_path_cmd_z": [],
        "F_path_meas_x": [], "F_path_meas_y": [], "F_path_meas_z": [],
        "tau_path_meas_x": [], "tau_path_meas_y": [], "tau_path_meas_z": [],
        "F_world_cmd_x": [], "F_world_cmd_y": [], "F_world_cmd_z": [],
        "tau_world_cmd_x": [], "tau_world_cmd_y": [], "tau_world_cmd_z": [],
        "F_world_meas_x": [], "F_world_meas_y": [], "F_world_meas_z": [],
        "tau_world_meas_x": [], "tau_world_meas_y": [], "tau_world_meas_z": [],
        "F_path_cmd_mag": [], "F_world_meas_contact_mag": [],
        "F_world_meas_contact_x": [], "F_world_meas_contact_y": [], "F_world_meas_contact_z": [],

        # ====== Orientation =======
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

        bottle_pos = env.data.xpos[env.bottle_body_id].copy()
        ee_pos = env.data.site_xpos[env.gripper_site_id].copy()

        ee_vel_world = env.push_controller.get_ee_vel_world()
        ee_ang_vel_world = env.push_controller.get_ee_ang_vel_world()
        ee_vel_world_mag = np.linalg.norm(ee_vel_world)
        ee_ang_vel_world_z = ee_ang_vel_world[2]

        # Load Force Data
        F_path_cmd = env.push_controller.last_F_path_cmd
        tau_path_cmd = env.push_controller.last_tau_path_cmd
        F_world_cmd = env.push_controller.last_F_world_cmd
        tau_world_cmd = env.push_controller.last_tau_world_cmd
        F_world_meas, tau_world_meas = env.push_controller.get_measured_wrench_world()

        # Force Amplitudes
        F_path_cmd_mag = np.linalg.norm(F_path_cmd)
        F_world_meas_contact = env.contact_manager.get_contact_force()
        F_world_meas_contact_mag = np.linalg.norm(F_world_meas_contact)

        # Load orientations
        rpy = env.push_controller.get_rpy()
        R_path = env.push_controller.last_R_path.copy()

        # Load path vectors
        t_hat = env.push_controller.last_t_hat
        b_hat = env.push_controller.last_b_hat

        # Forces in local frame (path frame)
        F_path_meas = R_path.T @ F_world_meas
        tau_path_meas = R_path.T @ (tau_world_meas - np.cross(ee_pos,F_world_meas))

        # ================= APPEND =================
        logs["step"].append(step)
        logs["time"].append(sim_time)
        logs["progress"].append(info.get("progress", 0.0) * 100)
        logs["deviation"].append(info.get("deviation", 0.0) * 100)
        logs["tilt"].append(info.get("bottle_tilt", 1.0))
        logs["is_touching"].append(1.0 if info.get("is_touching", False) else 0.0)
        logs["action_0"].append(action[0])
        logs["action_1"].append(action[1])
        logs["action_2"].append(action[2])
        logs["ee_vel_world_mag"].append(ee_vel_world_mag * 100)
        logs["ee_ang_vel_world_z"].append(ee_ang_vel_world_z)

        logs["r_progress"].append(info.get("r_progress", 0.0))
        logs["r_deviation"].append(info.get("r_deviation", 0.0))
        logs["r_stability"].append(info.get("r_stability", 0.0))
        logs["r_contact"].append(info.get("r_contact", 0.0))
        logs["r_alignment"].append(info.get("r_alignment", 0.0))
        logs["r_velocity"].append(info.get("r_velocity", 0.0))
        logs["time_penalty"].append(info.get("time_penalty", 0.0))
        logs["r_total"].append(reward)

        logs["bottle_x"].append(float(bottle_pos[0]))
        logs["bottle_y"].append(float(bottle_pos[1]))
        logs["bottle_z"].append(float(bottle_pos[2]))
        logs["ee_pos_x"].append(float(ee_pos[0]))
        logs["ee_pos_y"].append(float(ee_pos[1]))
        logs["ee_pos_z"].append(float(ee_pos[2]))

        # ================= Forces =================
        logs["F_path_cmd_x"].append(F_path_cmd[0])
        logs["F_path_cmd_y"].append(F_path_cmd[1])
        logs["F_path_cmd_z"].append(F_path_cmd[2])
        logs["tau_path_cmd_x"].append(tau_path_cmd[0])
        logs["tau_path_cmd_y"].append(tau_path_cmd[1])
        logs["tau_path_cmd_z"].append(tau_path_cmd[2])

        logs["F_path_meas_x"].append(F_path_meas[0])
        logs["F_path_meas_y"].append(F_path_meas[1])
        logs["F_path_meas_z"].append(F_path_meas[2])
        logs["tau_path_meas_x"].append(tau_path_meas[0])
        logs["tau_path_meas_y"].append(tau_path_meas[1])
        logs["tau_path_meas_z"].append(tau_path_meas[2])
        
        logs["F_world_cmd_x"].append(F_world_cmd[0])
        logs["F_world_cmd_y"].append(F_world_cmd[1])
        logs["F_world_cmd_z"].append(F_world_cmd[2])
        logs["tau_world_cmd_x"].append(tau_world_cmd[0])
        logs["tau_world_cmd_y"].append(tau_world_cmd[1])
        logs["tau_world_cmd_z"].append(tau_world_cmd[2])
        
        logs["F_world_meas_x"].append(F_world_meas[0])
        logs["F_world_meas_y"].append(F_world_meas[1])
        logs["F_world_meas_z"].append(F_world_meas[2])
        logs["tau_world_meas_x"].append(tau_world_meas[0])
        logs["tau_world_meas_y"].append(tau_world_meas[1])
        logs["tau_world_meas_z"].append(tau_world_meas[2])

        logs["F_path_cmd_mag"].append(F_path_cmd_mag)
        logs["F_world_meas_contact_mag"].append(F_world_meas_contact_mag)
        logs["F_world_meas_contact_x"].append(F_world_meas_contact[0])
        logs["F_world_meas_contact_y"].append(F_world_meas_contact[1])
        logs["F_world_meas_contact_z"].append(F_world_meas_contact[2])

        # ======= Orientation =======
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

    # # =======================================================
    # # MARKO'S MATHEMATICAL VERIFICATION CHECK
    # # =======================================================
    # math_errors = []
    # for i in range(len(logs["time"])):
    #     f_ee = np.array([logs["F_cmd_ee_x"][i], logs["F_cmd_ee_y"][i], logs["F_cmd_ee_z"][i]])
    #     f_base = np.array([logs["F_cmd_base_x"][i], logs["F_cmd_base_y"][i], logs["F_cmd_base_z"][i]])
    #     r_mat = logs["R_path"][i]
    #
    #     # Calculate expected base force
    #     f_base_calculated = r_mat @ f_ee
    #     error = np.linalg.norm(f_base - f_base_calculated)
    #     math_errors.append(error)
    #
    # max_error = np.max(math_errors)
    # print("\n" + "="*60)
    # print("PIPELINE MATHEMATICAL VERIFICATION (MARKO'S CHECK)")
    # print("="*60)
    # print(f"Maximum Transformation Error: {max_error:.8f} Newtons")
    # if max_error < 1e-5:
    #     print("STATUS: PASSED. F_world_cmd perfectly matches R_path @ F_path_cmd.")
    # else:
    #     print("STATUS: FAILED. There is a mathematical mismatch in the controller.")
    # print("="*60 + "\n")

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
    figActions, axesActions = plt.subplots(6, 1, figsize=(16, 25), gridspec_kw={'hspace': 1.2})
    
    action_specs = [
        ("action_0", "Raw Action 0", "blue"),
        ("action_1", "Raw Action 1", "green"),
        ("action_2", "Raw Action 2", "purple"),
    ]
    for i, (key, title, color) in enumerate(action_specs):
        ax_full = axesActions[i * 2]
        ax_full.plot(time_axis, logs[key], color=color, linewidth=1.2)
        ax_full.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_full.set_xlim(0.0, t_end_plot)
        ax_full.set_ylim(-1.2, 1.2)
        ax_full.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_full.set_title(f"{title} - Full")
        ax_full.grid(True, alpha=0.3)

        ax_zoom = axesActions[i * 2 + 1]
        ax_zoom.plot(time_axis, logs[key], color=color, linewidth=1.2)
        ax_zoom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_zoom.set_xlim(0.0, t_zoom)
        ax_zoom.set_ylim(-1.2, 1.2)
        ax_zoom.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_zoom.set_title(f"{title} - Zoom 0.0 - {t_zoom:.2f}s")
        ax_zoom.grid(True, alpha=0.3)

    figActions.supxlabel("Simulation Time (seconds)", fontsize=12)
    figActions.supylabel("Action Command", fontsize=12)
    figActions.suptitle("Raw RL Actions [-1, 1]", fontsize=14, fontweight='bold')
    figures.append(("actions", figActions))

    # ==================== 3. Forces (X, Y, Z Components) ====================
    figForcesWorld, axesForcesWorld = plt.subplots(6, 1, figsize=(16, 22), gridspec_kw={'hspace': 0.8})
    force_specs = [
        ("F_world_cmd_x", "F_world_meas_x", "F_world_meas_contact_x", "Force World-x"),
        ("F_world_cmd_y", "F_world_meas_y", "F_world_meas_contact_y", "Force World-y"),
        ("F_world_cmd_z", "F_world_meas_z", "F_world_meas_contact_z", "Force World-z"),
    ]
    for i, (cmd_key, meas_key, meas_contact_key, title) in enumerate(force_specs):
        ax_full = axesForcesWorld[i * 2]
        ax_full.plot(time_axis, logs[cmd_key], label="F_cmd", color="red", linewidth=1.2)
        # Multiply measured force by -1 to mirror it appropriately as discussed
        ax_full.plot(time_axis, logs[meas_key], label="F_meas", color="blue", linewidth=1.2, alpha=0.7)
        ax_full.plot(time_axis, logs[meas_contact_key], label="F_meas_contact", color="green", linewidth=1.2, alpha=0.7)
        ax_full.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_full.set_xlim(0.0, t_end_plot)
        ax_full.set_title(f"{title} - Full")
        ax_full.legend(loc='upper right', fontsize=9)
        ax_full.grid(True, alpha=0.3)

        ax_zoom = axesForcesWorld[i * 2 + 1]
        ax_zoom.plot(time_axis, logs[cmd_key], color="red", linewidth=1.5)
        ax_zoom.plot(time_axis, logs[meas_key], color="blue", linewidth=1.5, alpha=0.7)
        ax_zoom.plot(time_axis, logs[meas_contact_key], color="green", linewidth=1.5, alpha=0.7)
        ax_zoom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_zoom.set_xlim(0.0, t_zoom)
        ax_zoom.set_title(f"{title} - Zoom: 0.0 - {t_zoom:.2f}s")
        ax_zoom.grid(True, alpha=0.3)

    figForcesWorld.supxlabel("Simulation Time (seconds)", fontsize=12)
    figForcesWorld.supylabel("Force (Newtons)", fontsize=12)
    figForcesWorld.suptitle("Force Components Profile (X, Y, Z)", fontsize=14, fontweight='bold')
    figures.append(("forces_xyz", figForcesWorld))

    # ==================== 3b. Total Planar Force Magnitude (XY) ====================
    figForceMag, (axesForceMag, axesForceMagZoom) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'hspace': 0.4})
 
    def plot_force_mag_on(ax):
        ax.plot(time_axis, logs["F_path_cmd_mag"],  label="||F_cmd||",
                color="red",  linewidth=1.5)
        ax.plot(time_axis, logs["F_world_meas_contact_mag"], label="||F_meas_contact||",
                color="blue", linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
 
    plot_force_mag_on(axesForceMag)
    axesForceMag.set_xlim(0.0, t_end_plot)
    axesForceMag.set_title("Force Magnitude - Full")
    axesForceMag.legend(loc='upper right', fontsize=9)
 
    plot_force_mag_on(axesForceMagZoom)
    axesForceMagZoom.set_xlim(0.0, t_zoom)
    axesForceMagZoom.set_title(f"Force Magnitude - Zoom: 0.0 - {t_zoom:.2f}s")
    axesForceMagZoom.legend(loc='upper right', fontsize=9)
 
    figForceMag.supxlabel("Simulation Time (seconds)", fontsize=12)
    figForceMag.supylabel("Force Magnitude (Newtons)", fontsize=12)
    figForceMag.suptitle("Total Force Magnitude (Commanded vs Measured)", fontsize=14, fontweight='bold')
    figures.append(("forces_total", figForceMag))

    # ==================== 4. Linear Velocity ====================
    figLinVel, (axesLinVel, axesLinVelZoom) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'hspace': 0.4})
    def plot_linvel_on(ax):
        ax.plot(time_axis, logs["ee_vel_world_mag"], label="v_ee_mag (cm/s)",  color="green", linewidth=1.5)
        ax.axhline(y=env.config.v_target_limit * 100, color='red', linestyle=':', label="Penalty Threshold")

    plot_linvel_on(axesLinVel)
    axesLinVel.set_title("Linear Velocity Magnitude (Eucl. Norm) - Full")
    axesLinVel.set_xlim(0.0, t_end_plot)
    axesLinVel.legend(loc='upper right', fontsize=9)
    axesLinVel.grid(True, alpha=0.3)

    plot_linvel_on(axesLinVelZoom)
    axesLinVelZoom.set_xlim(0.0, t_zoom)
    axesLinVelZoom.set_title(f"Linear Velocity Magnitude (Eucl. Norm) - Zoom: 0.0 - {t_zoom:.2f}s")
    axesLinVelZoom.legend(loc='upper right', fontsize=9)
    axesLinVelZoom.grid(True, alpha=0.3)

    figLinVel.supxlabel("Simulation Time (seconds)", fontsize=12)
    figLinVel.supylabel("Velocity (cm/s)", fontsize=12)
    figLinVel.suptitle("Linear Velocity Tracking (Mag)", fontsize=14, fontweight='bold')
    figures.append(("linear_velocity", figLinVel))

    # ==================== 5. Angular Velocity ====================
    figAngVel, axesAngVel = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'hspace': 0.6})
    
    axesAngVel[0].plot(time_axis, logs["ee_ang_vel_world_z"], color="purple", linewidth=1.5, label="w_ee_z")
    axesAngVel[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axesAngVel[0].set_xlim(0.0, t_end_plot)
    axesAngVel[0].set_title("Angular Velocity EE in World - Full")
    axesAngVel[0].grid(True, alpha=0.3)
    axesAngVel[0].legend()

    axesAngVel[1].plot(time_axis, logs["ee_ang_vel_world_z"], color="purple", linewidth=1.5, label="w_ee_z")
    axesAngVel[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axesAngVel[1].set_xlim(0.0, t_zoom)
    axesAngVel[1].set_title(f"Angular Velocity EE in World - Zoom 0.0 - {t_zoom:.2f}s")
    axesAngVel[1].grid(True, alpha=0.3)
    axesAngVel[1].legend()

    figAngVel.supxlabel("Simulation Time (seconds)", fontsize=12)
    figAngVel.supylabel("Angular Velocity (rad/s)", fontsize=12)
    figAngVel.suptitle("Angular Velocity Tracking", fontsize=14, fontweight='bold')
    figures.append(("angular_velocity", figAngVel))

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

    # ==================== 7. Commanded Forces in Local Frame (Path Frame) ====================
    figForceLocal, axesForceLocal = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.6})
    force_local_specs = [
        ("F_path_cmd_x", "F_path_meas_x", "Force X (tangent)"),
        ("F_path_cmd_y", "F_path_meas_y", "Force Y (lateral)"),
        ("F_path_cmd_z", "F_path_meas_z", "Force Z (normal)"),
    ]
    for i, (cmd, meas, title) in enumerate(force_local_specs):
        axesForceLocal[i].plot(time_axis, logs[cmd], label="F_path_cmd", color="red", linewidth=1.5)
        # Note: Inverted measured force for visual alignment
        axesForceLocal[i].plot(time_axis, logs[meas], label="F_path_meas (joints)", color="blue", alpha=0.7, linewidth=1.5)
        axesForceLocal[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axesForceLocal[i].set_xlim(0.0, t_end_plot)
        axesForceLocal[i].set_title(title)
        axesForceLocal[i].legend(loc='upper right')
        axesForceLocal[i].grid(True, alpha=0.3)
    figForceLocal.suptitle("Forces in Local Frame (Path Frame)", fontsize=14, fontweight='bold')
    figures.append(("local_forces", figForceLocal))

    # ==================== 8. Commanded Moments in Local Frame (Path Frame) ====================
    figMomLocal, axesMomLocal = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.6})
    tau_local_specs = [
        ("tau_path_cmd_x", "tau_path_meas_x", "Base Torque X (Roll)"),
        ("tau_path_cmd_y", "tau_path_meas_y", "Base Torque Y (Pitch)"),
        ("tau_path_cmd_z", "tau_path_meas_z", "Base Torque Z (Yaw)"),
    ]
    for i, (cmd, meas, title) in enumerate(tau_local_specs):
        axesMomLocal[i].plot(time_axis, logs[cmd], label="tau_path_cmd", color="red", linewidth=1.5)
        axesMomLocal[i].plot(time_axis, logs[meas], label="tau_path_meas (joints)", color="blue", alpha=0.7, linewidth=1.5)
        axesMomLocal[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axesMomLocal[i].set_xlim(0.0, t_end_plot)
        axesMomLocal[i].set_title(title)
        axesMomLocal[i].legend(loc='upper right')
        axesMomLocal[i].grid(True, alpha=0.3)
    figMomLocal.suptitle("Moments in Local Frame (Path Frame)", fontsize=14, fontweight='bold')
    figures.append(("local_moments", figMomLocal))

    # # ==================== 9. EE vs Base Commands ====================
    # fig10, axes10 = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'hspace': 0.4})
    # axes10[0].plot(time_axis, logs["F_cmd_ee_x"], label="EE Local Fx", color="purple")
    # axes10[0].plot(time_axis, logs["F_cmd_base_x"], label="Base World Fx", color="red", alpha=0.7)
    # axes10[0].set_title("Force Command: Local End-Effector vs World Base (X-Axis)")
    # axes10[0].legend()
    # axes10[0].set_xlim(0.0, t_end_plot)
    # axes10[0].grid(True, alpha=0.3)
    #
    # axes10[1].plot(time_axis, logs["tau_cmd_ee_z"], label="EE Local Tau_z (Yaw)", color="purple")
    # axes10[1].plot(time_axis, logs["tau_cmd_base_z"], label="Base World Tau_z", color="red", alpha=0.7)
    # axes10[1].set_title("Torque Command: Local End-Effector vs World Base (Z-Axis)")
    # axes10[1].legend()
    # axes10[1].set_xlim(0.0, t_end_plot)
    # axes10[1].grid(True, alpha=0.3)
    # fig10.suptitle("Frame Rotation Verification", fontsize=14, fontweight='bold')
    # figures.append(("frame_commands", fig10))

    # ==================== 10. RPY Angles ====================
    figRPY, axesRPY = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.6})
    rpy_specs = [
        ("roll", "Roll (Degrees)", "orange"),
        ("pitch", "Pitch (Degrees)", "green"),
        ("yaw", "Yaw (Degrees)", "purple"),
    ]
    for i, (key, title, color) in enumerate(rpy_specs):
        axesRPY[i].plot(time_axis, np.degrees(logs[key]), color=color, linewidth=1.5)
        axesRPY[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axesRPY[i].set_xlim(0.0, t_end_plot)
        axesRPY[i].set_title(title)
        axesRPY[i].grid(True, alpha=0.3)
    figRPY.suptitle("End-Effector Orientation in World", fontsize=14, fontweight='bold')
    figures.append(("rpy", figRPY))

    # ==================== 11. Bird's Eye View (MUST BE LAST) ====================
    figTrajectory, axesTrajectory = plt.subplots(figsize=(12, 10))
    if trajectory is not None:
        axesTrajectory.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=3,
                 label="Desired Trajectory", zorder=1)
        n_arrows = 6
        n_points = len(trajectory)
        indices = np.linspace(0, n_points - 10, n_arrows, dtype=int)
        for idx in indices:
            dx = trajectory[idx + 5, 0] - trajectory[idx, 0]
            dy = trajectory[idx + 5, 1] - trajectory[idx, 1]
            axesTrajectory.annotate(
                '',
                xy=(trajectory[idx, 0] + dx * 0.3, trajectory[idx, 1] + dy * 0.3),
                xytext=(trajectory[idx, 0], trajectory[idx, 1]),
                arrowprops=dict(
                    arrowstyle='-|>,head_length=1.2,head_width=0.8',
                    color='red', lw=2, alpha=0.9,
                ),
                zorder=2,
            )
 
    axesTrajectory.plot(logs["bottle_x"], logs["bottle_y"], 'b-', linewidth=2,
             label="Actual Bottle Path", zorder=3)
 
    step_size = max(1, len(logs["bottle_x"]) // 30)
    axesTrajectory.quiver(
        logs["bottle_x"][::step_size],
        logs["bottle_y"][::step_size],
        logs["F_world_meas_contact_x"][::step_size],
        logs["F_world_meas_contact_y"][::step_size],
        color='purple', angles='xy', scale_units='xy',
        scale=150, width=0.005, alpha=0.6,
        label="Measured Force Vectors (robot -> bottle)",
        zorder=4
    )
 
    axesTrajectory.plot(logs["ee_pos_x"], logs["ee_pos_y"], 'g-', linewidth=1, alpha=0.5,
             label="Hand Path", zorder=1)
    axesTrajectory.scatter(logs["bottle_x"][0],  logs["bottle_y"][0],
                c='black', s=120, marker='o', zorder=4, label="Start")
    axesTrajectory.scatter(logs["bottle_x"][-1], logs["bottle_y"][-1],
                c='green', s=120, marker='o', zorder=4, label="End")
 
    mid_idx = len(logs["bottle_x"]) // 2
    if mid_idx + 5 < len(logs["bottle_x"]):
        dx = logs["bottle_x"][mid_idx + 5] - logs["bottle_x"][mid_idx]
        dy = logs["bottle_y"][mid_idx + 5] - logs["bottle_y"][mid_idx]
        axesTrajectory.annotate(
            '',
            xy=(logs["bottle_x"][mid_idx] + dx * 3, logs["bottle_y"][mid_idx] + dy * 3),
            xytext=(logs["bottle_x"][mid_idx], logs["bottle_y"][mid_idx]),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2.5),
            zorder=5,
        )
 
    axesTrajectory.set_title("Bird's Eye View: X-Y Trajectory Map")
    axesTrajectory.set_xlabel("X Position (m)")
    axesTrajectory.set_ylabel("Y Position (m)")
    axesTrajectory.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        frameon=True,
    )
    axesTrajectory.set_aspect('equal')
    
    # We append the Bird's Eye view LAST so the time-formatting loop ignores it!
    figures.append(("trajectory", figTrajectory))

    # Plotting Path Tangential and Lateral Vectors (For Debugging only)
    figPathFrame, axesPathFrame = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.6})

    normal_specs = [
        ("t_hat_x", "b_hat_x"),
        ("t_hat_y", "b_hat_y"),
        ("t_hat_z", "b_hat_z"),
    ]
    for i, (t_plot, b_plot) in enumerate(normal_specs):
        axesPathFrame[i].plot(time_axis, logs[t_plot], label=t_plot, color="red", linewidth=1.5)
        axesPathFrame[i].plot(time_axis, logs[b_plot], label=b_plot, color="blue",
                      alpha=0.7, linewidth=1.5)
        axesPathFrame[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axesPathFrame[i].set_xlim(0.0, t_end_plot)
        axesPathFrame[i].legend(loc='upper right')
        axesPathFrame[i].grid(True, alpha=0.3)
    figPathFrame.suptitle("Path Frame", fontsize=14, fontweight='bold')
    figures.append(("path_frame", figPathFrame))
 
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

    model = PPO.load(model_path, env=env, device="cpu")
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
        # folder_name = f"Run_{clean_model_name}_{args.trajectory_type}_{timestamp}"
        folder_name = f"Run_{clean_model_name}_{args.trajectory_type}"
        save_dir = os.path.join(analysis_dir, folder_name)
        
        # Make the directory
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nCreated new directory for results:\n-> {save_dir}")

    # Pass save_dir instead of save_path
    plot_rewards(logs, trajectory, info, env, save_dir=save_dir)
    env.close()

if __name__ == "__main__":
    main()