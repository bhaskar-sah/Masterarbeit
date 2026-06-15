################################################################################################
# UPDATED: Reward components now match current reward.py
#   Active: r_progress, r_deviation, r_stability, r_contact, r_velocity
#   Removed: r_alignment, r_position, r_orientation (commented out in reward.py)
################################################################################################
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import matplotlib.ticker as ticker

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
        # Reward components (matches current reward.py)
        "r_progress": [], "r_deviation": [], "r_stability": [],
        "r_contact": [], "r_velocity": [], "r_total": [],
        "bottle_x": [], "bottle_y": [], "hand_x": [], "hand_y": [],
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

        v_curr = env.push_controller.get_ee_vel_base()
        w_curr = env.push_controller.get_ee_ang_vel_base()
        v_des_mag = np.linalg.norm([action[0] * env.config.v_max, action[1] * env.config.v_max])
        v_curr_mag = np.linalg.norm(v_curr[:2])
        wz_des = action[2] * env.config.w_max
        wz_curr = w_curr[2]

        F_cmd = env.push_controller.last_F_cmd
        F_meas = env.contact_manager.get_contact_force()
        F_cmd_mag = np.linalg.norm(F_cmd[:2])
        F_meas_mag = np.linalg.norm(F_meas[:2])

        logs["step"].append(step)
        logs["time"].append(sim_time)
        logs["progress"].append(info.get("progress", 0.0) * 100)
        logs["deviation"].append(info.get("deviation", 0.0) * 100)
        logs["tilt"].append(info.get("bottle_tilt", 1.0))
        logs["is_touching"].append(1.0 if info.get("is_touching", False) else 0.0)
        logs["action_vx"].append(action[0])
        logs["action_vy"].append(action[1])
        logs["action_wz"].append(action[2])
        logs["v_des_mag"].append(v_des_mag * 100)
        logs["v_curr_mag"].append(v_curr_mag * 100)
        logs["wz_des"].append(wz_des)
        logs["wz_curr"].append(wz_curr)
        logs["F_cmd_mag"].append(F_cmd_mag)
        logs["F_meas_mag"].append(F_meas_mag)
        # Forces per axis (no negation - signs are correct after contact.py fix)
        logs["F_cmd_x"].append(F_cmd[0])
        logs["F_cmd_y"].append(F_cmd[1])
        logs["F_cmd_z"].append(F_cmd[2])
        logs["F_meas_x"].append(F_meas[0])
        logs["F_meas_y"].append(F_meas[1])
        logs["F_meas_z"].append(F_meas[2])

        # Reward components (only active ones now)
        logs["r_progress"].append(info.get("r_progress", 0.0))
        logs["r_deviation"].append(info.get("r_deviation", 0.0))
        logs["r_stability"].append(info.get("r_stability", 0.0))
        logs["r_contact"].append(info.get("r_contact", 0.0))
        logs["r_velocity"].append(info.get("r_velocity", 0.0))
        logs["r_total"].append(reward)

        logs["bottle_x"].append(float(bottle_xy[0]))
        logs["bottle_y"].append(float(bottle_xy[1]))
        logs["hand_x"].append(float(hand_pos[0]))
        logs["hand_y"].append(float(hand_pos[1]))

        step += 1

    for key in logs:
        logs[key] = np.array(logs[key])

    print(f"  Episode duration (real sim time): {logs['time'][-1]:.2f} s")
    print(f"  Total control steps: {len(logs['time'])}")
    print(f"  Mean control_dt: {(logs['time'][-1] / max(len(logs['time']) - 1, 1)):.4f} s")

    return logs, trajectory, info


def plot_rewards(logs, trajectory, info, save_path=None):
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
    # Matches reward.py: r_progress + r_deviation + r_stability + r_contact + r_velocity - time_penalty
    reward_specs = [
        ("r_progress",  "Progress (+)",         "green"),
        ("r_deviation", "Deviation (-)",        "red"),
        ("r_stability", "Stability (-)",        "orange"),
        ("r_contact",   "Contact/Dist (-)",     "brown"),
        ("r_velocity",  "Velocity Penalty (-)", "purple"),
        ("r_total",     "Total Reward",         "black"),
    ]

    n_components = len(reward_specs)
    fig1, axes1 = plt.subplots(n_components, 1, figsize=(16, 18), gridspec_kw={'hspace': 0.6})

    for i, (key, title, color) in enumerate(reward_specs):
        ax = axes1[i]
        ax.plot(time_axis, logs[key], color=color, linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlim(0.0, t_end_plot)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig1.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig1.supylabel("Reward Value", fontsize=12)
    fig1.suptitle(f"Reward Components Breakdown | {suptitle}", fontsize=14, fontweight='bold')
    figures.append(("rewards", fig1))

    # ==================== 2. Raw Actions ====================
    fig2, axes2 = plt.subplots(6, 1, figsize=(16, 25), gridspec_kw={'hspace': 1.2})
    action_specs = [
        ("action_vx", "vx (Forward)", "blue"),
        ("action_vy", "vy (Lateral)", "green"),
        ("action_wz", "wz (Yaw)",     "purple"),
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
        ax_full.plot(time_axis, logs[meas_key], label="F_meas (Actual)", color="blue", linewidth=1.2, alpha=0.7)
        ax_full.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_full.set_xlim(0.0, t_end_plot)
        ax_full.set_title(f"{title} - Full")
        ax_full.legend(loc='upper right', fontsize=9)
        ax_full.grid(True, alpha=0.3)

        ax_zoom = axes3[i * 2 + 1]
        ax_zoom.plot(time_axis, logs[cmd_key], color="red", linewidth=1.5)
        ax_zoom.plot(time_axis, logs[meas_key], color="blue", linewidth=1.5, alpha=0.7)
        ax_zoom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_zoom.set_xlim(0.0, t_zoom)
        ax_zoom.set_title(f"{title} - Zoom: 0.0 - {t_zoom:.2f}s")
        ax_zoom.grid(True, alpha=0.3)

    fig3.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig3.supylabel("Force (Newtons)", fontsize=12)
    fig3.suptitle("Force Components Profile (X, Y, Z)", fontsize=14, fontweight='bold')
    figures.append(("forces_xyz", fig3))

    # ==================== 3b. Total Planar Force Magnitude (XY) ====================
    # Shows ||F_xy|| over time. Useful for spotting force-magnitude bang-bang
    # without worrying about direction. Magnitudes are always >= 0.
    fig3b, (ax3b, ax3bz) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'hspace': 0.4})
 
    def plot_force_mag_on(ax):
        ax.plot(time_axis, logs["F_cmd_mag"],  label="||F_cmd||  (Commanded, XY)",
                color="red",  linewidth=1.5)
        ax.plot(time_axis, logs["F_meas_mag"], label="||F_meas|| (Contact reaction, XY)",
                color="blue", linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
 
    # Full view
    plot_force_mag_on(ax3b)
    ax3b.set_xlim(0.0, t_end_plot)
    ax3b.set_title("Total Planar Force Magnitude (XY plane) - Full")
    ax3b.legend(loc='upper right', fontsize=9)
 
    # Zoom view
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
        ax.plot(time_axis, logs["v_des_mag"],  label="Desired v (cm/s)", color="red",   linewidth=1.5)
        ax.plot(time_axis, logs["v_curr_mag"], label="Actual v (cm/s)",  color="green", linewidth=1.5)
    plot_linvel_on(ax4)
    ax4.set_title("Velocity magnitude (Eucl. Norm) - Full")
    ax4.set_xlim(0.0, t_end_plot)
    ax4.legend(loc='upper right', fontsize=9)
    plot_linvel_on(ax4z)
    ax4z.set_xlim(0.0, t_zoom)
    ax4z.set_title(f"Velocity magnitude (Eucl. Norm) - Zoom: 0.0 - {t_zoom:.2f}s")
    fig4.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig4.supylabel("Velocity (cm/s)", fontsize=12)
    fig4.suptitle("Planar Velocity Tracking (Mag)", fontsize=14, fontweight='bold')
    figures.append(("linear_velocity", fig4))

    # ==================== 5. Angular Velocity ====================
    fig5, axes5 = plt.subplots(4, 1, figsize=(16, 14), gridspec_kw={'hspace': 0.6})
    angvel_specs = [
        ("wz_des",  "Desired wz",  "red"),
        ("wz_curr", "Actual wz",   "purple"),
    ]
    for i, (key, title, color) in enumerate(angvel_specs):
        ax_full = axes5[i * 2]
        ax_full.plot(time_axis, logs[key], color=color, linewidth=1.5)
        ax_full.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_full.set_xlim(0.0, t_end_plot)
        ax_full.set_title(f"{title} - Full")
        ax_zoom = axes5[i * 2 + 1]
        ax_zoom.plot(time_axis, logs[key], color=color, linewidth=1.5)
        ax_zoom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_zoom.set_xlim(0.0, t_zoom)
        ax_zoom.set_title(f"{title} - Zoom 0.0 - {t_zoom:.2f}s")
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

    # ==================== 7. Bird's Eye View ====================
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
 
    # Force arrows (no negation - signs are correct after contact.py fix)
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
    # Legend placed OUTSIDE the plot on the right.
    # bbox_to_anchor=(1.02, 1) puts the top-left corner of the legend just to the right of the axes.
    ax7.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        frameon=True,
    )
    ax7.set_aspect('equal')
    figures.append(("trajectory", fig7))
 
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
    if save_path:
        base, ext = os.path.splitext(save_path)
        for name, fig in figures:
            fig_path = f"{base}_{name}{ext}"
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {fig_path}")
    else:
        plt.show()
 
    # plt.show()
    return figures


def main():
    parser = argparse.ArgumentParser(description="Plot diagnostic dashboard for pure velocity control.")
    parser.add_argument("--model_dir", type=str, required=True, help="Name of the model directory")
    parser.add_argument("--trajectory_type", type=str, default="s_curve",
                        choices=["straight", "curved", "s_curve"])
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

    save_path = None
    if args.save:
        analysis_dir = os.path.join(current_dir, "reward_results")
        os.makedirs(analysis_dir, exist_ok=True)
        base_name = f"{args.model_dir}_{args.trajectory_type}_mujoco_time"
        save_path = os.path.join(analysis_dir, f"{base_name}.png")
        counter = 1
        while os.path.exists(save_path):
            save_path = os.path.join(analysis_dir, f"{base_name}_{counter}.png")
            counter += 1

    plot_rewards(logs, trajectory, info, save_path=save_path)
    env.close()


if __name__ == "__main__":
    main()