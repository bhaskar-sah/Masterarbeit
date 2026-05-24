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
    """Run one episode and collect detailed logs based on pure velocity control."""

    obs, _ = env.reset(options={"trajectory_type": trajectory_type})

    # ============ ANCHOR t=0 TO POST-RESET MUJOCO CLOCK ============
    # env.data.time is the live MuJoCo simulation clock (in seconds).
    # It is advanced automatically by 'dt' every time mj_step() runs.
    # We record its value right after reset so the plotted x-axis starts at 0.
    t_start = env.data.time
    # ================================================================

    logs = {
        "step": [],
        "time": [],          # Real simulation time from MuJoCo (seconds)
        "progress": [],
        "deviation": [],
        "tilt": [],
        "is_touching": [],

        # Actions
        "action_vx": [],
        "action_vy": [],
        "action_wz": [],

        # Velocities
        "v_des_mag": [],
        "v_curr_mag": [],
        "wz_des": [],
        "wz_curr": [],

        # Forces
        "F_cmd_mag": [],
        "F_meas_mag": [],
        # extracting x, y and z axis
        "F_cmd_x": [], "F_cmd_y": [], "F_cmd_z": [],   
        "F_meas_x": [], "F_meas_y": [], "F_meas_z": [], 

        # Individual rewards
        "r_progress": [],
        "r_deviation": [],
        "r_stability": [],
        "r_contact": [],
        "r_alignment": [],
        "r_position": [],
        "r_orientation": [],
        "r_total": [],

        # Positions for bird's eye view
        "bottle_x": [],
        "bottle_y": [],
        "hand_x": [],
        "hand_y": [],
    }

    # Store trajectory for plotting
    trajectory = env.traj_manager.trajectory.copy() if env.traj_manager.trajectory is not None else None

    done = False
    step = 0

    while not done:
        # Predict action (deterministic=True strips out random training noise)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ============ REAL MUJOCO TIMESTAMP ============
        # env.data.time has now advanced by control_dt = n_substeps * dt = 20 * 0.002 = 0.04 s
        # Subtract t_start so the axis starts at 0 even if the post-reset clock isn't exactly zero.
        sim_time = env.data.time - t_start
        # ================================================

        # Extract Physics / Tracking
        bottle_xy = env.data.xpos[env.bottle_body_id][:2].copy()
        hand_pos = env.data.xpos[env.hand_body_id].copy()

        # Velocities
        v_curr = env.push_controller.get_ee_velocity()
        w_curr = env.push_controller.get_ee_angular_velocity()

        v_des_mag = np.linalg.norm([action[0] * env.config.v_max, action[1] * env.config.v_max])
        v_curr_mag = np.linalg.norm(v_curr[:2])  # Only XY for planar pushing
        wz_des = action[2] * env.config.w_max
        wz_curr = w_curr[2]

        # Forces
        F_cmd = env.push_controller.last_F_cmd
        F_meas = env.contact_manager.get_contact_force()
        F_cmd_mag = np.linalg.norm(F_cmd[:2])    # Planar force command
        F_meas_mag = np.linalg.norm(F_meas[:2])  # Planar measured force

        # Log everything
        logs["step"].append(step)
        logs["time"].append(sim_time)            # <-- Real MuJoCo time, not step * 0.002
        logs["progress"].append(info.get("progress", 0.0) * 100)
        logs["deviation"].append(info.get("deviation", 0.0) * 100)  # cm
        logs["tilt"].append(info.get("bottle_tilt", 1.0))
        logs["is_touching"].append(1.0 if info.get("is_touching", False) else 0.0)

        logs["action_vx"].append(action[0])
        logs["action_vy"].append(action[1])
        logs["action_wz"].append(action[2])

        logs["v_des_mag"].append(v_des_mag * 100)    # cm/s for readable plots
        logs["v_curr_mag"].append(v_curr_mag * 100)  # cm/s
        logs["wz_des"].append(wz_des)
        logs["wz_curr"].append(wz_curr)

        logs["F_cmd_mag"].append(F_cmd_mag)
        logs["F_meas_mag"].append(F_meas_mag)

        # ---> logging commanded and measured forces for each axes <---
        logs["F_cmd_x"].append(-F_cmd[0])
        logs["F_cmd_y"].append(-F_cmd[1])
        logs["F_cmd_z"].append(-F_cmd[2])
        logs["F_meas_x"].append(F_meas[0])
        logs["F_meas_y"].append(F_meas[1])
        logs["F_meas_z"].append(F_meas[2])

        logs["r_progress"].append(info.get("r_progress", 0.0))
        logs["r_deviation"].append(info.get("r_deviation", 0.0))
        logs["r_stability"].append(info.get("r_stability", 0.0))
        logs["r_contact"].append(info.get("r_contact", 0.0))
        logs["r_alignment"].append(info.get("r_alignment", 0.0))
        logs["r_position"].append(info.get("r_position", 0.0))
        logs["r_orientation"].append(info.get("r_orientation", 0.0))
        logs["r_total"].append(reward)

        logs["bottle_x"].append(float(bottle_xy[0]))
        logs["bottle_y"].append(float(bottle_xy[1]))
        logs["hand_x"].append(float(hand_pos[0]))
        logs["hand_y"].append(float(hand_pos[1]))

        step += 1

    # Convert lists to numpy arrays
    for key in logs:
        logs[key] = np.array(logs[key])

    print(f"  Episode duration (real sim time): {logs['time'][-1]:.2f} s")
    print(f"  Total control steps: {len(logs['time'])}")
    print(f"  Mean control_dt: {(logs['time'][-1] / max(len(logs['time']) - 1, 1)):.4f} s")

    return logs, trajectory, info


def plot_rewards(logs, trajectory, info, save_path=None):
    """Create separate diagnostic plots, one per figure."""

    time_axis = logs["time"]

    # ============ DATA-DRIVEN AXIS LIMITS ============
    # No more hardcoded 2.2s — use the actual episode duration from MuJoCo.
    t_end = float(time_axis[-1])
    # Zoom window for first-transient plots: first ~10% of the episode, or 1s, whichever is smaller.
    t_zoom = min(1.0, t_end * 0.1)
    # Round t_end up to the nearest 0.5 s for clean tick alignment.
    t_end_plot = np.ceil(t_end * 2) / 2
    # ===================================================

    figures = []

    suptitle = (
        f"Final Progress: {logs['progress'][-1]:.1f}% | "
        f"Final Dev: {logs['deviation'][-1]:.1f}cm | "
        f"Success: {info.get('is_success', False)}"
    )

    # ==================== 1. Individual Rewards ====================
    reward_specs = [
        ("r_progress",    "Progress (+)",        "green"),
        ("r_alignment",   "Force Align (+)",     "blue"),
        ("r_deviation",   "Deviation (-)",       "red"),
        ("r_position",    "Position Err (-)",    "purple"),
        ("r_orientation", "Orientation Err (-)", "orange"),
        ("r_contact",     "Contact/Dist (-)",    "brown"),
    ]

    n_components = len(reward_specs)
    fig1, axes1 = plt.subplots(n_components, 1, figsize=(16, 18),
                                gridspec_kw={'hspace': 0.6})

    for i, (key, title, color) in enumerate(reward_specs):
        ax = axes1[i]
        ax.plot(time_axis, logs[key], color=color, linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlim(0.0, t_end_plot)
        ax.set_title(title)

    fig1.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig1.supylabel("Reward Value", fontsize=12)
    fig1.suptitle(f"Reward Components Breakdown | {suptitle}", fontsize=14, fontweight='bold')
    figures.append(("rewards", fig1))

    # ==================== 2. Raw Actions (3 actions × 2 views, all stacked) ====================
    fig2, axes2 = plt.subplots(6, 1, figsize=(16, 25), gridspec_kw={'hspace': 1.2})

    action_specs = [
        ("action_vx", "vx (Forward)", "blue"),
        ("action_vy", "vy (Lateral)", "green"),
        ("action_wz", "wz (Yaw)",     "purple"),
    ]

    for i, (key, title, color) in enumerate(action_specs):
        # Full view (rows 0, 2, 4)
        ax_full = axes2[i * 2]
        ax_full.plot(time_axis, logs[key], color=color, linewidth=1.2)
        ax_full.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_full.set_xlim(0.0, t_end_plot)
        ax_full.set_ylim(-1.2, 1.2)
        ax_full.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_full.set_title(f"{title} – Full")
        ax_full.grid(True, alpha=0.3)

        # Zoom view directly below (rows 1, 3, 5)
        ax_zoom = axes2[i * 2 + 1]
        ax_zoom.plot(time_axis, logs[key], color=color, linewidth=1.2)
        ax_zoom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_zoom.set_xlim(0.0, t_zoom)
        ax_zoom.set_ylim(-1.2, 1.2)
        ax_zoom.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_zoom.set_title(f"{title} – Zoom 0.0 – {t_zoom:.2f}s")
        ax_zoom.grid(True, alpha=0.3)

    fig2.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig2.supylabel("Action Command", fontsize=12)
    fig2.suptitle("Raw Neural Network Actions [-1, 1]", fontsize=14, fontweight='bold')
    figures.append(("actions", fig2))

    # # ==================== 3. Forces ====================
    # fig3, (ax3, ax3z) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'hspace': 0.4})

    # def plot_forces_on(ax):
    #     ax.plot(time_axis, logs["F_cmd_mag"], label="F_cmd (from Impedance)", color="red")
    #     ax.plot(time_axis, logs["F_meas_mag"], label="F_measured (Actual Contact)", color="blue", linewidth=1.5)

    # # Full view
    # plot_forces_on(ax3)
    # ax3.set_xlim(0.0, t_end_plot)
    # ax3.set_title("Command vs Reality (XY Plane) – Full")
    # ax3.legend(loc='upper right', fontsize=9)

    # # Zoom view
    # plot_forces_on(ax3z)
    # ax3z.set_xlim(0.0, t_zoom)
    # ax3z.set_title(f"Command vs Reality (XY Plane) – Zoom: 0.0 – {t_zoom:.2f}s")

    # fig3.supxlabel("Simulation Time (seconds)", fontsize=12)
    # fig3.supylabel("Force (Newtons)", fontsize=12)
    # fig3.suptitle("Force Profile", fontsize=14, fontweight='bold')
    # figures.append(("forces", fig3))

    # ==================== 3. Forces (X, Y, Z Components) ====================
    fig3, axes3 = plt.subplots(6, 1, figsize=(16, 22), gridspec_kw={'hspace': 0.8})

    force_specs = [
        ("F_cmd_x", "F_meas_x", "X-Axis Force (Forward/Back)"),
        ("F_cmd_y", "F_meas_y", "Y-Axis Force (Left/Right)"),
        ("F_cmd_z", "F_meas_z", "Z-Axis Force (Up/Down)"),
    ]

    for i, (cmd_key, meas_key, title) in enumerate(force_specs):
        # Full view
        ax_full = axes3[i * 2]
        ax_full.plot(time_axis, logs[cmd_key], label="F_cmd (Command)", color="red", linewidth=1.2)
        ax_full.plot(time_axis, logs[meas_key], label="F_meas (Actual)", color="blue", linewidth=1.2, alpha=0.7)
        ax_full.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_full.set_xlim(0.0, t_end_plot)
        ax_full.set_title(f"{title} – Full")
        ax_full.legend(loc='upper right', fontsize=9)
        ax_full.grid(True, alpha=0.3)

        # Zoom view
        ax_zoom = axes3[i * 2 + 1]
        ax_zoom.plot(time_axis, logs[cmd_key], color="red", linewidth=1.5)
        ax_zoom.plot(time_axis, logs[meas_key], color="blue", linewidth=1.5, alpha=0.7)
        ax_zoom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_zoom.set_xlim(0.0, t_zoom)
        ax_zoom.set_title(f"{title} – Zoom: 0.0 – {t_zoom:.2f}s")
        ax_zoom.grid(True, alpha=0.3)

    fig3.supxlabel("Simulation Time (seconds)", fontsize=12)
    fig3.supylabel("Force (Newtons)", fontsize=12)
    fig3.suptitle("Force Components Profile (X, Y, Z)", fontsize=14, fontweight='bold')
    figures.append(("forces_xyz", fig3))

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
    ax4z.set_title(f"Velocity magnitude (Eucl. Norm) - Zoom: 0.0 – {t_zoom:.2f}s")

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
        # Full view
        ax_full = axes5[i * 2]
        ax_full.plot(time_axis, logs[key], color=color, linewidth=1.5)
        ax_full.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_full.set_xlim(0.0, t_end_plot)
        ax_full.set_title(f"{title} – Full")

        # Zoom view
        ax_zoom = axes5[i * 2 + 1]
        ax_zoom.plot(time_axis, logs[key], color=color, linewidth=1.5)
        ax_zoom.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_zoom.set_xlim(0.0, t_zoom)
        ax_zoom.set_title(f"{title} – Zoom 0.0 – {t_zoom:.2f}s")

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

    # Desired trajectory (reference)
    if trajectory is not None:
        ax7.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=3,
                 label="Desired Trajectory", zorder=1)

        # Tangent arrows along the desired trajectory
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
                    color='red',
                    lw=2,
                    alpha=0.9,
                ),
                zorder=2,
            )

    # Actual bottle path
    ax7.plot(logs["bottle_x"], logs["bottle_y"], 'b-', linewidth=2,
             label="Actual Bottle Path", zorder=3)

    # Hand path
    ax7.plot(logs["hand_x"], logs["hand_y"], 'g-', linewidth=1, alpha=0.5,
             label="Hand Path", zorder=1)

    # Start and end markers
    ax7.scatter(logs["bottle_x"][0],  logs["bottle_y"][0],
                c='black', s=120, marker='o', zorder=4, label="Start")
    ax7.scatter(logs["bottle_x"][-1], logs["bottle_y"][-1],
                c='green', s=120, marker='o', zorder=4, label="End")

    # Direction arrow on the actual bottle path (midpoint)
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
    ax7.legend(loc='lower left', bbox_to_anchor=(0.0, 0.6), fontsize=8)
    ax7.set_aspect('equal')
    figures.append(("trajectory", fig7))

    # ==================== TICK FORMATTING (data-driven) ====================
    # Major ticks every 1s for full views, every 0.1s for zoom views.
    # We detect zoom axes by their x-limit (any axis with xlim <= t_zoom + epsilon).
    for _, fig in figures[:6]:  # everything except trajectory
        for ax in fig.axes:
            xlim = ax.get_xlim()
            # Skip axes that share a y-axis (twinx) and have no x data of their own
            if xlim[1] - xlim[0] <= t_zoom + 0.05:
                # Zoom subplot
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
            else:
                # Full subplot
                # Pick step size so we always get ~10 ticks regardless of episode length
                step = max(0.5, round(t_end_plot / 10, 1))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax.tick_params(axis='x', labelsize=8, rotation=45)

    # Apply tight layout to each
    for _, fig in figures:
        fig.tight_layout()

    # Save or show
    # if save_path:
    #     base, ext = os.path.splitext(save_path)
    #     for name, fig in figures:
    #         fig_path = f"{base}_{name}{ext}"
    #         fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    #         print(f"Saved: {fig_path}")
    # else:
    #     plt.show()

    plt.show()

    return figures


def main():
    parser = argparse.ArgumentParser(description="Plot diagnostic dashboard for pure velocity control.")
    parser.add_argument("--model_dir", type=str, required=True, help="Name of the model directory")
    parser.add_argument("--trajectory_type", type=str, default="s_curve",
                        choices=["straight", "curved", "s_curve"])
    parser.add_argument("--save", action="store_true", help="Save the plot to the reward_results directory")
    args = parser.parse_args()

    # Create environment
    env = PandaPushTrajectoryEnv(render_mode=None, trajectory_type=args.trajectory_type)

    # =========================================================================
    # PATH FIX: Stepping in and out of folders correctly
    # =========================================================================
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)

    model_path = os.path.join(parent_dir, "saved_models_new", args.model_dir, "best_model.zip")

    print(f"\nLoading best model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Could not find best_model.zip at {model_path}")
        exit()

    model = PPO.load(model_path, env=env)

    # Run episode
    print(f"Running episode with trajectory: {args.trajectory_type}...")
    logs, trajectory, info = run_episode_and_collect(env, model, args.trajectory_type)

    print(f"\nEpisode complete! Generating plots...")
    print(f"  Progress: {logs['progress'][-1]:.1f}%")
    print(f"  Final deviation: {logs['deviation'][-1]:.1f}cm")

    # Generate save path inside the nested reward_results folder
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