"""
Plot reward components and trajectory metrics for the Panda Push Environment.

Usage:
    python plot_rewards.py --model_path <path_to_model.zip> --trajectory_type straight

This runs one episode and plots:
    1. Individual reward components over time
    2. Trajectory deviation + progress
    3. Angle error θ + wrist rotation
    4. Force magnitude + stiffness K
    5. Bottle tilt
    6. Actual bottle path vs desired trajectory (bird's eye view)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys


def run_episode_and_collect(env, model, trajectory_type="straight"):
    """Run one episode and collect detailed logs."""

    obs, _ = env.reset(options={"trajectory_type": trajectory_type})

    logs = {
        "step": [],
        "progress": [],
        "deviation": [],
        "angle_error_deg": [],
        "force_mag": [],
        "tilt": [],
        "stiffness": [],
        "wrist_deg": [],
        "target_z": [],
        "is_touching": [],
        "is_settling": [],
        # Individual rewards
        "r_progress": [],
        "r_deviation": [],
        "r_angle": [],
        "r_stability": [],
        "r_contact": [],
        "r_total": [],
        # Positions
        "bottle_x": [],
        "bottle_y": [],
        "hand_x": [],
        "hand_y": [],
        "hand_z": [],
    }

    # Store trajectory for plotting
    trajectory = env.trajectory.copy() if env.trajectory is not None else None

    done = False
    step = 0
    prev_progress = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Get current state
        bottle_xy = env.data.xpos[env.bottle_body_id][:2]
        hand_pos = env.data.xpos[env.hand_body_id]
        tilt = env._get_bottle_tilt()
        is_touching = env._is_touching()
        force = env._get_contact_force()
        force_mag = float(np.linalg.norm(force))
        angle_error = env._get_angle_error(bottle_xy)
        angle_deg = np.degrees(angle_error)
        _, deviation_mag = env._get_path_deviation(bottle_xy)
        progress = env._get_progress(bottle_xy)
        K_avg = float(np.mean(env.current_K))
        wrist_deg = float(np.degrees(env.wrist_offset))

        # Adaptive target z
        # Adaptive target z (safe for both old and new env)
        bottle_y = env.data.xpos[env.bottle_body_id][1]
        bottle_start_y = getattr(env, 'bottle_start_y', 0.2)
        distance_from_start = abs(bottle_y - bottle_start_y)
        if hasattr(env, 'bottle_start_y'):
            current_target_z = env.target_z + 0.02 * distance_from_start
            current_target_z = max(current_target_z, 0.88)
        else:
            current_target_z = env.target_z  # Old env: fixed height

        # Compute individual reward components (mirror the reward function)
        if env.in_approach:
            dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)
            r_progress = 0.0
            r_deviation = 0.0
            r_angle = 0.0
            r_stability = 0.0
            r_contact = 0.0
            r_total = -2.0 * dist_to_bottle - 5.0 * abs(hand_pos[2] - env.target_z)
        else:
            # Progress
            progress_delta = progress - prev_progress
            r_progress = 100.0 * max(progress_delta, 0)

            # Deviation
            r_deviation = 5.0 - 100.0 * deviation_mag
            r_deviation = max(r_deviation, -15.0)

            # Angle
            if angle_deg < 10:
                r_angle = 2.0
            elif angle_deg < 30:
                r_angle = 1.0
            elif angle_deg < 60:
                r_angle = 0.0
            else:
                r_angle = -2.0

            # Stability
            if tilt > 0.995:
                r_stability = 3.0
            elif tilt > 0.99:
                r_stability = 1.0
            elif tilt > 0.98:
                r_stability = 0.0
            else:
                r_stability = -15.0 * (1 - tilt)

            # Contact
            dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)
            if is_touching:
                r_contact = 2.0
            else:
                r_contact = -10.0 * dist_to_bottle
                if dist_to_bottle > 0.08:
                    r_contact -= 5.0

            r_total = r_progress + r_deviation + r_angle + r_stability + r_contact - 0.005

        prev_progress = progress

        # Log everything
        logs["step"].append(step)
        logs["progress"].append(progress * 100)
        logs["deviation"].append(deviation_mag * 100)  # cm
        logs["angle_error_deg"].append(angle_deg)
        logs["force_mag"].append(force_mag)
        logs["tilt"].append(tilt)
        logs["stiffness"].append(K_avg)
        logs["wrist_deg"].append(wrist_deg)
        logs["target_z"].append(current_target_z)
        logs["is_touching"].append(1.0 if is_touching else 0.0)
        logs["is_settling"].append(1.0 if env.is_settling else 0.0)
        logs["r_progress"].append(r_progress)
        logs["r_deviation"].append(r_deviation)
        logs["r_angle"].append(r_angle)
        logs["r_stability"].append(r_stability)
        logs["r_contact"].append(r_contact)
        logs["r_total"].append(r_total)
        logs["bottle_x"].append(float(bottle_xy[0]))
        logs["bottle_y"].append(float(bottle_xy[1]))
        logs["hand_x"].append(float(hand_pos[0]))
        logs["hand_y"].append(float(hand_pos[1]))
        logs["hand_z"].append(float(hand_pos[2]))

        step += 1

    # Convert to numpy
    for key in logs:
        logs[key] = np.array(logs[key])

    return logs, trajectory, info


def plot_rewards(logs, trajectory, info, save_path=None):
    """Create comprehensive reward analysis plots."""

    steps = logs["step"]

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(6, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"Episode Analysis | Progress: {logs['progress'][-1]:.1f}% | "
        f"Final Dev: {logs['deviation'][-1]:.1f}cm | "
        f"Success: {info.get('is_success', False)}",
        fontsize=14, fontweight='bold'
    )

    # ==================== 1. Individual Reward Components ====================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(steps, logs["r_progress"], label="r_progress", alpha=0.8, linewidth=1)
    ax1.plot(steps, logs["r_deviation"], label="r_deviation", alpha=0.8, linewidth=1)
    ax1.plot(steps, logs["r_angle"], label="r_angle", alpha=0.8, linewidth=1)
    ax1.plot(steps, logs["r_stability"], label="r_stability", alpha=0.8, linewidth=1)
    ax1.plot(steps, logs["r_contact"], label="r_contact", alpha=0.8, linewidth=1)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Reward")
    ax1.set_title("Individual Reward Components")
    ax1.legend(loc='upper right', ncol=5, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ==================== 2. Total Reward ====================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(steps, logs["r_total"], color='black', alpha=0.8, linewidth=1)
    # Add cumulative reward on secondary axis
    ax2b = ax2.twinx()
    cumulative = np.cumsum(logs["r_total"])
    ax2b.plot(steps, cumulative, color='blue', alpha=0.5, linewidth=1, linestyle='--')
    ax2b.set_ylabel("Cumulative Reward", color='blue')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Step Reward")
    ax2.set_title("Total Reward (solid) + Cumulative (dashed)")
    ax2.grid(True, alpha=0.3)

    # ==================== 3. Progress + Deviation ====================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(steps, logs["progress"], label="Progress (%)", color='green', linewidth=2)
    ax3b = ax3.twinx()
    ax3b.plot(steps, logs["deviation"], label="Deviation (cm)", color='red', linewidth=1.5)
    ax3b.axhline(y=5.0, color='red', linestyle='--', alpha=0.3, label="5cm tolerance")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Progress (%)", color='green')
    ax3b.set_ylabel("Deviation (cm)", color='red')
    ax3.set_title("Progress & Deviation")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ==================== 4. Angle Error + Wrist ====================
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(steps, logs["angle_error_deg"], label="θ (angle error)", color='purple', alpha=0.7, linewidth=1)
    ax4b = ax4.twinx()
    ax4b.plot(steps, logs["wrist_deg"], label="Wrist (deg)", color='orange', alpha=0.7, linewidth=1)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Angle Error θ (deg)", color='purple')
    ax4b.set_ylabel("Wrist Rotation (deg)", color='orange')
    ax4.set_title("Force Alignment & Wrist Rotation")
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ==================== 5. Force + Stiffness ====================
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(steps, logs["force_mag"], label="Force (N)", color='blue', alpha=0.7, linewidth=1)
    ax5b = ax5.twinx()
    ax5b.plot(steps, logs["stiffness"], label="K (N/m)", color='brown', alpha=0.7, linewidth=1)
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Contact Force (N)", color='blue')
    ax5b.set_ylabel("Stiffness K (N/m)", color='brown')
    ax5.set_title("Force & Stiffness Profile")
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ==================== 6. Tilt + Contact + Settle ====================
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(steps, logs["tilt"], label="Tilt", color='darkgreen', linewidth=1.5)
    ax6.axhline(y=0.99, color='green', linestyle='--', alpha=0.3, label="tilt_ok (0.99)")
    ax6.axhline(y=0.96, color='orange', linestyle='--', alpha=0.3, label="tilt_stop (0.96)")
    ax6.axhline(y=0.97, color='blue', linestyle='--', alpha=0.3, label="settle_exit (0.97)")
    # Shade settle regions
    settle = logs["is_settling"]
    for i in range(len(settle) - 1):
        if settle[i] > 0.5:
            ax6.axvspan(steps[i], steps[i + 1], color='red', alpha=0.1)
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Tilt (1.0 = upright)")
    ax6.set_title("Bottle Tilt (red = settling)")
    ax6.legend(loc='lower left', fontsize=8)
    ax6.set_ylim(0.9, 1.01)
    ax6.grid(True, alpha=0.3)

    # ==================== 7. Hand Height + Adaptive Z ====================
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.plot(steps, logs["hand_z"], label="Hand Z (actual)", color='blue', linewidth=1.5)
    ax7.plot(steps, logs["target_z"], label="Target Z (adaptive)", color='red', linewidth=1, linestyle='--')
    ax7.axhline(y=0.88, color='black', linestyle=':', alpha=0.3, label="min_safe_z (0.88)")
    ax7.set_xlabel("Step")
    ax7.set_ylabel("Height (m)")
    ax7.set_title("Hand Height vs Adaptive Target")
    ax7.legend(loc='upper left', fontsize=8)
    ax7.grid(True, alpha=0.3)

    # ==================== 8. Bird's Eye View: Trajectory ====================
    ax8 = fig.add_subplot(gs[4:, :])
    if trajectory is not None:
        ax8.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label="Desired trajectory", zorder=1)
    ax8.plot(logs["bottle_x"], logs["bottle_y"], 'b-', linewidth=1.5, alpha=0.8, label="Actual bottle path", zorder=2)
    ax8.plot(logs["hand_x"], logs["hand_y"], 'g-', linewidth=0.8, alpha=0.5, label="Hand path", zorder=1)

    # Mark start and end
    ax8.scatter(logs["bottle_x"][0], logs["bottle_y"][0], c='blue', s=100, marker='o', zorder=3, label="Start")
    ax8.scatter(logs["bottle_x"][-1], logs["bottle_y"][-1], c='blue', s=100, marker='*', zorder=3, label="End")

    # Color dots by deviation magnitude
    scatter = ax8.scatter(
        logs["bottle_x"][::10], logs["bottle_y"][::10],
        c=logs["deviation"][::10], cmap='RdYlGn_r', s=20,
        vmin=0, vmax=5, zorder=4, alpha=0.8
    )
    plt.colorbar(scatter, ax=ax8, label="Deviation (cm)", shrink=0.6)

    ax8.set_xlabel("X (m)")
    ax8.set_ylabel("Y (m)")
    ax8.set_title("Bird's Eye View: Desired vs Actual Trajectory")
    ax8.legend(loc='upper right', fontsize=8)
    ax8.set_aspect('equal')
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot reward components for push environment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .zip file")
    parser.add_argument("--trajectory_type", type=str, default="straight", choices=["straight", "curved", "s_curve"])
    parser.add_argument("--save_path", type=str, default=None, help="Save plot to file (e.g., rewards.png)")
    parser.add_argument("--env_path", type=str, default=None, help="Path to environment directory")
    args = parser.parse_args()

    # Import after parsing args so we can add to path if needed
    if args.env_path:
        sys.path.insert(0, args.env_path)

    from stable_baselines3 import PPO
    from push_with_finger_learn_force_impedance_14_v01_straight import PandaPushTrajectoryEnv

    # Create environment (no rendering for data collection)
    env = PandaPushTrajectoryEnv(render_mode=None, trajectory_type=args.trajectory_type)

    # Load model
    print(f"Loading model: {args.model_path}")
    model = PPO.load(args.model_path, env=env)

    # Run episode and collect data
    print(f"Running episode with trajectory: {args.trajectory_type}")
    logs, trajectory, info = run_episode_and_collect(env, model, args.trajectory_type)

    print(f"\nEpisode complete:")
    print(f"  Progress: {logs['progress'][-1]:.1f}%")
    print(f"  Final deviation: {logs['deviation'][-1]:.1f}cm")
    print(f"  Max deviation: {np.max(logs['deviation']):.1f}cm")
    print(f"  Success: {info.get('is_success', False)}")
    print(f"  Steps: {len(logs['step'])}")

    # Generate save path if not specified
    if args.save_path is None:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        args.save_path = f"reward_analysis_{model_name}_{args.trajectory_type}.png"

    # Plot
    plot_rewards(logs, trajectory, info, save_path=args.save_path)

    env.close()


if __name__ == "__main__":
    main()