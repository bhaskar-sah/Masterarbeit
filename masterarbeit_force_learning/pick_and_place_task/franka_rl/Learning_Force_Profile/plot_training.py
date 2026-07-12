# """
# Plot Training Diagnostics from CSV Step Log.

# Usage:
#     python plot_training.py logs/training_run_01.csv

# Generates a multi-panel figure showing:
# 1. Hand & Bottle XY trajectory
# 2. Progress over time
# 3. Force analysis (commanded vs measured)
# 4. Action distributions
# 5. Reward component breakdown
# 6. Position error & deviation
# """

# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec


# def load_data(filepath):
#     """Load CSV and return DataFrame."""
#     df = pd.read_csv(filepath)
#     print(f"Loaded {len(df)} steps from {filepath}")
#     print(f"Episodes: {df['episode_num'].nunique()}")
#     print(f"Columns: {list(df.columns)}")
#     return df


# def plot_single_episode(df, episode_num=None, save_path=None):
#     """Plot detailed analysis for a single episode."""
#     if episode_num is not None:
#         ep = df[df["episode_num"] == episode_num]
#     else:
#         # Use last complete episode
#         episode_num = df["episode_num"].max()
#         ep = df[df["episode_num"] == episode_num]

#     if len(ep) == 0:
#         print(f"No data for episode {episode_num}")
#         return

#     steps = ep["episode_step"].values
#     print(f"\nPlotting episode {episode_num} ({len(ep)} steps)")

#     fig = plt.figure(figsize=(20, 16))
#     fig.suptitle(f"Episode {episode_num} Analysis ({len(ep)} steps)", fontsize=16)
#     gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

#     # ── 1. XY Trajectory ──
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.plot(ep["hand_x"], ep["hand_y"], "b-", alpha=0.5, label="Hand", linewidth=0.8)
#     ax1.plot(ep["bottle_x"], ep["bottle_y"], "r-", alpha=0.7, label="Bottle", linewidth=1.2)
#     ax1.plot(ep["hand_x"].iloc[0], ep["hand_y"].iloc[0], "bo", markersize=8, label="Hand start")
#     ax1.plot(ep["bottle_x"].iloc[0], ep["bottle_y"].iloc[0], "ro", markersize=8, label="Bottle start")
#     ax1.plot(0.4, -0.4, "r*", markersize=15, label="Goal")
#     ax1.set_xlabel("X (m)")
#     ax1.set_ylabel("Y (m)")
#     ax1.set_title("XY Trajectory")
#     ax1.legend(fontsize=7)
#     ax1.set_aspect("equal")
#     ax1.grid(True, alpha=0.3)

#     # ── 2. Progress over time ──
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax2.plot(steps, ep["progress"] * 100, "g-", linewidth=1.5)
#     ax2.set_xlabel("Step")
#     ax2.set_ylabel("Progress (%)")
#     ax2.set_title("Progress Along Trajectory")
#     ax2.grid(True, alpha=0.3)
#     ax2.set_ylim(-5, 105)

#     # ── 3. Deviation over time ──
#     ax3 = fig.add_subplot(gs[0, 2])
#     ax3.plot(steps, ep["deviation_m"] * 100, "orange", linewidth=1)
#     ax3.axhline(y=5.0, color="r", linestyle="--", alpha=0.5, label="5cm tolerance")
#     ax3.set_xlabel("Step")
#     ax3.set_ylabel("Deviation (cm)")
#     ax3.set_title("Path Deviation")
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)

#     # ── 4. Force: Commanded vs Measured ──
#     ax4 = fig.add_subplot(gs[1, 0])
#     ax4.plot(steps, ep["F_des_mag"], "b-", alpha=0.6, label="F_des (commanded)", linewidth=0.8)
#     ax4.plot(steps, ep["F_measured_mag"], "r-", alpha=0.6, label="F_measured", linewidth=0.8)
#     ax4.plot(steps, ep["F_cmd_mag"], "g-", alpha=0.4, label="F_cmd (total)", linewidth=0.5)
#     ax4.set_xlabel("Step")
#     ax4.set_ylabel("Force (N)")
#     ax4.set_title("Force Analysis")
#     ax4.legend(fontsize=7)
#     ax4.grid(True, alpha=0.3)

#     # ── 5. Raw Actions over time ──
#     ax5 = fig.add_subplot(gs[1, 1])
#     ax5.plot(steps, ep["a_vx"], alpha=0.5, label="a_vx", linewidth=0.5)
#     ax5.plot(steps, ep["a_vy"], alpha=0.5, label="a_vy", linewidth=0.5)
#     ax5.plot(steps, ep["a_wz"], alpha=0.5, label="a_wz", linewidth=0.5)
#     ax5.plot(steps, ep["a_f"], alpha=0.7, label="a_f", linewidth=1.0, color="red")
#     ax5.set_xlabel("Step")
#     ax5.set_ylabel("Action value [-1, 1]")
#     ax5.set_title("Raw RL Actions")
#     ax5.legend(fontsize=7)
#     ax5.set_ylim(-1.2, 1.2)
#     ax5.grid(True, alpha=0.3)

#     # ── 6. Action Histograms ──
#     ax6 = fig.add_subplot(gs[1, 2])
#     ax6.hist(ep["a_vx"], bins=30, alpha=0.4, label="vx", density=True)
#     ax6.hist(ep["a_vy"], bins=30, alpha=0.4, label="vy", density=True)
#     ax6.hist(ep["a_f"], bins=30, alpha=0.6, label="f", density=True, color="red")
#     ax6.set_xlabel("Action value")
#     ax6.set_ylabel("Density")
#     ax6.set_title("Action Distributions")
#     ax6.legend(fontsize=7)
#     ax6.grid(True, alpha=0.3)

#     # ── 7. Reward Components ──
#     ax7 = fig.add_subplot(gs[2, 0:2])
#     ax7.plot(steps, ep["r_progress"], label="r_progress", linewidth=0.8)
#     ax7.plot(steps, ep["r_deviation"], label="r_deviation", linewidth=0.8)
#     ax7.plot(steps, ep["r_stability"], label="r_stability", linewidth=0.8)
#     ax7.plot(steps, ep["r_contact"], label="r_contact", linewidth=0.8)
#     ax7.plot(steps, ep["r_smoothness"], label="r_smoothness", linewidth=0.8)
#     ax7.plot(steps, ep["reward_total"], label="TOTAL", linewidth=1.5, color="black", alpha=0.7)
#     ax7.set_xlabel("Step")
#     ax7.set_ylabel("Reward")
#     ax7.set_title("Reward Components Over Time")
#     ax7.legend(fontsize=7, ncol=3)
#     ax7.grid(True, alpha=0.3)

#     # ── 8. Cumulative Reward ──
#     ax8 = fig.add_subplot(gs[2, 2])
#     cumulative = ep["reward_total"].cumsum()
#     ax8.plot(steps, cumulative, "k-", linewidth=1.5)
#     ax8.set_xlabel("Step")
#     ax8.set_ylabel("Cumulative Reward")
#     ax8.set_title(f"Cumulative Reward (Total: {cumulative.iloc[-1]:.1f})")
#     ax8.grid(True, alpha=0.3)

#     # ── 9. Hand-to-Bottle Distance ──
#     ax9 = fig.add_subplot(gs[3, 0])
#     ax9.plot(steps, ep["hand_to_bottle_dist"] * 100, "purple", linewidth=0.8)
#     ax9.set_xlabel("Step")
#     ax9.set_ylabel("Distance (cm)")
#     ax9.set_title("Hand-to-Bottle Distance")
#     ax9.grid(True, alpha=0.3)

#     # ── 10. Contact over time ──
#     ax10 = fig.add_subplot(gs[3, 1])
#     ax10.fill_between(steps, 0, ep["is_touching"], alpha=0.4, color="green", label="Contact")
#     contact_pct = ep["is_touching"].mean() * 100
#     ax10.set_xlabel("Step")
#     ax10.set_ylabel("Contact")
#     ax10.set_title(f"Contact ({contact_pct:.1f}% of episode)")
#     ax10.set_ylim(-0.1, 1.3)
#     ax10.grid(True, alpha=0.3)

#     # ── 11. Position Error (p_des vs p_actual) ──
#     ax11 = fig.add_subplot(gs[3, 2])
#     ax11.plot(steps, ep["p_error_mag"] * 1000, "brown", linewidth=0.8)
#     ax11.set_xlabel("Step")
#     ax11.set_ylabel("Error (mm)")
#     ax11.set_title("Position Tracking Error |p_des - p|")
#     ax11.grid(True, alpha=0.3)

#     # ── Summary stats ──
#     print(f"\n{'='*50}")
#     print(f"EPISODE {episode_num} SUMMARY")
#     print(f"{'='*50}")
#     print(f"  Final progress:     {ep['progress'].iloc[-1]*100:.1f}%")
#     print(f"  Mean deviation:     {ep['deviation_m'].mean()*100:.2f} cm")
#     print(f"  Max deviation:      {ep['deviation_m'].max()*100:.2f} cm")
#     print(f"  Contact rate:       {contact_pct:.1f}%")
#     print(f"  Mean F_des:         {ep['F_des_mag'].mean():.2f} N")
#     print(f"  Mean F_measured:    {ep['F_measured_mag'].mean():.2f} N")
#     print(f"  Mean F_cmd:         {ep['F_cmd_mag'].mean():.2f} N")
#     print(f"  Total reward:       {ep['reward_total'].sum():.1f}")
#     print(f"  Mean action f:      {ep['a_f'].mean():.3f}")
#     print(f"  Std action f:       {ep['a_f'].std():.3f}")
#     print(f"  Mean p_error:       {ep['p_error_mag'].mean()*1000:.2f} mm")
#     print(f"{'='*50}")

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"Plot saved to: {save_path}")
#     else:
#         plt.show()


# def plot_training_overview(df, save_path=None):
#     """Plot overview across all episodes."""
#     episodes = df.groupby("episode_num")

#     ep_nums = []
#     final_progress = []
#     mean_reward = []
#     total_reward = []
#     contact_rate = []
#     mean_f_des = []
#     mean_f_meas = []
#     mean_deviation = []

#     for ep_num, ep in episodes:
#         ep_nums.append(ep_num)
#         final_progress.append(ep["progress"].iloc[-1] * 100)
#         mean_reward.append(ep["reward_total"].mean())
#         total_reward.append(ep["reward_total"].sum())
#         contact_rate.append(ep["is_touching"].mean() * 100)
#         mean_f_des.append(ep["F_des_mag"].mean())
#         mean_f_meas.append(ep["F_measured_mag"].mean())
#         mean_deviation.append(ep["deviation_m"].mean() * 100)

#     fig, axes = plt.subplots(3, 2, figsize=(14, 12))
#     fig.suptitle(f"Training Overview ({len(ep_nums)} episodes)", fontsize=14)

#     axes[0, 0].plot(ep_nums, final_progress, "g.-")
#     axes[0, 0].set_ylabel("Final Progress (%)")
#     axes[0, 0].set_title("Progress per Episode")
#     axes[0, 0].grid(True, alpha=0.3)

#     axes[0, 1].plot(ep_nums, total_reward, "b.-")
#     axes[0, 1].set_ylabel("Total Reward")
#     axes[0, 1].set_title("Episode Return")
#     axes[0, 1].grid(True, alpha=0.3)

#     axes[1, 0].plot(ep_nums, contact_rate, "m.-")
#     axes[1, 0].set_ylabel("Contact Rate (%)")
#     axes[1, 0].set_title("Contact Maintenance")
#     axes[1, 0].grid(True, alpha=0.3)

#     axes[1, 1].plot(ep_nums, mean_f_des, "b.-", alpha=0.7, label="F_des")
#     axes[1, 1].plot(ep_nums, mean_f_meas, "r.-", alpha=0.7, label="F_measured")
#     axes[1, 1].set_ylabel("Force (N)")
#     axes[1, 1].set_title("Mean Force per Episode")
#     axes[1, 1].legend()
#     axes[1, 1].grid(True, alpha=0.3)

#     axes[2, 0].plot(ep_nums, mean_deviation, "orange", marker=".")
#     axes[2, 0].set_ylabel("Deviation (cm)")
#     axes[2, 0].set_xlabel("Episode")
#     axes[2, 0].set_title("Mean Path Deviation")
#     axes[2, 0].grid(True, alpha=0.3)

#     axes[2, 1].plot(ep_nums, mean_reward, "k.-")
#     axes[2, 1].set_ylabel("Mean Step Reward")
#     axes[2, 1].set_xlabel("Episode")
#     axes[2, 1].set_title("Mean Reward per Step")
#     axes[2, 1].grid(True, alpha=0.3)

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"Overview saved to: {save_path}")
#     else:
#         plt.show()


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python plot_training.py <csv_file> [episode_num]")
#         print("  python plot_training.py logs/run01.csv          # plot last episode")
#         print("  python plot_training.py logs/run01.csv 5        # plot episode 5")
#         print("  python plot_training.py logs/run01.csv overview  # plot all episodes")
#         sys.exit(1)

#     filepath = sys.argv[1]
#     df = load_data(filepath)

#     if len(sys.argv) > 2 and sys.argv[2] == "overview":
#         plot_training_overview(df, save_path=filepath.replace(".csv", "_overview.png"))
#     elif len(sys.argv) > 2:
#         ep_num = int(sys.argv[2])
#         plot_single_episode(df, episode_num=ep_num,
#                            save_path=filepath.replace(".csv", f"_ep{ep_num}.png"))
#     else:
#         plot_single_episode(df, save_path=filepath.replace(".csv", "_latest.png"))



####################################################################################################
"""
Plot Training Diagnostics from CSV Step Log.

Usage:
    python plot_training.py logs/training_run_01.csv

Generates a multi-panel figure showing:
1. Hand & Bottle XY trajectory
2. Progress over time
3. Force analysis (commanded vs measured)
4. Action distributions
5. Reward component breakdown
6. Position error & deviation
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_data(filepath):
    """Load CSV and return DataFrame."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} steps from {filepath}")
    print(f"Episodes: {df['episode_num'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    return df


def plot_single_episode(df, episode_num=None, save_path=None):
    """Plot detailed analysis for a single episode."""
    if episode_num is not None:
        ep = df[df["episode_num"] == episode_num]
    else:
        # Use last complete episode
        episode_num = df["episode_num"].max()
        ep = df[df["episode_num"] == episode_num]

    if len(ep) == 0:
        print(f"No data for episode {episode_num}")
        return

    steps = ep["episode_step"].values
    print(f"\nPlotting episode {episode_num} ({len(ep)} steps)")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Episode {episode_num} Analysis ({len(ep)} steps)", fontsize=16)
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── 1. XY Trajectory ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep["hand_x"], ep["hand_y"], "b-", alpha=0.5, label="Hand", linewidth=0.8)
    ax1.plot(ep["bottle_x"], ep["bottle_y"], "r-", alpha=0.7, label="Bottle", linewidth=1.2)
    ax1.plot(ep["hand_x"].iloc[0], ep["hand_y"].iloc[0], "bo", markersize=8, label="Hand start")
    ax1.plot(ep["bottle_x"].iloc[0], ep["bottle_y"].iloc[0], "ro", markersize=8, label="Bottle start")
    ax1.plot(0.4, -0.4, "r*", markersize=15, label="Goal")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("XY Trajectory")
    ax1.legend(fontsize=7)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # ── 2. Progress over time ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, ep["progress"] * 100, "g-", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Progress (%)")
    ax2.set_title("Progress Along Trajectory")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    # ── 3. Deviation over time ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, ep["deviation_m"] * 100, "orange", linewidth=1)
    ax3.axhline(y=5.0, color="r", linestyle="--", alpha=0.5, label="5cm tolerance")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Deviation (cm)")
    ax3.set_title("Path Deviation")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ── 4. Force: Commanded vs Measured ──
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(steps, ep["F_measured_mag"], "r-", alpha=0.8, label="F_measured", linewidth=1.0)
    ax4.plot(steps, ep["F_cmd_mag"], "g-", alpha=0.6, label="F_cmd (total)", linewidth=0.8)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Force (N)")
    ax4.set_title("Force Analysis")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # ── 5. Raw Actions over time ──
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(steps, ep["a_vx"], alpha=0.7, label="a_vx", linewidth=0.8)
    ax5.plot(steps, ep["a_vy"], alpha=0.7, label="a_vy", linewidth=0.8)
    ax5.plot(steps, ep["a_wz"], alpha=0.7, label="a_wz", linewidth=0.8)
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Action value [-1, 1]")
    ax5.set_title("Raw RL Actions")
    ax5.legend(fontsize=7)
    ax5.set_ylim(-1.2, 1.2)
    ax5.grid(True, alpha=0.3)

    # ── 6. Action Histograms ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(ep["a_vx"], bins=30, alpha=0.4, label="vx", density=True)
    ax6.hist(ep["a_vy"], bins=30, alpha=0.4, label="vy", density=True)
    ax6.hist(ep["a_wz"], bins=30, alpha=0.4, label="wz", density=True)
    ax6.set_xlabel("Action value")
    ax6.set_ylabel("Density")
    ax6.set_title("Action Distributions")
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    # ── 7. Reward Components ──
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.plot(steps, ep["r_progress"], label="r_progress", linewidth=0.8)
    ax7.plot(steps, ep["r_deviation"], label="r_deviation", linewidth=0.8)
    ax7.plot(steps, ep["r_stability"], label="r_stability", linewidth=0.8)
    ax7.plot(steps, ep["r_contact"], label="r_contact", linewidth=0.8)
    ax7.plot(steps, ep["r_alignment"], label="r_alignment", linewidth=0.8)
    ax7.plot(steps, ep["r_position"], label="r_position", linewidth=0.8)
    ax7.plot(steps, ep["reward_total"], label="TOTAL", linewidth=1.5, color="black", alpha=0.7)
    ax7.set_xlabel("Step")
    ax7.set_ylabel("Reward")
    ax7.set_title("Reward Components Over Time")
    ax7.legend(fontsize=7, ncol=4)
    ax7.grid(True, alpha=0.3)

    # ── 8. Cumulative Reward ──
    ax8 = fig.add_subplot(gs[2, 2])
    cumulative = ep["reward_total"].cumsum()
    ax8.plot(steps, cumulative, "k-", linewidth=1.5)
    ax8.set_xlabel("Step")
    ax8.set_ylabel("Cumulative Reward")
    ax8.set_title(f"Cumulative Reward (Total: {cumulative.iloc[-1]:.1f})")
    ax8.grid(True, alpha=0.3)

    # ── 9. Hand-to-Bottle Distance ──
    ax9 = fig.add_subplot(gs[3, 0])
    ax9.plot(steps, ep["hand_to_bottle_dist"] * 100, "purple", linewidth=0.8)
    ax9.set_xlabel("Step")
    ax9.set_ylabel("Distance (cm)")
    ax9.set_title("Hand-to-Bottle Distance")
    ax9.grid(True, alpha=0.3)

    # ── 10. Contact over time ──
    ax10 = fig.add_subplot(gs[3, 1])
    ax10.fill_between(steps, 0, ep["is_touching"], alpha=0.4, color="green", label="Contact")
    contact_pct = ep["is_touching"].mean() * 100
    ax10.set_xlabel("Step")
    ax10.set_ylabel("Contact")
    ax10.set_title(f"Contact ({contact_pct:.1f}% of episode)")
    ax10.set_ylim(-0.1, 1.3)
    ax10.grid(True, alpha=0.3)

    # ── 11. Position Error (p_des vs p_actual) ──
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.plot(steps, ep["p_error_mag"] * 1000, "brown", linewidth=0.8)
    ax11.set_xlabel("Step")
    ax11.set_ylabel("Error (mm)")
    ax11.set_title("Position Tracking Error |p_des - p|")
    ax11.grid(True, alpha=0.3)

    # ── Summary stats ──
    print(f"\n{'='*50}")
    print(f"EPISODE {episode_num} SUMMARY")
    print(f"{'='*50}")
    print(f"  Final progress:     {ep['progress'].iloc[-1]*100:.1f}%")
    print(f"  Mean deviation:     {ep['deviation_m'].mean()*100:.2f} cm")
    print(f"  Max deviation:      {ep['deviation_m'].max()*100:.2f} cm")
    print(f"  Contact rate:       {contact_pct:.1f}%")
    print(f"  Mean F_measured:    {ep['F_measured_mag'].mean():.2f} N")
    print(f"  Mean F_cmd:         {ep['F_cmd_mag'].mean():.2f} N")
    print(f"  Total reward:       {ep['reward_total'].sum():.1f}")
    print(f"  Mean action vx:     {ep['a_vx'].mean():.3f}")
    print(f"  Std action vx:      {ep['a_vx'].std():.3f}")
    print(f"  Mean p_error:       {ep['p_error_mag'].mean()*1000:.2f} mm")
    print(f"{'='*50}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    
def plot_supervisor_tracking(df, episode_num=None, save_path=None):
    """Plot the exact Desired vs Actual tracking graphs requested by the supervisor."""
    if episode_num is not None:
        ep = df[df["episode_num"] == episode_num]
    else:
        episode_num = df["episode_num"].max()
        ep = df[df["episode_num"] == episode_num]

    if len(ep) == 0:
        print(f"No data for episode {episode_num}")
        return

    steps = ep["episode_step"].values
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Controller Tracking Analysis (Episode {episode_num})", fontsize=16)

    # --- 1. p_des vs p (X-Axis) ---
    axs[0, 0].plot(steps, ep["p_des_x"], label="Target (p_des_x)", linestyle="--", color="blue")
    axs[0, 0].plot(steps, ep["hand_x"], label="Actual Hand (hand_x)", color="cyan")
    axs[0, 0].set_title("Position Tracking (X-Axis)")
    axs[0, 0].set_ylabel("Position (m)")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # --- 2. p_des vs p (Y-Axis) ---
    axs[0, 1].plot(steps, ep["p_des_y"], label="Target (p_des_y)", linestyle="--", color="red")
    axs[0, 1].plot(steps, ep["hand_y"], label="Actual Hand (hand_y)", color="orange")
    axs[0, 1].set_title("Position Tracking (Y-Axis)")
    axs[0, 1].set_ylabel("Position (m)")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # --- 3. v_des vs v (X-Axis) ---
    axs[1, 0].plot(steps, ep["vx_mps"], label="Commanded Vel (v_des_x)", linestyle="--", color="blue")
    # Using v_curr_x which we just added to your StepLogger!
    axs[1, 0].plot(steps, ep["v_curr_x"], label="Actual Vel (v_curr_x)", color="cyan")
    axs[1, 0].set_title("Velocity Tracking (X-Axis)")
    axs[1, 0].set_ylabel("Velocity (m/s)")
    axs[1, 0].set_xlabel("Step")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # --- 4. v_des vs v (Y-Axis) ---
    axs[1, 1].plot(steps, ep["vy_mps"], label="Commanded Vel (v_des_y)", linestyle="--", color="red")
    axs[1, 1].plot(steps, ep["v_curr_y"], label="Actual Vel (v_curr_y)", color="orange")
    axs[1, 1].set_title("Velocity Tracking (Y-Axis)")
    axs[1, 1].set_ylabel("Velocity (m/s)")
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Tracking plot saved to: {save_path}")
    else:
        plt.show()

def plot_training_overview(df, save_path=None):
    """Plot overview across all episodes."""
    episodes = df.groupby("episode_num")

    ep_nums = []
    final_progress = []
    mean_reward = []
    total_reward = []
    contact_rate = []
    mean_f_cmd = []
    mean_f_meas = []
    mean_deviation = []

    for ep_num, ep in episodes:
        ep_nums.append(ep_num)
        final_progress.append(ep["progress"].iloc[-1] * 100)
        mean_reward.append(ep["reward_total"].mean())
        total_reward.append(ep["reward_total"].sum())
        contact_rate.append(ep["is_touching"].mean() * 100)
        mean_f_cmd.append(ep["F_cmd_mag"].mean())
        mean_f_meas.append(ep["F_measured_mag"].mean())
        mean_deviation.append(ep["deviation_m"].mean() * 100)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Training Overview ({len(ep_nums)} episodes)", fontsize=14)

    axes[0, 0].plot(ep_nums, final_progress, "g.-")
    axes[0, 0].set_ylabel("Final Progress (%)")
    axes[0, 0].set_title("Progress per Episode")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ep_nums, total_reward, "b.-")
    axes[0, 1].set_ylabel("Total Reward")
    axes[0, 1].set_title("Episode Return")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ep_nums, contact_rate, "m.-")
    axes[1, 0].set_ylabel("Contact Rate (%)")
    axes[1, 0].set_title("Contact Maintenance")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ep_nums, mean_f_cmd, "g.-", alpha=0.7, label="F_cmd")
    axes[1, 1].plot(ep_nums, mean_f_meas, "r.-", alpha=0.7, label="F_measured")
    axes[1, 1].set_ylabel("Force (N)")
    axes[1, 1].set_title("Mean Force per Episode")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(ep_nums, mean_deviation, "orange", marker=".")
    axes[2, 0].set_ylabel("Deviation (cm)")
    axes[2, 0].set_xlabel("Episode")
    axes[2, 0].set_title("Mean Path Deviation")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(ep_nums, mean_reward, "k.-")
    axes[2, 1].set_ylabel("Mean Step Reward")
    axes[2, 1].set_xlabel("Episode")
    axes[2, 1].set_title("Mean Reward per Step")
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Overview saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training.py <csv_file> [episode_num|overview|tracking]")
        print("  python plot_training.py logs/run01.csv          # plot last episode dashboard")
        print("  python plot_training.py logs/run01.csv overview # plot all episodes")
        print("  python plot_training.py logs/run01.csv tracking # plot supervisor tracking graphs")
        sys.exit(1)

    filepath = sys.argv[1]
    df = load_data(filepath)

    if len(sys.argv) > 2:
        if sys.argv[2] == "overview":
            plot_training_overview(df, save_path=filepath.replace(".csv", "_overview.png"))
        elif sys.argv[2] == "tracking":
            plot_supervisor_tracking(df, save_path=filepath.replace(".csv", "_tracking.png"))
        else:
            ep_num = int(sys.argv[2])
            plot_single_episode(df, episode_num=ep_num,
                               save_path=filepath.replace(".csv", f"_ep{ep_num}.png"))
    else:
        plot_single_episode(df, save_path=filepath.replace(".csv", "_latest.png"))