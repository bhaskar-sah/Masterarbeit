"""
Comprehensive Visualization for Push Task

This script visualizes:
1. Trajectory and bottle path (bird's eye view)
2. Deviation from trajectory over time
3. Force profile over time
4. Stiffness (learned) over time
5. Bottle tilt over time
6. Push state (push/slow/stop/settle)

Run this to debug and understand the robot's behavior!
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import time

# CHANGE THIS TO YOUR ENVIRONMENT FILE
from masterarbeit_force_learning.pick_and_place_task.franka_rl.Learning_Force_Profile.obsolete_files.push_learn_force_impedance_7 import PandaPushTrajectoryEnv


def run_visualization(max_steps=800, render=True, slow_motion=False):
    """
    Run one episode and visualize everything.

    Args:
        max_steps: Maximum steps to run
        render: Whether to show MuJoCo viewer
        slow_motion: Add delay between steps for viewing
    """

    # Create environment
    render_mode = "human" if render else None
    env = PandaPushTrajectoryEnv(render_mode=render_mode, trajectory_type="straight")
    obs, info = env.reset()

    # Get trajectory for plotting
    trajectory = env.trajectory.copy()

    # Data storage
    data = {
        'steps': [],
        'bottle_x': [],
        'bottle_y': [],
        'hand_x': [],
        'hand_y': [],
        'deviation': [],
        'progress': [],
        'force_x': [],
        'force_y': [],
        'force_mag': [],
        'stiffness_x': [],
        'stiffness_y': [],
        'tilt': [],
        'push_state': [],
        'reward': [],
        'is_touching': [],
    }

    print("=" * 60)
    print("VISUALIZATION RUNNING")
    print("=" * 60)

    # Run episode
    total_reward = 0
    for step in range(max_steps):
        # Action: Let agent push forward with moderate stiffness
        # You can also load a trained model here
        action = np.array([0.3, 0.0, 0.4, 0.4], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Collect data
        hand_pos = env.data.xpos[env.hand_body_id]
        bottle_pos = env.data.xpos[env.bottle_body_id]
        force = env._get_contact_force()

        data['steps'].append(step)
        data['bottle_x'].append(bottle_pos[0])
        data['bottle_y'].append(bottle_pos[1])
        data['hand_x'].append(hand_pos[0])
        data['hand_y'].append(hand_pos[1])
        data['deviation'].append(info.get('deviation', 0))
        data['progress'].append(info.get('progress', 0))
        data['force_x'].append(force[0])
        data['force_y'].append(force[1])
        data['force_mag'].append(info.get('force_magnitude', np.linalg.norm(force)))
        data['stiffness_x'].append(env.current_K[0])
        data['stiffness_y'].append(env.current_K[1])
        data['tilt'].append(info.get('bottle_tilt', env._get_bottle_tilt()))
        data['push_state'].append(info.get('push_state', 'unknown'))
        data['reward'].append(reward)
        data['is_touching'].append(1 if env._is_touching() else 0)

        if slow_motion:
            time.sleep(0.02)

        if terminated or truncated:
            result = "SUCCESS!" if info.get('is_success') else "FAILED"
            print(f"\nEpisode ended: {result}")
            print(f"Final progress: {info.get('progress', 0):.1%}")
            print(f"Final deviation: {info.get('deviation', 0):.3f}m")
            print(f"Total reward: {total_reward:.1f}")
            break

    env.close()

    # Convert to numpy
    for key in data:
        if key != 'push_state':
            data[key] = np.array(data[key])

    # Plot everything
    plot_results(data, trajectory)

    return data


def plot_results(data, trajectory):
    """Create comprehensive visualization plots."""

    fig = plt.figure(figsize=(16, 12))

    # ===== PLOT 1: Bird's Eye View (Trajectory) =====
    ax1 = fig.add_subplot(2, 3, 1)

    # Plot trajectory (target path)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b--', linewidth=2,
             label='Target Trajectory', alpha=0.7)

    # Plot actual bottle path with color based on deviation
    points = np.array([data['bottle_x'], data['bottle_y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by deviation
    norm = plt.Normalize(0, 0.1)
    lc = LineCollection(segments, cmap='RdYlGn_r', norm=norm)
    lc.set_array(data['deviation'][:-1])
    lc.set_linewidth(3)
    ax1.add_collection(lc)

    # Plot hand path
    ax1.plot(data['hand_x'], data['hand_y'], 'r-', linewidth=1,
             alpha=0.5, label='Hand Path')

    # Start and end markers
    ax1.scatter(data['bottle_x'][0], data['bottle_y'][0],
                c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(data['bottle_x'][-1], data['bottle_y'][-1],
                c='red', s=100, marker='x', label='End', zorder=5)
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1],
                c='blue', s=100, marker='*', label='Goal', zorder=5)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Bird\'s Eye View: Trajectory Following')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax1)
    cbar.set_label('Deviation (m)')

    # ===== PLOT 2: Deviation Over Time =====
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.fill_between(data['steps'], 0, data['deviation'],
                     color='red', alpha=0.3)
    ax2.plot(data['steps'], data['deviation'], 'r-', linewidth=2)
    ax2.axhline(0.05, color='orange', linestyle='--', label='Tolerance (5cm)')
    ax2.axhline(0.02, color='green', linestyle='--', label='Good (2cm)')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Deviation (m)')
    ax2.set_title('Deviation from Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # ===== PLOT 3: Progress Over Time =====
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(data['steps'], data['progress'] * 100, 'g-', linewidth=2)
    ax3.fill_between(data['steps'], 0, data['progress'] * 100,
                     color='green', alpha=0.3)
    ax3.axhline(95, color='blue', linestyle='--', label='Success (95%)')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Progress (%)')
    ax3.set_title('Progress Along Trajectory')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)

    # ===== PLOT 4: Force Profile =====
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(data['steps'], data['force_mag'], 'b-', linewidth=1.5,
             label='Force Magnitude')
    ax4.axhline(4, color='green', linestyle='--', alpha=0.7, label='Target (4N)')
    ax4.axhline(8, color='orange', linestyle='--', alpha=0.7, label='Max (8N)')

    # Shade contact regions
    contact_mask = data['is_touching'] > 0.5
    ax4.fill_between(data['steps'], 0, data['force_mag'].max() * 1.1,
                     where=contact_mask, color='blue', alpha=0.1, label='In Contact')

    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Force (N)')
    ax4.set_title('Contact Force Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    # ===== PLOT 5: Stiffness (Learned Force Profile) =====
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(data['steps'], data['stiffness_x'], 'b-', linewidth=1.5,
             label='Kx', alpha=0.8)
    ax5.plot(data['steps'], data['stiffness_y'], 'r-', linewidth=1.5,
             label='Ky', alpha=0.8)
    ax5.axhline(100, color='gray', linestyle=':', alpha=0.5, label='K_min')
    ax5.axhline(800, color='gray', linestyle=':', alpha=0.5, label='K_max')
    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Stiffness (N/m)')
    ax5.set_title('Learned Stiffness (Force Profile)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ===== PLOT 6: Bottle Tilt =====
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(data['steps'], data['tilt'], 'purple', linewidth=2)
    ax6.axhline(0.995, color='green', linestyle='--', alpha=0.7, label='OK (0.995)')
    ax6.axhline(0.975, color='orange', linestyle='--', alpha=0.7, label='Stop (0.975)')
    ax6.axhline(0.96, color='red', linestyle='--', alpha=0.7, label='Settle (0.96)')

    # Color regions based on push state
    ax6.fill_between(data['steps'], 0.9, 1.0,
                     where=data['tilt'] > 0.995, color='green', alpha=0.2)
    ax6.fill_between(data['steps'], 0.9, 1.0,
                     where=(data['tilt'] <= 0.995) & (data['tilt'] > 0.975),
                     color='yellow', alpha=0.2)
    ax6.fill_between(data['steps'], 0.9, 1.0,
                     where=data['tilt'] <= 0.975, color='red', alpha=0.2)

    ax6.set_xlabel('Steps')
    ax6.set_ylabel('Tilt (1.0 = upright)')
    ax6.set_title('Bottle Tilt (Stability)')
    ax6.legend(loc='lower left')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0.9, 1.01)

    plt.tight_layout()
    plt.savefig('push_visualization.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: push_visualization.png")
    plt.show()


def plot_force_vs_deviation(data):
    """
    Special plot: Force profile vs Deviation
    This is KEY for your thesis - shows the learned force response!
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with color by time
    scatter = ax.scatter(data['deviation'], data['force_mag'],
                         c=data['steps'], cmap='viridis',
                         alpha=0.6, s=20)

    # Add colorbar for time
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time (steps)')

    # Fit and plot trend line
    valid = data['force_mag'] > 0.5  # Only when in contact
    if valid.sum() > 10:
        z = np.polyfit(data['deviation'][valid], data['force_mag'][valid], 2)
        p = np.poly1d(z)
        x_line = np.linspace(0, data['deviation'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')

    ax.set_xlabel('Deviation from Trajectory (m)')
    ax.set_ylabel('Contact Force (N)')
    ax.set_title('Force Profile: How force changes with deviation\n(This is what RL learns!)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('force_vs_deviation.png', dpi=150, bbox_inches='tight')
    print("Force vs Deviation plot saved to: force_vs_deviation.png")
    plt.show()


def plot_stiffness_vs_deviation(data):
    """
    Special plot: Stiffness vs Deviation
    Shows how the learned stiffness responds to deviation!
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    K_avg = (data['stiffness_x'] + data['stiffness_y']) / 2

    scatter = ax.scatter(data['deviation'], K_avg,
                         c=data['progress'], cmap='coolwarm',
                         alpha=0.6, s=20)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Progress (%)')

    ax.set_xlabel('Deviation from Trajectory (m)')
    ax.set_ylabel('Average Stiffness K (N/m)')
    ax.set_title('Learned Stiffness Profile: K vs Deviation\n(Higher K = Stronger correction)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stiffness_vs_deviation.png', dpi=150, bbox_inches='tight')
    print("Stiffness vs Deviation plot saved to: stiffness_vs_deviation.png")
    plt.show()


def compare_episodes(env_class, n_episodes=5):
    """
    Run multiple episodes and compare their trajectories.
    Useful to see consistency and variability.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, n_episodes))

    all_progress = []
    all_deviation = []

    for i in range(n_episodes):
        print(f"\nRunning episode {i + 1}/{n_episodes}...")
        env = env_class(render_mode=None, trajectory_type="straight")
        obs, _ = env.reset()
        trajectory = env.trajectory.copy()

        bottle_path = []
        deviations = []

        for step in range(1500):
            action = np.array([0.3, 0.0, 0.4, 0.4], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            bottle_pos = env.data.xpos[env.bottle_body_id]
            bottle_path.append(bottle_pos[:2].copy())
            deviations.append(info.get('deviation', 0))

            if terminated or truncated:
                break

        env.close()

        bottle_path = np.array(bottle_path)
        deviations = np.array(deviations)

        # Plot trajectory
        axes[0].plot(bottle_path[:, 0], bottle_path[:, 1],
                     color=colors[i], linewidth=1.5, alpha=0.7,
                     label=f'Episode {i + 1}')

        # Plot deviation
        axes[1].plot(deviations, color=colors[i], linewidth=1.5, alpha=0.7,
                     label=f'Episode {i + 1}')

        all_progress.append(info.get('progress', 0))
        all_deviation.append(deviations[-1] if len(deviations) > 0 else 0)

    # Plot target trajectory
    axes[0].plot(trajectory[:, 0], trajectory[:, 1], 'k--', linewidth=2,
                 label='Target', alpha=0.5)

    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Bottle Paths Across Episodes')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0.05, color='red', linestyle='--', alpha=0.5, label='Tolerance')
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Deviation (m)')
    axes[1].set_title('Deviation Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('episode_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to: episode_comparison.png")
    print(f"Average progress: {np.mean(all_progress):.1%}")
    print(f"Average final deviation: {np.mean(all_deviation):.3f}m")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("PUSH TASK VISUALIZATION")
    print("=" * 60)
    print("\nOptions:")
    print("1. Run single episode with full visualization")
    print("2. Run single episode (no render, just plots)")
    print("3. Compare multiple episodes")
    print("4. Run and show force vs deviation analysis")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        data = run_visualization(max_steps=1500, render=True, slow_motion=True)
        plot_force_vs_deviation(data)
        plot_stiffness_vs_deviation(data)
    elif choice == "2":
        data = run_visualization(max_steps=1500, render=False, slow_motion=False)
        plot_force_vs_deviation(data)
        plot_stiffness_vs_deviation(data)
    elif choice == "3":
        compare_episodes(PandaPushTrajectoryEnv, n_episodes=5)
    elif choice == "4":
        data = run_visualization(max_steps=1500, render=False, slow_motion=False)
        plot_force_vs_deviation(data)
        plot_stiffness_vs_deviation(data)
    else:
        print("Running default visualization...")
        data = run_visualization(max_steps=1500, render=True, slow_motion=False)