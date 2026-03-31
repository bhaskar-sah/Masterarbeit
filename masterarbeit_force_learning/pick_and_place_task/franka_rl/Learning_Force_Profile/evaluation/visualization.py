import matplotlib.pyplot as plt
import numpy as np
import time

# IMPORT YOUR SPECIFIC ENV FILE
from masterarbeit_force_learning.pick_and_place_task.franka_rl.Learning_Force_Profile.obsolete_files.push_learn_force_impedance_1 import PandaPushTrajectoryEnv


def run_debug_plot():
    # 1. Start the environment (Human mode to see the robot)
    env = PandaPushTrajectoryEnv(render_mode="human", trajectory_type="straight")
    obs, info = env.reset()

    # 2. Lists to store data for plotting
    deviations = []
    forces = []
    stiffness_x = []
    steps_list = []

    print("=" * 60)
    print("DEBUG VISUALIZATION STARTED")
    print("Goal: Verify if the robot steers back to the center.")
    print("=" * 60)

    # 3. Run one short episode
    for step in range(400):  # Run for 400 steps max

        # --- DUMMY ACTION ---
        # We don't need the AI brain here. We just want to test if your
        # math fix works. We command the robot to push STRAIGHT forward.
        # If your fix is good, the internal logic will automatically steer it.

        # Action = [vx, vy, kx, ky]
        # vx = 0.0 (No AI steering)
        # vy = -0.8 (Push forward)
        # k  = Random (To see lines on the plot)
        k_val = np.random.uniform(0.2, 0.8)
        action = np.array([0.0, -0.8, k_val, k_val], dtype=np.float32)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Collect Data
        deviations.append(info["deviation"])
        forces.append(info.get("force_magnitude", 0.0))

        # Re-calculate stiffness for plotting (just mapping action 0-1 to 50-2000)
        k_real = 50.0 + k_val * (2000.0 - 50.0)
        stiffness_x.append(k_real)
        steps_list.append(step)

        # Slow down slightly so you can watch the render window
        time.sleep(0.01)

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    env.close()

    # 4. PLOT THE RESULTS
    print("Plotting data...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Plot Deviation
    ax1.plot(steps_list, deviations, 'r-', linewidth=2)
    ax1.axhline(0, color='black', linestyle='--')
    ax1.set_ylabel('Deviation (m)')
    ax1.set_title('Does the deviation go back to Zero? (Red Line)')
    ax1.grid(True)

    # Plot Stiffness
    ax2.plot(steps_list, stiffness_x, 'b-', alpha=0.6)
    ax2.set_ylabel('Stiffness X')
    ax2.set_title('Stiffness Input (Action)')
    ax2.grid(True)

    # Plot Force
    ax3.plot(steps_list, forces, 'g-', alpha=0.6)
    ax3.set_ylabel('Force (N)')
    ax3.set_xlabel('Steps')
    ax3.set_title('Contact Force')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_debug_plot()