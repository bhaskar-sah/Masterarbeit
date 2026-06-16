import os
import time
from stable_baselines3 import PPO
from sympy.stats.sampling.sample_numpy import numpy

# Import the env
from env import PandaPushTrajectoryEnv

# ==================== CONFIGURATION ====================
# Define the trajectory and the exact version folder you want to test
# TRAJECTORY_TYPE = "straight"  # Should match training or test generalization
# TRAJECTORY_TYPE = "curved"  # Should match training or test generalization
TRAJECTORY_TYPE = "s_curve"

NUM_EPISODES = 10
VISUAL_DELAY = 0.00 # was 0.02  # Seconds between frames (0 for max speed)

# ==================== SETUP ====================
print("="*60)
print("TESTING: Panda Push Trajectory Following")
print("="*60)

# Create environment with rendering
env = PandaPushTrajectoryEnv(
    render_mode="human",
    trajectory_type= TRAJECTORY_TYPE, # "curved"
    controller_type = "motion_manual"
)
print("Environment created.")

# ==================== RUN TEST EPISODES ====================
print(f"\nRunning {NUM_EPISODES} test episodes...")
print(f"Trajectory type: {TRAJECTORY_TYPE}")
print("-"*60)

results = {
    "successes": 0,
    "total_rewards": [],
    "final_progress": [],
    "final_deviation": [],
}

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    step_count = 0

    while not terminated and not truncated:
        # Define simple action which only moves with 0.5 * v_max along the path
        action = numpy.array([0.5, 0.0, 0.0])  # [tangential, lateral, angular]

        # Take action
        obs, reward, terminated, truncated, info = env.step(action)

        # Render
        env.render()

        episode_reward += reward
        step_count += 1

        # Optional delay for visualization
        if VISUAL_DELAY > 0:
            time.sleep(VISUAL_DELAY)

    # Record results
    success = info.get("is_success", False)
    progress = info.get("progress", 0)
    deviation = info.get("deviation", 0)

    results["total_rewards"].append(episode_reward)
    results["final_progress"].append(progress)
    results["final_deviation"].append(deviation)
    if success:
        results["successes"] += 1

    status = "SUCCESS" if success else "FAILED"
    print(f"Episode {episode + 1}/{NUM_EPISODES}: {status} | "
          f"Progress: {progress:.1%} | "
          f"Deviation: {deviation:.3f}m | "
          f"Reward: {episode_reward:.1f} | "
          f"Steps: {step_count}")

# ==================== SUMMARY ====================
env.close()

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Success Rate: {results['successes']}/{NUM_EPISODES} "
      f"({100 * results['successes'] / NUM_EPISODES:.1f}%)")
print(f"Avg Reward: {sum(results['total_rewards']) / NUM_EPISODES:.1f}")
print(f"Avg Progress: {100 * sum(results['final_progress']) / NUM_EPISODES:.1f}%")
print(f"Avg Deviation: {sum(results['final_deviation']) / NUM_EPISODES:.4f}m")
print("=" * 60)