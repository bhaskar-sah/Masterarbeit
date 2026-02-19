import os
import time
from stable_baselines3 import PPO
from push_with_finger_learn_force_impedance_14_v01_curriculum_learning import PandaPushTrajectoryEnv

# ==================== CONFIGURATION ====================
MODEL_NAME = "push_curriculum_learning_02_final.zip"  # Your curriculum-trained model
TEST_ALL_TRAJECTORIES = True  # NEW: Test all trajectory types
NUM_EPISODES = 5  # Episodes per trajectory type
VISUAL_DELAY = 0.02

# ==================== SETUP ====================
print("=" * 60)
print("TESTING: Panda Push - CURRICULUM TRAINED MODEL")
print("=" * 60)

# ==================== LOAD MODEL ====================
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "saved_models", MODEL_NAME)

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Available models in saved_models/:")
    models_dir = os.path.join(current_dir, "saved_models")
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".zip"):
                print(f"  - {f}")
    exit()

model = PPO.load(model_path)
print(f"Model loaded from: {model_path}")

# ==================== TEST ALL TRAJECTORIES ====================
if TEST_ALL_TRAJECTORIES:
    trajectory_types = ["straight", "curved", "s_curve"]
else:
    trajectory_types = ["straight"]  # Single trajectory

# Store results for all trajectories
all_results = {}

for traj_type in trajectory_types:
    print(f"\n{'=' * 60}")
    print(f"TESTING: {traj_type.upper()}")
    print(f"{'=' * 60}")

    # Create environment for this trajectory type
    env = PandaPushTrajectoryEnv(
        render_mode="human",
        trajectory_type=traj_type  # Fixed trajectory type for testing
    )

    results = {
        "successes": 0,
        "total_rewards": [],
        "final_progress": [],
        "final_deviation": [],
    }

    for episode in range(NUM_EPISODES):
        # Use options to override trajectory type (ensures correct type)
        obs, info = env.reset(options={"trajectory_type": traj_type})
        terminated = False
        truncated = False
        episode_reward = 0
        step_count = 0

        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            episode_reward += reward
            step_count += 1

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

        status = "SUCCESS ✓" if success else "FAILED ✗"
        reason = ""
        if not success:
            if info.get("bottle_fallen"):
                reason = "(fallen)"
            elif info.get("off_path"):
                reason = "(off path)"
            else:
                reason = "(timeout)"

        print(f"  Episode {episode + 1}/{NUM_EPISODES}: {status} {reason} | "
              f"Progress: {progress:.1%} | "
              f"Deviation: {deviation:.3f}m | "
              f"Reward: {episode_reward:.1f} | "
              f"Steps: {step_count}")

    # Store results for this trajectory type
    all_results[traj_type] = {
        "success_rate": results["successes"] / NUM_EPISODES * 100,
        "avg_reward": sum(results["total_rewards"]) / NUM_EPISODES,
        "avg_progress": sum(results["final_progress"]) / NUM_EPISODES * 100,
        "avg_deviation": sum(results["final_deviation"]) / NUM_EPISODES,
        "successes": results["successes"],
        "total": NUM_EPISODES,
    }

    print(f"\n  {traj_type} Summary: {results['successes']}/{NUM_EPISODES} "
          f"({all_results[traj_type]['success_rate']:.0f}%)")

    env.close()

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 60)
print("FINAL TEST SUMMARY")
print("=" * 60)
print(f"{'Trajectory':<12} {'Success Rate':<15} {'Avg Progress':<15} {'Avg Deviation':<15}")
print("-" * 60)

total_successes = 0
total_episodes = 0

for traj_type, data in all_results.items():
    status = "✓" if data["success_rate"] >= 80 else "△" if data["success_rate"] >= 50 else "✗"
    print(f"{traj_type:<12} {data['successes']}/{data['total']} ({data['success_rate']:>5.1f}%) {status}   "
          f"{data['avg_progress']:>10.1f}%      "
          f"{data['avg_deviation']:>10.4f}m")
    total_successes += data["successes"]
    total_episodes += data["total"]

print("-" * 60)
overall_rate = total_successes / total_episodes * 100
print(f"{'OVERALL':<12} {total_successes}/{total_episodes} ({overall_rate:>5.1f}%)")
print("=" * 60)

# Performance assessment
print("\nPERFORMANCE ASSESSMENT:")
if all_results["straight"]["success_rate"] >= 80:
    print("  ✓ Straight trajectory: GOOD")
else:
    print("  ✗ Straight trajectory: NEEDS IMPROVEMENT")

if all_results["curved"]["success_rate"] >= 60:
    print("  ✓ Curved trajectory: GOOD")
else:
    print("  ✗ Curved trajectory: NEEDS IMPROVEMENT")

if all_results["s_curve"]["success_rate"] >= 50:
    print("  ✓ S-curve trajectory: GOOD")
else:
    print("  ✗ S-curve trajectory: NEEDS IMPROVEMENT")

print("=" * 60)