"""
Test Trained Reach Model.

Loads a trained reach model and runs test episodes.
The reach phase is trajectory-agnostic — no trajectory type needed.

Usage:
    python reach_test.py
"""

import os
import time
from stable_baselines3 import PPO
from reach_env import PandaReachEnv


# ==================== CONFIGURATION ====================
MODEL_NAME = "reach_model_15.zip"
NUM_EPISODES = 50
VISUAL_DELAY = 0.1  # Seconds between frames (0 for max speed)

# ==================== SETUP ====================
print("=" * 60)
print("TESTING: Panda Reach to Bottle")
print("=" * 60)

env = PandaReachEnv(render_mode="human")
print("Environment created.")

# ==================== LOAD MODEL ====================
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "saved_models", MODEL_NAME)

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    save_dir = os.path.join(current_dir, "saved_models")
    if os.path.exists(save_dir):
        print("Available models:")
        for f in sorted(os.listdir(save_dir)):
            if f.endswith(".zip"):
                print(f"  - {f}")
    env.close()
    exit()

try:
    model = PPO.load(model_path, env=env)
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    env.close()
    exit()

# ==================== RUN TEST EPISODES ====================
print(f"\nRunning {NUM_EPISODES} test episodes...")
print("-" * 60)

results = {
    "successes": 0,
    "total_rewards": [],
    "steps_to_contact": [],
}

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
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

    success = info.get("reach_success", False)
    results["total_rewards"].append(episode_reward)
    results["steps_to_contact"].append(step_count)
    if success:
        results["successes"] += 1

    status = "SUCCESS" if success else "FAILED"
    fail_reason = ""
    if not success:
        if info.get("bottle_knocked"):
            fail_reason = " (bottle knocked)"
        else:
            fail_reason = " (timeout)"

    print(f"Episode {episode + 1}/{NUM_EPISODES}: {status}{fail_reason} | "
          f"Steps: {step_count} | Reward: {episode_reward:.1f}")

# ==================== SUMMARY ====================
env.close()

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Success Rate: {results['successes']}/{NUM_EPISODES} "
      f"({100 * results['successes'] / NUM_EPISODES:.1f}%)")
print(f"Avg Reward: {sum(results['total_rewards']) / NUM_EPISODES:.1f}")
print(f"Avg Steps: {sum(results['steps_to_contact']) / NUM_EPISODES:.0f}")
print("=" * 60)