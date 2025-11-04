import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

print("\n--- 5. Loading and Evaluating Trained Model ---")

# Load the saved model
# We add device="cpu" to silence the GPU warning
model = PPO.load("ppo_hopper_gpu_ipynb", device="cpu")

# Create a new environment for testing, using v5
# This time, we don't set render_mode="human" for a fast, clean evaluation
test_env = gym.make("Hopper-v5")
# Wrap the environment with Monitor to silence the evaluation warning
test_env = Monitor(test_env)

# This runs 10 full "episodes" and collects the rewards
mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)

print(f"\n--- Done ---")
print(f"Mean reward over 10 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

test_env.close()

# If you want to SEE it run, you can do this *after* evaluation:
print("\n--- 6. Running visual test ---")
vis_env = gym.make("Hopper-v5", render_mode="human")
obs, info = vis_env.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = vis_env.step(action)
    if terminated or truncated:
        obs, info = vis_env.reset()
vis_env.close()