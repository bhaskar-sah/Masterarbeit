import os
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
# Import your custom environment class
from franka_push_env_old import PandaPushEnv

model_path = "trained_models/ppo_panda_push_model1"
# model_path = "trained_models/ppo_panda_push_model"

# --- 1. Load the Trained Model ---
print(f"Loading model from {model_path}...")
try:
    model = PPO.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Did you run 'train.py' and save the model to {model_path}?")
    exit()

# --- 2. Create the Environment (with rendering) ---
# render_mode="human" to open the MuJoCo window
env = PandaPushEnv(render_mode="human")

# --- 3. Run Test Episodes ---
print("Testing the trained model... Press Ctrl+C to stop.")
num_episodes = 100

for episode in range(num_episodes):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0

    print(f"--- Starting Episode {episode + 1} ---")

    while not terminated and not truncated:
        # Get action from the trained model (deterministic=True means
        # it will always pick the "best" action, not explore)
        action, _states = model.predict(obs, deterministic=True)

        # Take the step
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        episode_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

env.close()
print("Testing finished.")