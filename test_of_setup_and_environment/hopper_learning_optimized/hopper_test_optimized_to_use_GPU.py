import gymnasium as gym
from stable_baselines3 import PPO

# --- 1. Load the saved model ---
print("Loading model...")
model = PPO.load("ppo_hopper_gpu")

# --- 2. Create a *single* environment with rendering ---
print("Creating environment to watch...")
env = gym.make("Hopper-v4", render_mode="human")

# --- 3. Run the agent ---
obs, _ = env.reset()
for _ in range(2000):
    # Get the agent's action
    action, _ = model.predict(obs, deterministic=True)

    # Perform the action in the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Reset if the episode ends
    if terminated or truncated:
        print("Episode finished. Resetting.")
        obs, _ = env.reset()

env.close()