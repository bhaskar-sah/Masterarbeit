import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from learn_to_reach_bottle import PandaPushEnv  # Import your custom environment

# --- 1. Setup Environment ---
# Create the environment
env = PandaPushEnv(render_mode=None) # No rendering during training for speed
print("Environment created.")

# Optional: Check if the environment follows the Gymnasium API
# This is a good check to run at least once
try:
    check_env(env)
    print("Environment check passed!")
except Exception as e:
    print(f"Environment check failed: {e}")
    exit()

# --- 2. Define Model ---
# We use the MlpPolicy because our observations are vectors (not images)
# verbose=1 will print the training progress
model = PPO("MlpPolicy", env, verbose=1)

# --- 3. Train Model ---
print("Starting training...")
# 100,000 steps is a good start. For a harder task, you might need 1,000,000+.
# This will take a few minutes.
model.learn(total_timesteps=2000000)

# --- 4. Save Model ---
# The model will be saved as "panda_reach_model.zip"
model_save_path = "panda_reach_model6_with_1_2m_dist_4_with_0_85"
model.save(model_save_path)

env.close()
print(f"Training complete! Model saved to {model_save_path}.zip")