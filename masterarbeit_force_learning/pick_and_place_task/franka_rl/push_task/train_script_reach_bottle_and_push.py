import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from reach_bottle_and_push import PandaPushEnv  # Import your custom environment

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
# step size: 100,000
model.learn(total_timesteps=100000)

# --- 4. Save Model ---
print("Saving Trained Model...")
current_script_dir = os.path.dirname(os.path.realpath(__file__))
save_folder = os.path.join(current_script_dir, "saved_models")
os.makedirs(save_folder, exist_ok=True)

model_name = "reach_bottle_and_push_01"
model_save_path = os.path.join(save_folder, model_name)

model.save(model_save_path)

env.close()
print(f"Training complete! Model saved to: {model_save_path}.zip")