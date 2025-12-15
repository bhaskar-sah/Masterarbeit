import os
import time
from stable_baselines3 import PPO
# from reach_bottle_new import PandaPushEnv  # Import your custom environment
from reach_bottle_and_align import PandaPushEnv  # Import your custom environment

# --- 1. Setup Environment ---
# We MUST use render_mode="human" to see the simulation
env = PandaPushEnv(render_mode="human")
print("Environment created.")

# --- 2. Load Model ---
# --- 2. Load Model ---
# Get the directory where this test script is located
current_dir = os.path.dirname(os.path.realpath(__file__))

# distinct filename matches the one you used in training
model_name = "reach_bottle_and_push_extend_13.zip"

# Construct the full path
model_path = os.path.join(current_dir, "saved_models", model_name)

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Check the filename and ensure the 'saved_models' folder exists.")
    exit()

try:
    model = PPO.load(model_path, env=env)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 3. Run Test Episodes ---
print("Running test episodes...")
num_episodes = 30
visual_deLay = 0.03

for episode in range(num_episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0

    while not terminated and not truncated:
        # Get the action from the trained model (deterministic=True)
        action, _states = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the environment
        # Your env's render() method handles the viewer.sync()
        env.render()

        episode_reward += reward

        # Optional: Add a small delay so you can watch it
        # time.sleep(0.01)
        time.sleep(visual_deLay)

    print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}")

env.close()
print("Testing complete.")