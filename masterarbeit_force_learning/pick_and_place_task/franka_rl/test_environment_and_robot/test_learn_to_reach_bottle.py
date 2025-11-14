import time
from stable_baselines3 import PPO
from learn_to_reach_bottle import PandaPushEnv  # Import your custom environment

# --- 1. Setup Environment ---
# We MUST use render_mode="human" to see the simulation
env = PandaPushEnv(render_mode="human")
print("Environment created.")

# --- 2. Load Model ---
model_path = "panda_reach_model5.zip"
try:
    model = PPO.load(model_path, env=env)
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Did you run train.py first to create the model file?")
    exit()

# --- 3. Run Test Episodes ---
print("Running test episodes...")
num_episodes = 30

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

    print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}")

env.close()
print("Testing complete.")