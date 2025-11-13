"""
Your original testing script - FIXED
This WILL show MuJoCo visualization window!
Only added the missing VecNormalize wrapper
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from franka_push_env import PandaPushEnv

# Your model path
model_path = "ppo_panda_logs/ppo_panda_push_final"
vec_normalize_path = "ppo_panda_logs/vec_normalize.pkl"

# --- 1. Load the Trained Model ---
print(f"Loading model from {model_path}...")
try:
    model = PPO.load(model_path)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Did you run training and save the model to {model_path}?")
    exit()

# --- 2. Create the Environment (with rendering) ---
# render_mode="human" to open the MuJoCo window ✓
env = PandaPushEnv(render_mode="human")  # ← This WILL show visualization!

# ⭐ ADD THESE 4 LINES (The critical fix!) ⭐
env = DummyVecEnv([lambda: env])  # Wrap in vectorized env
env = VecNormalize.load(vec_normalize_path, env)  # Load normalization stats
env.training = False  # Don't update stats during testing
env.norm_reward = False  # Don't normalize rewards

# --- 3. Run Test Episodes ---
print("Testing the trained model... Press Ctrl+C to stop.")
print("MuJoCo visualization window will open!\n")

num_episodes = 10000  # You can change this

for episode in range(num_episodes):
    obs = env.reset()  # Note: no unpacking needed with VecEnv
    done = False
    episode_reward = 0
    steps = 0

    print(f"--- Starting Episode {episode + 1} ---")

    while not done:
        # Get action from the trained model
        action, _states = model.predict(obs, deterministic=True)

        # Take the step
        obs, reward, done, info = env.step(action)

        # Extract values (VecEnv returns arrays)
        episode_reward += reward[0]
        done = done[0]
        steps += 1

        # Render - shows MuJoCo window ✓
        env.render()

    # Episode finished - print results
    success = info[0].get('success', False)
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"{status} - Steps: {steps}, Reward: {episode_reward:.2f}")

env.close()
print("\nTesting finished.")