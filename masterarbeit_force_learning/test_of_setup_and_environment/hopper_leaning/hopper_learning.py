import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

print("--- 1. Creating Environment ---")
# Create the MuJoCo "Hopper" environment
# render_mode="human" will open a window to show the simulation
env = gym.make("Hopper-v4", render_mode="human")

# Note: PPO works best with vectorized environments (multiple envs in parallel)
# but we'll use a single env here so you can see the window.
# For serious training, you'd use:
# env = make_vec_env("Hopper-v4", n_envs=4)

print("--- 2. Creating PPO Agent ---")
# Create the PPO agent
# "MlpPolicy" means it will use a simple neural network (Multi-Layer Perceptron)
model = PPO("MlpPolicy", env, verbose=1)

print("--- 3. Training Agent ---")
# Train the agent for 500,000 timesteps
# This will take a few minutes.
# You'll see the Hopper fall over a lot as it learns.
model.learn(total_timesteps=100000)

print("--- 4. Saving Trained Model ---")
# Save the trained model to a file
model.save("ppo_hopper")

# Clean up the environment
env.close()

