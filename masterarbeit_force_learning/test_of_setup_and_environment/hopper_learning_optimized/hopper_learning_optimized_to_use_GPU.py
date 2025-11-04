import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# --- 1. Create Parallel Environments ---
# This is the most important change.
# We create 16 "Hopper" environments to run in parallel on the CPU.
# We remove render_mode="human" to stop the slow rendering.
print("--- 1. Creating Vectorized Environment ---")
env = make_vec_env("Hopper-v4", n_envs=16)
# env = make_vec_env("Hopper-v4", render_mode="human")

# --- 2. Create PPO Agent on the GPU ---
# We add device="cuda" to tell the model to use the GPU.
# We increase the batch_size to give the GPU more work to do at once,
# which is more efficient.
print("--- 2. Creating PPO Agent on GPU ---")
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda",      # <-- Tell it to use the GPU
    batch_size=512,     # <-- Give the GPU a bigger chunk of work
    n_steps=4096        # <-- Collect more data before each update
)

# --- 3. Training Agent ---
# This will now be much faster and you'll see higher GPU-Util
print("--- 3. Training Agent ---")
model.learn(total_timesteps=10000000) # Increased steps, as it's faster now

print("--- 4. Saving Trained Model ---")
model.save("ppo_hopper_gpu")

# Clean up the environments
env.close()