"""
Minimal working training script
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from franka_push_env import PandaPushEnv

# Create log directory
log_dir = "ppo_panda_logs/"
os.makedirs(log_dir, exist_ok=True)

# Create environment WITHOUT rendering (important!)
print("Creating environment...")
env = PandaPushEnv(render_mode="human")  # None, not "human"!

# Wrap in DummyVecEnv
env = DummyVecEnv([lambda: env])

# Add normalization (CRITICAL!)
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0
)

# Create PPO model with device='cpu' to avoid GPU warning
print("Creating PPO model...")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1,
    device='cpu',  # Use CPU (PPO is faster on CPU anyway)
    tensorboard_log=log_dir
)

# Train
print("Starting training...")
print("This will take 1-2 hours for 500k timesteps")
model.learn(total_timesteps=50_000, progress_bar=True)

# Save model
model_path = os.path.join(log_dir, "ppo_panda_push_final")
print(f"\nSaving model to {model_path}...")
model.save(model_path)
env.save(os.path.join(log_dir, "vec_normalize.pkl"))

print("Training complete!")
env.close()