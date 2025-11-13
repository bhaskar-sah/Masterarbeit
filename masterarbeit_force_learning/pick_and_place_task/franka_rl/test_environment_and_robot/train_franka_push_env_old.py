import gymnasium as gym
from stable_baselines3 import PPO
import os
from franka_push_env_old import PandaPushEnv

log_dir = "test1/ppo_panda_logs_old/"
os.makedirs(log_dir, exist_ok=True)

model_path = os.path.join(log_dir, "ppo_panda_push_model3.zip")

env = PandaPushEnv(render_mode="human")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

print("Starting training...")
model.learn(total_timesteps=50_000)

print("Training is done! Now saving the model...")
model.save(model_path)

print(f"Training fully finished and the model saved to {model_path}")
env.close()