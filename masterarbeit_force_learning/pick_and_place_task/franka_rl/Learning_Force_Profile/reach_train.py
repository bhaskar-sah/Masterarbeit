"""
Training Script for Reach Phase.

Trains a PPO agent to reach from home position to stable contact with bottle.
This is trajectory-agnostic: the agent learns to approach and contact the
bottle from the home position regardless of what trajectory will be pushed.

Usage:
    python reach_train.py
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from reach_config import ReachConfig
from reach_env import PandaReachEnv


# ==================== CONFIGURATION ====================
TOTAL_TIMESTEPS = 200_000
MODEL_NAME = "reach_model_31"

# ==================== SETUP ====================
print("=" * 60)
print("TRAINING: Panda Reach to Bottle Contact")
print("=" * 60)
print(f"Algorithm: PPO")
print(f"Timesteps: {TOTAL_TIMESTEPS}")
print(f"Goal: Reach bottle and establish stable contact")
print("=" * 60)

config = ReachConfig()
env = PandaReachEnv(render_mode=None, config=config)
print("Training environment created.")

try:
    check_env(env)
    print("Environment check passed!")
except Exception as e:
    print(f"Environment check failed: {e}")
    env.close()
    exit()

# ==================== CREATE MODEL ====================
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_reach_tensorboard/",
    device="cpu"
)

# ==================== CALLBACKS ====================
current_script_dir = os.path.dirname(os.path.realpath(__file__))
save_folder = os.path.join(current_script_dir, "saved_models")
os.makedirs(save_folder, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=save_folder,
    name_prefix=MODEL_NAME
)

# ==================== TRAIN ====================
print("\nStarting training...")
print("Monitor with: tensorboard --logdir ./ppo_reach_tensorboard/")
print("-" * 60)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    progress_bar=True
)

# ==================== SAVE ====================
model_save_path = os.path.join(save_folder, MODEL_NAME)
model.save(model_save_path)
env.close()

print("\n" + "=" * 60)
print(f"Training complete!")
print(f"Model saved to: {model_save_path}.zip")
print("=" * 60)