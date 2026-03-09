import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
# from push_learn_force import PandaPushTrajectoryEnv
# from push_learn_force_impedance import PandaPushTrajectoryEnv
# from push_learn_force_impedance_1 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_2 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_3 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_4 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_5 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_6 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_7 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_8 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_9 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_9_increase_impedance import PandaPushTrajectoryEnv
# from push_learn_force_impedance_9_orientation_added import PandaPushTrajectoryEnv
# from push_learn_force_impedance_9_orientation_added_tilt_correction import PandaPushTrajectoryEnv
# from push_learn_force_impedance_10 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_10_correction_both_sides import PandaPushTrajectoryEnv
# from push_learn_force_impedance_11 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_12 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_13 import PandaPushTrajectoryEnv
# from push_learn_force_impedance_13_rotation_correction import PandaPushTrajectoryEnv
# from push_learn_force_impedance_13_rotation_correction_new import PandaPushTrajectoryEnv
# from push_with_finger_learn_force_impedance_14 import PandaPushTrajectoryEnv
# from push_with_finger_learn_force_impedance_14_v01 import PandaPushTrajectoryEnv
from push_with_finger_learn_force_impedance_14_v01_straight import PandaPushTrajectoryEnv

ALGORITHM = "PPO"
# TRAJECTORY_TYPE = "straight"
# TRAJECTORY_TYPE = "curved"
TRAJECTORY_TYPE = "s_curve"
TOTAL_TIMESTEPS = 1_000_000
# MODEL_NAME = f"push_trajectory_{TRAJECTORY_TYPE}_02"
MODEL_NAME = f"push_with_finger_learn_force_impedance_14_v01_straight_{TRAJECTORY_TYPE}_202"
# MODEL_NAME = f"push_learn_force_impedance_9_orientation_added_{TRAJECTORY_TYPE}_01"


# ==================== SETUP ====================
print("="*60)
print("TRAINING: Panda Push Trajectory Following")
print("="*60)
print(f"Algorithm: {ALGORITHM}")
print(f"Trajectory: {TRAJECTORY_TYPE}")
print(f"Timesteps: {TOTAL_TIMESTEPS}")
print("="*60)

env = PandaPushTrajectoryEnv(
    render_mode=None,
    trajectory_type=TRAJECTORY_TYPE
)
print("Environment Created!!!!!")

# Check Environment
try:
    check_env(env)
    print("Environment check passed!")
except Exception as e:
    print(f"Environment check failed: {e}")
    env.close()
    exit()

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
    tensorboard_log="./ppo_push_tensorboard/"
)

# --- CALLBACKS ---
current_script_dir = os.path.dirname(os.path.realpath(__file__))
save_folder = os.path.join(current_script_dir, "saved_models")
os.makedirs(save_folder, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=save_folder,
    name_prefix=MODEL_NAME
)

# --- 3. Train Model ---
print("Starting training......")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    # callback=checkpoint_callback,
    progress_bar=True
)

"""
# This is what happens inside model.learn(total_timesteps=200000)

steps_taken = 0
obs = env.reset() # Start the first game

while steps_taken < 200000:
    
    # 1. The Brain decides (Policy)
    action, _ = model.predict(obs)
    
    # 2. YOUR FUNCTION runs here!
    # This calls the 'step' method you wrote in PandaPushEnv
    new_obs, reward, terminated, truncated, info = env.step(action)
    
    # 3. Model learns from the result
    model.store_transition(obs, action, reward, new_obs)
    
    obs = new_obs
    steps_taken += 1
    
    # 4. If the bottle fell or time ran out (terminated/truncated)
    if terminated or truncated:
        obs = env.reset() # Restart the environment
"""

# --- 4. Save Model ---
model_save_path = os.path.join(save_folder, MODEL_NAME)
model.save(model_save_path)
env.close()

print("\n" + "="*60)
print(f"Training complete!")
print(f"Model saved to: {model_save_path}.zip")
