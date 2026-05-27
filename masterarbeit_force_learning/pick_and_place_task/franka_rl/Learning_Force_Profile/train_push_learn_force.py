# import os
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# # from push_learn_force import PandaPushTrajectoryEnv
# # from push_learn_force_impedance import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_1 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_2 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_3 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_4 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_5 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_6 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_7 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_8 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_9 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_9_increase_impedance import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_9_orientation_added import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_9_orientation_added_tilt_correction import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_10 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_10_correction_both_sides import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_11 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_12 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_13 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_13_rotation_correction import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_13_rotation_correction_new import PandaPushTrajectoryEnv
# # from push_with_finger_learn_force_impedance_14 import PandaPushTrajectoryEnv
# # from push_with_finger_learn_force_impedance_14_v01 import PandaPushTrajectoryEnv
# # from push_with_finger_learn_force_impedance_14_v01_straight import PandaPushTrajectoryEnv
# # from push_with_finger_learn_force_impedance_14_v01_straight_reward_correction import PandaPushTrajectoryEnv
# # from push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning import PandaPushTrajectoryEnv
# from env import PandaPushTrajectoryEnv

# ALGORITHM = "PPO"
# # TRAJECTORY_TYPE = "straight"
# # TRAJECTORY_TYPE = "curved"
# TRAJECTORY_TYPE = "s_curve"
# TOTAL_TIMESTEPS = 1_000_000
# # MODEL_NAME = f"push_trajectory_{TRAJECTORY_TYPE}_02"
# MODEL_NAME = f"trained_model_{TRAJECTORY_TYPE}_only_push_v24"
# # MODEL_NAME = f"push_learn_force_impedance_9_orientation_added_{TRAJECTORY_TYPE}_01"


# # ==================== SETUP ====================
# print("="*60)
# print("TRAINING: Panda Push Trajectory Following")
# print("="*60)
# print(f"Algorithm: {ALGORITHM}")
# print(f"Trajectory: {TRAJECTORY_TYPE}")
# print(f"Timesteps: {TOTAL_TIMESTEPS}")
# print(f"Goal: (0.20, -0.40) - down left diagonal")
# print("="*60)

# # Create training environment
# env = PandaPushTrajectoryEnv(
#     render_mode="human",
#     # render_mode=None,
#     trajectory_type=TRAJECTORY_TYPE
# )
# print("Environment Created!!!!!")

# # Check Environment
# try:
#     check_env(env)
#     print("Environment check passed!")
# except Exception as e:
#     print(f"Environment check failed: {e}")
#     env.close()
#     exit()

# # Wrap in VecEnv for potential normalization (optional but recommended)
# # env = DummyVecEnv([lambda: env])
# # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# # Create evaluation environment (separate from training)
# eval_env = PandaPushTrajectoryEnv(
#     render_mode="human",
#     # render_mode=None,
#     trajectory_type=TRAJECTORY_TYPE
# )
# eval_env = Monitor(eval_env)

# model = PPO(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     learning_rate=3e-4,
#     n_steps=2048,
#     batch_size=64,
#     n_epochs=10,
#     gamma=0.99,
#     gae_lambda=0.95,
#     clip_range=0.2,
#     ent_coef=0.01,  # Exploration bonus
#     vf_coef=0.5,
#     max_grad_norm=0.5,
#     tensorboard_log="./ppo_push_tensorboard/",
#     device="auto"  # Uses GPU if available
# )

# # --- CALLBACKS ---
# current_script_dir = os.path.dirname(os.path.realpath(__file__))
# save_folder = os.path.join(current_script_dir, "saved_models")
# os.makedirs(save_folder, exist_ok=True)

# checkpoint_callback = CheckpointCallback(
#     save_freq=50000,
#     save_path=save_folder,
#     name_prefix=MODEL_NAME
# )

# # Evaluate model every 20k steps
# eval_callback = EvalCallback(
#     eval_env,
#     # best_model_save_path=os.path.join(save_folder, "best_model"),
#     best_model_save_path=os.path.join(save_folder, f"trained_model_{TRAJECTORY_TYPE}_only_push_v24"),
#     # log_path=os.path.join(save_folder, "eval_logs"),
#     log_path=os.path.join(save_folder, f"eval_logs_trained_model_{TRAJECTORY_TYPE}_only_push_v24"),
#     eval_freq=20000,
#     n_eval_episodes=5,
#     deterministic=True,
#     render=False
# )

# # --- 3. Train Model ---
# print("\nStarting training...")
# print("Monitor with: tensorboard --logdir ./ppo_push_tensorboard/")
# print("-"*60)

# model.learn(
#     total_timesteps=TOTAL_TIMESTEPS,
#     # callback=[checkpoint_callback, eval_callback],
#     callback=[checkpoint_callback],

#     progress_bar=True
# )

# """
# # This is what happens inside model.learn(total_timesteps=200000)

# steps_taken = 0
# obs = env.reset() # Start the first game

# while steps_taken < 200000:
    
#     # 1. The Brain decides (Policy)
#     action, _ = model.predict(obs)
    
#     # 2. YOUR FUNCTION runs here!
#     # This calls the 'step' method you wrote in PandaPushEnv
#     new_obs, reward, terminated, truncated, info = env.step(action)
    
#     # 3. Model learns from the result
#     model.store_transition(obs, action, reward, new_obs)
    
#     obs = new_obs
#     steps_taken += 1
    
#     # 4. If the bottle fell or time ran out (terminated/truncated)
#     if terminated or truncated:
#         obs = env.reset() # Restart the environment
# """

# # --- 4. Save Model ---
# model_save_path = os.path.join(save_folder, MODEL_NAME)
# model.save(model_save_path)
# env.close()
# eval_env.close()

# print("\n" + "="*60)
# print(f"Training complete!")
# print(f"Model saved to: {model_save_path}.zip")
# print(f"Best model saved to: {save_folder}/best_trained_model_{TRAJECTORY_TYPE}_only_push_v24/")
# print("="*60)


####################################################################################################

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env import PandaPushTrajectoryEnv

ALGORITHM = "PPO"
# TRAJECTORY_TYPE = "straight"
# TRAJECTORY_TYPE = "curved"
TRAJECTORY_TYPE = "s_curve"
TOTAL_TIMESTEPS = 2_000_000
MODEL_NAME = f"trained_model_{TRAJECTORY_TYPE}_only_push_v36_force"

# ==================== FOLDER SETUP ====================
current_script_dir = os.path.dirname(os.path.realpath(__file__))
# This creates: masterarbeit/.../saved_models_new/trained_model_s_curve_only_push_v25/
save_folder = os.path.join(current_script_dir, "saved_models_new", MODEL_NAME)
os.makedirs(save_folder, exist_ok=True)

# ==================== SETUP ====================
print("="*60)
print("TRAINING: Panda Push Trajectory Following")
print("="*60)
print(f"Algorithm: {ALGORITHM}")
print(f"Trajectory: {TRAJECTORY_TYPE}")
print(f"Timesteps: {TOTAL_TIMESTEPS}")
print(f"Goal: (0.20, -0.40) - down left diagonal")
print(f"Save Directory: {save_folder}")
print("="*60)

# Create training environment
env = PandaPushTrajectoryEnv(
    render_mode= "human", # was None,
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

# Wrap in VecEnv for potential normalization (optional but recommended)
# env = DummyVecEnv([lambda: env])
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Create evaluation environment (separate from training)
eval_env = PandaPushTrajectoryEnv(
    render_mode=None,
    trajectory_type=TRAJECTORY_TYPE
)
eval_env = Monitor(eval_env)

# ==================== MODEL DEFINITION ====================
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
    ent_coef=0.01,  # Exploration bonus
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=os.path.join(save_folder, "tensorboard_logs"), # Nested inside v25
    device="auto"  # Uses GPU if available
)

# ==================== CALLBACKS ====================
# Save periodic backups directly in the v25 folder
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=save_folder,
    name_prefix="checkpoint"
)

# Evaluate model every 20k steps and save best_model.zip in the v25 folder
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=save_folder,
    log_path=save_folder,
    eval_freq=20000,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

# ==================== TRAIN ====================
print("\nStarting training...")
print(f"Monitor with: tensorboard --logdir {os.path.join(save_folder, 'tensorboard_logs')}")
print("-"*60)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback], # Both are active!
    progress_bar=True
)

# ==================== SAVE FINAL MODEL ====================
model_save_path = os.path.join(save_folder, "final_model")
model.save(model_save_path)

env.close()
eval_env.close()

print("\n" + "="*60)
print(f"Training complete!")
print(f"All model files, checkpoints, and logs are saved inside:")
print(f"{save_folder}/")
print("="*60)