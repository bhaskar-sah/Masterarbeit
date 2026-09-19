# import os
# import time
# from stable_baselines3 import PPO

# # Import the env
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
# # from push_learn_force_impedance_9_orientation_added import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_9_orientation_added_tilt_correction import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_9_increase_impedance import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_10 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_11 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_12 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_13 import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_13_rotation_correction import PandaPushTrajectoryEnv
# # from push_learn_force_impedance_13_rotation_correction_new import PandaPushTrajectoryEnv
# # from push_with_finger_learn_force_impedance_14 import PandaPushTrajectoryEnv
# # from push_with_finger_learn_force_impedance_14_v01_curriculum_learning import PandaPushTrajectoryEnv
# #################
# # from push_with_finger_learn_force_impedance_14_v01_straight import PandaPushTrajectoryEnv
# from env import PandaPushTrajectoryEnv
# ############
# # from push_with_finger_learn_force_impedance_14_v01_straight_reward_correction import PandaPushTrajectoryEnv
# # from push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning import PandaPushTrajectoryEnv
# # from push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_1 import PandaPushTrajectoryEnv

# # ==================== CONFIGURATION ====================
# # MODEL_NAME = "push_trajectory_impedance_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_1_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_1_straight_02.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_2_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_3_straight_02.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_4_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_8_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_9_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_9_straight_01_.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_9_orientation_added_tilt_correction_straight_03_increased_bottle_size.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_9_increase_impedance_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_10_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_10_correction_both_sides_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_12_straight_04.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_13_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_13_rotation_correction_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_13_rotation_correction_new_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_curved_04_radius_0_2.zip"  # Change to your model name
# # MODEL_NAME = "push_curriculum_learning_03_final.zip"  # Change to your model name
# # MODEL_NAME = "push_trajectory_impedance_9_orientation_added_straight_03.zip"  # Change to your model name
# # MODEL_NAME = "push_learn_force_impedance_9_orientation_added_straight_01.zip"  # Change to your model name
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_straight_18.zip"  # Change to your model name
# #####################################################################################################################
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_straight_18.zip"  # Change to your model name
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_curved_19.zip"  # Change to your model name
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_s_curve_20.zip"  # Change to your model name here 0.08
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_s_curve_21.zip"  # make the amplitude 0.04 for this model
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_s_curve_22.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_s_curve_23.zip"  # make the amplitude 0.04 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_s_curve_24.zip"  # make the amplitude 0.04 for this model with longer distance goal [0.4;-0.4]
# ############################################################################################################
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_straight_25.zip"  # make the amplitude 0.04 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_curved_26.zip"  # make the amplitude 0.04 for this model with longer distance goal [0.4;-0.4]
# #############################################################################################################
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_straight_27.zip"  # make the amplitude 0.04 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_straight_200.zip"  # make the amplitude 0.04 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_curved_200.zip"  # make the amplitude 0.04 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_curved_201.zip"  # make the amplitude 0.04 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_curved_201.zip"  # make the amplitude 0.04 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_s_curve_202.zip"  # make the amplitude 0.04 for this model with smaller distance goal [0.4;-0.2]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_s_curve_203_0_08.zip"  # make the amplitude 0.08 for this model with smaller distance goal [0.4;-0.2]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_s_curve_203_0_ 08_longer_distance.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_s_curve_203_0_08_longer_distance_new.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# #################################################
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_straight.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_straight_1.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_curved_1.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_curved_1_.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2_1.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2_2.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2_3.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2_4.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2_5.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2_6.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2_7.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2_8.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2_10.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2_11.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_s_curve_2_12.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# ###############################################################################################################
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_1_straight_1_straight.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_straight_1_straight.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# ################################################################################
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_straight_2_straight.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_straight_3_straight.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_straight_4_straight.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_straight_6_straight.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# ###############################################################################################################
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_1_straight.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_1_straight_1.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_1_curved.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_1_curved_1.zip"  # make the amplitude 0.08 for this model with longer distance goal [0.4;-0.4]
# # MODEL_NAME = "best_model.zip"
# ###############################################
# # MODEL_NAME = "trained_model_straight.zip"
# # MODEL_NAME = "trained_model_curved.zip"
# # MODEL_NAME = "trained_model_s_curve.zip"
# # MODEL_NAME = "trained_model_straight_r_angle.zip"
# # MODEL_NAME = "trained_model_curved_r_angle_1.zip"

# # MODEL_NAME = "trained_model_s_curve_r_angle_3.zip"
# # MODEL_NAME = "trained_model_straight_only_push_v18.zip"
# MODEL_NAME = "trained_model_s_curve_only_push_v24.zip"

# # TRAJECTORY_TYPE = "straight"  # Should match training or test generalization
# # TRAJECTORY_TYPE = "curved"  # Should match training or test generalization
# TRAJECTORY_TYPE = "s_curve"  # Should match training or test generalization
# NUM_EPISODES = 10
# VISUAL_DELAY = 0.02  # Seconds between frames (0 for max speed)

# # ==================== SETUP ====================
# print("="*60)
# print("TESTING: Panda Push Trajectory Following")
# print("="*60)

# # Create environment with rendering
# env = PandaPushTrajectoryEnv(
#     render_mode="human",
#     trajectory_type=TRAJECTORY_TYPE
# )
# print("Environment created.")

# # ==================== LOAD MODEL ====================
# current_dir = os.path.dirname(os.path.realpath(__file__))
# # model_path = os.path.join(current_dir, "saved_models/best_model_straight_1", MODEL_NAME)
# # model_path = os.path.join(current_dir, "saved_models/best_model_curved", MODEL_NAME)
# model_path = os.path.join(current_dir, "saved_models", MODEL_NAME)

# if not os.path.exists(model_path):
#     print(f"Error: Model file not found at {model_path}")
#     print("Available models in saved_models/:")
#     # models_dir = os.path.join(current_dir, "saved_models/best_model_straight_1")
#     models_dir = os.path.join(current_dir, "saved_models")
#     if os.path.exists(models_dir):
#         for f in os.listdir(models_dir):
#             if f.endswith(".zip"):
#                 print(f"  - {f}")
#     env.close()
#     exit()

# try:
#     model = PPO.load(model_path, env=env)
#     print(f"Model loaded successfully from {model_path}")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()

# print(f"Model loaded from: {model_path}")

# # ==================== RUN TEST EPISODES ====================
# print(f"\nRunning {NUM_EPISODES} test episodes...")
# print(f"Trajectory type: {TRAJECTORY_TYPE}")
# print("-"*60)

# results = {
#     "successes": 0,
#     "total_rewards": [],
#     "final_progress": [],
#     "final_deviation": [],
# }

# for episode in range(NUM_EPISODES):
#     obs, info = env.reset()
#     terminated = False
#     truncated = False
#     episode_reward = 0
#     step_count = 0

#     while not terminated and not truncated:
#         # Get action from trained model
#         action, _states = model.predict(obs, deterministic=True)

#         # Take action
#         obs, reward, terminated, truncated, info = env.step(action)

#         # Render
#         env.render()

#         episode_reward += reward
#         step_count += 1

#         # Optional delay for visualization
#         if VISUAL_DELAY > 0:
#             time.sleep(VISUAL_DELAY)

#     # Record results
#     success = info.get("is_success", False)
#     progress = info.get("progress", 0)
#     deviation = info.get("deviation", 0)

#     results["total_rewards"].append(episode_reward)
#     results["final_progress"].append(progress)
#     results["final_deviation"].append(deviation)
#     if success:
#         results["successes"] += 1

#     status = "SUCCESS" if success else "FAILED"
#     print(f"Episode {episode + 1}/{NUM_EPISODES}: {status} | "
#           f"Progress: {progress:.1%} | "
#           f"Deviation: {deviation:.3f}m | "
#           f"Reward: {episode_reward:.1f} | "
#           f"Steps: {step_count}")

# # ==================== SUMMARY ====================
# env.close()

# print("\n" + "=" * 60)
# print("TEST SUMMARY")
# print("=" * 60)
# print(f"Success Rate: {results['successes']}/{NUM_EPISODES} "
#       f"({100 * results['successes'] / NUM_EPISODES:.1f}%)")
# print(f"Avg Reward: {sum(results['total_rewards']) / NUM_EPISODES:.1f}")
# print(f"Avg Progress: {100 * sum(results['final_progress']) / NUM_EPISODES:.1f}%")
# print(f"Avg Deviation: {sum(results['final_deviation']) / NUM_EPISODES:.4f}m")
# print("=" * 60)

#######################################################################################################

import os
import time
from stable_baselines3 import PPO

# Import the env
from env import PandaPushTrajectoryEnv

# ==================== CONFIGURATION ====================
# Define the trajectory and the exact version folder you want to test
# TRAJECTORY_TYPE = "straight"  # Should match training or test generalization
# TRAJECTORY_TYPE = "curved"  # Should match training or test generalization
TRAJECTORY_TYPE = "s_curve"  
MODEL_DIR = f"trained_model_{TRAJECTORY_TYPE}_only_push_v37_force" # this is working <--- Change to v26, v27, etc. in the future
# MODEL_DIR = f"trained_model_{TRAJECTORY_TYPE}_only_push_v46_force_normal" # <--- Change to v26, v27, etc. in the future
# MODEL_DIR = f"trained_model_{TRAJECTORY_TYPE}_only_push_v48_force_normal"
# MODEL_DIR = f"trained_model_curriculum_only_push_v48_force_normal"
# MODEL_DIR = f"trained_model_{TRAJECTORY_TYPE}_only_push_v55_force_normal_frames"
# MODEL_DIR = f"trained_model_{TRAJECTORY_TYPE}_only_push_v60_force_normal_frames_roll_plus_pitch_para_tune"
# MODEL_DIR = f"trained_model_{TRAJECTORY_TYPE}_only_push_v60_force_normal_motion"
# MODEL_DIR = "trained_model_dr_mass_curriculum_v62"

NUM_EPISODES = 10
VISUAL_DELAY = 0.02  # Seconds between frames (0 for max speed)

# ==================== SETUP ====================
print("="*60)
print("TESTING: Panda Push Trajectory Following")
print("="*60)

# Create environment with rendering
env = PandaPushTrajectoryEnv(
    render_mode="human",
    trajectory_type= TRAJECTORY_TYPE # "curved"
)
print("Environment created.")

# ==================== LOAD MODEL ====================
current_dir = os.path.dirname(os.path.realpath(__file__))

# Point directly to 'best_model.zip' inside the specific version folder
model_path = os.path.join(current_dir, "saved_models_new", MODEL_DIR, "best_model.zip")



# # =========================FOR CURRICULUM LEARNING RUN=============================
# # Point to the final phase of the curriculum
# PHASE_FOLDER = "phase_4_s_curve" # Or whatever the exact folder name is in your directory

# model_path = os.path.join(current_dir, "saved_models_new", MODEL_DIR, PHASE_FOLDER, "best_model.zip")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print(f"Make sure you have trained {MODEL_DIR} and the eval_callback saved best_model.zip!")
    env.close()
    exit()

try:
    model = PPO.load(model_path, env=env, device="cpu")
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    env.close()
    exit()

# ==================== RUN TEST EPISODES ====================
print(f"\nRunning {NUM_EPISODES} test episodes...")
print(f"Trajectory type: {TRAJECTORY_TYPE}")
print("-"*60)

results = {
    "successes": 0,
    "total_rewards": [],
    "final_progress": [],
    "final_deviation": [],
}

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    step_count = 0

    while not terminated and not truncated:
        # Get action from trained model (deterministic=True strips out random training noise)
        action, _states = model.predict(obs, deterministic=True)

        # Take action
        obs, reward, terminated, truncated, info = env.step(action)

        # Render
        env.render()

        episode_reward += reward
        step_count += 1

        # Optional delay for visualization
        if VISUAL_DELAY > 0:
            time.sleep(VISUAL_DELAY)

    # Record results
    success = info.get("is_success", False)
    progress = info.get("progress", 0)
    deviation = info.get("deviation", 0)

    results["total_rewards"].append(episode_reward)
    results["final_progress"].append(progress)
    results["final_deviation"].append(deviation)
    if success:
        results["successes"] += 1

    status = "SUCCESS" if success else "FAILED"
    print(f"Episode {episode + 1}/{NUM_EPISODES}: {status} | "
          f"Progress: {progress:.1%} | "
          f"Deviation: {deviation:.3f}m | "
          f"Reward: {episode_reward:.1f} | "
          f"Steps: {step_count}")

# ==================== SUMMARY ====================
env.close()

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Tested Model: {MODEL_DIR}/best_model.zip")
print(f"Success Rate: {results['successes']}/{NUM_EPISODES} "
      f"({100 * results['successes'] / NUM_EPISODES:.1f}%)")
print(f"Avg Reward: {sum(results['total_rewards']) / NUM_EPISODES:.1f}")
print(f"Avg Progress: {100 * sum(results['final_progress']) / NUM_EPISODES:.1f}%")
print(f"Avg Deviation: {sum(results['final_deviation']) / NUM_EPISODES:.4f}m")
print("=" * 60)