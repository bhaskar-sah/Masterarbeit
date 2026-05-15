import os
import time
import imageio  # <--- NEW: Required for programmatic video saving
import numpy as np
from stable_baselines3 import PPO

# Import the env
from env import PandaPushTrajectoryEnv

# ==================== CONFIGURATION ====================
TRAJECTORY_TYPE = "s_curve"  
MODEL_DIR = f"trained_model_{TRAJECTORY_TYPE}_only_push_v25" 

NUM_EPISODES = 1
# VISUAL_DELAY is no longer needed since rendering is handled off-screen

# ==================== SETUP ====================
print("="*60)
print("TESTING: Panda Push Trajectory Following (Video Export Mode)")
print("="*60)

# Create environment with off-screen rendering
env = PandaPushTrajectoryEnv(
    render_mode="rgb_array",  # <--- CHANGED: "rgb_array" allows frame extraction
    trajectory_type=TRAJECTORY_TYPE
)
print("Environment created.")

# ==================== LOAD MODEL ====================
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "saved_models_new", MODEL_DIR, "best_model.zip")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    env.close()
    exit()

try:
    model = PPO.load(model_path, env=env)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    env.close()
    exit()

# Directory to save evaluation videos
video_dir = os.path.join(current_dir, "eval_videos", MODEL_DIR)
os.makedirs(video_dir, exist_ok=True)
print(f"Videos will be saved to: {video_dir}")

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
    
    episode_frames = []  # <--- NEW: Store RGB matrices for this episode

    while not terminated and not truncated:
        # Get action from trained model
        action, _states = model.predict(obs, deterministic=True)

        # Take action
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract the frame buffer matrix
        frame = env.render()

        # ADD THIS DIAGNOSTIC LINE TEMPORARILY:
        # === ADD THIS DIAGNOSTIC LINE ===
        if step_count == 10:
            print(f"\n[DIAGNOSTIC] Frame type: {type(frame)}")
            if frame is not None:
                print(f"[DIAGNOSTIC] Frame shape: {np.shape(frame)}\n")
            else:
                print("[DIAGNOSTIC] WARNING: Frame is None! Your env.py is not returning image arrays.\n")
        
        # DOWN-SAMPLING LOGIC:
        # Physics dt = 0.002s (500 steps = 1 second of physics).
        # To make a standard 30 FPS video, we need 1 frame every 0.0333 seconds of physics.
        # 0.0333 / 0.002 = ~16.66. We capture a frame every 16 steps.
        if step_count % 16 == 0 and frame is not None:
            episode_frames.append(frame)

        episode_reward += reward
        step_count += 1

    # === SAVE VIDEO FOR THE EPISODE ===
    if len(episode_frames) > 0:
        video_filename = os.path.join(video_dir, f"episode_{episode + 1}.mp4")
        # Save at 30 FPS so that simulated time translates perfectly to video time
        imageio.mimsave(video_filename, episode_frames, fps=30)
    
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
          f"Steps: {step_count} | Video Saved!")

# ==================== SUMMARY ====================
# Force explicit cleanup of the background renderer before closing the env
if hasattr(env, '_mujoco_offscreen_renderer'):
    try:
        env._mujoco_offscreen_renderer.close()
    except Exception:
        pass
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
