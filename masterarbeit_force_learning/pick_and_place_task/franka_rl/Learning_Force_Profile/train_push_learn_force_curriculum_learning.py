import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from push_with_finger_learn_force_impedance_14_v01_curriculum_learning import PandaPushTrajectoryEnv

ALGORITHM = "PPO"
TRAJECTORY_TYPE = "curriculum"
TOTAL_TIMESTEPS = 2_000_000
MODEL_NAME = f"push_curriculum_learning_02"


# ==================== CURRICULUM CALLBACK ====================
class CurriculumCallback(BaseCallback):
    """Callback to print curriculum progress during training."""

    def __init__(self, print_freq=50000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self):
        if self.n_calls % self.print_freq == 0:
            env = self.training_env.envs[0]
            print(f"\n[Timestep {self.n_calls}] {env.get_curriculum_stage()}")
            env.print_statistics()
        return True

# ==================== SETUP ====================
print("=" * 60)
print("TRAINING: Panda Push - CURRICULUM LEARNING")
print("=" * 60)
print(f"Algorithm: {ALGORITHM}")
print(f"Trajectory Mode: {TRAJECTORY_TYPE}")
print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
print("=" * 60)
print("\nCurriculum Schedule:")
print("  Episodes 0-999:     straight only")
print("  Episodes 1000-2999: straight + curved")
print("  Episodes 3000+:     straight + curved + s_curve")
print("=" * 60)

# Create environment with curriculum learning
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
    device="cpu",  # Use CPU for MlpPolicy (faster than GPU), mentioned in official website
    tensorboard_log="./ppo_push_tensorboard/"
)

# --- CALLBACKS ---
current_script_dir = os.path.dirname(os.path.realpath(__file__))
save_folder = os.path.join(current_script_dir, "saved_models")
os.makedirs(save_folder, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=save_folder,
    name_prefix=MODEL_NAME
)

curriculum_callback = CurriculumCallback(print_freq=50000)

# --- 3. Train Model ---
print("Starting training......")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    # callback=[checkpoint_callback, curriculum_callback],  # Both callbacks
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
model_save_path = os.path.join(save_folder, MODEL_NAME + "_final")
model.save(model_save_path)

# print final statistics
print("\n" + "=" * 60)
print(f"Training complete!")
print("=" * 60)
env.print_statistics()
print(f"\nModel saved to: {model_save_path}.zip")

env.close()