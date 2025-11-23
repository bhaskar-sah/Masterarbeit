# ================================================
#     LOAD AND TEST THE TRAINED MODEL
# ================================================

print("\n--- 5. Loading and Testing Trained Model ---")

# Delete the old model and environment
del model
del env

# Load the saved model
model = PPO.load("ppo_hopper")

# Create a new environment for testing
test_env = gym.make("Hopper-v4", render_mode="human")
obs, info = test_env.reset()

# Run the trained model for 2000 steps
for _ in range(2000):
    # 'deterministic=True' makes the agent's actions predictable (no random exploration)
    action, _ = model.predict(obs, deterministic=True)
    
    # Take the action in the environment
    obs, reward, terminated, truncated, info = test_env.step(action)
    
    # If the episode is over (e.g., the hopper fell), reset it
    if terminated or truncated:
        print("Episode finished. Resetting.")
        obs, info = test_env.reset()

test_env.close()
print("--- Done ---")