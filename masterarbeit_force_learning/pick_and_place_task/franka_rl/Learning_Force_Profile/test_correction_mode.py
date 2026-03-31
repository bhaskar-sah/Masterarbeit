# """
# Test correction mode without trained model.
# Uses random actions to see if the correction mode activates and works correctly.
# """
#
# import numpy as np
# # from push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning import PandaPushTrajectoryEnv
# from push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_1  import PandaPushTrajectoryEnv
#
# # Create environment
# env = PandaPushTrajectoryEnv(render_mode="human", trajectory_type="straight")
# # env = PandaPushTrajectoryEnv(render_mode="human", trajectory_type="curved")
#
# # Run episodes
# for episode in range(10):
#     obs, _ = env.reset()
#     done = False
#     step = 0
#
#     while not done and step < 500:
#         # Random action (or zero action to see baseline behavior)
#         # action = env.action_space.sample()  # Random
#         action = np.array([0.0, -0.5, 0.3, 0.3], dtype=np.float32)  # Neutral action: no speed mod, no wrist mod, medium K
#
#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         step += 1
#
#         # Render
#         env.render()
#
#     print(
#         f"\nEpisode {episode + 1} finished: {step} steps, progress={info.get('progress', 0):.1%}, deviation={info.get('deviation', 0) * 100:.1f}cm")
#
# env.close()
# print("\nTest complete!")


###########################################################################################
"""
Test script for the more RL-based Panda pushing environment.

Purpose:
    - sanity-check the new action meaning
    - verify that lateral_mod actually changes behavior
    - test fixed actions before RL training
"""

import numpy as np
from pick_and_place_task.franka_rl.Learning_Force_Profile.other_recent_files.push_with_finger_learn_force_impedance_14_v01_straight_reward_correction_tuning_1 import PandaPushTrajectoryEnv
# Replace the import above with your actual Python filename


def run_episode(env, action, max_steps=500, render=True, label=""):
    obs, _ = env.reset()
    done = False
    step = 0
    info = {}

    print("\n" + "=" * 70)
    print(f"RUN: {label}")
    print(f"FIXED ACTION = {action}")
    print("=" * 70)

    while not done and step < max_steps:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if render:
            env.render()

    print(
        f"Finished: steps={step}, "
        f"progress={info.get('progress', 0.0):.1%}, "
        f"deviation={info.get('deviation', 0.0) * 100:.1f}cm, "
        f"tilt={info.get('bottle_tilt', 0.0):.4f}, "
        f"force={info.get('force_magnitude', 0.0):.2f}N"
    )

    return info


def main():
    env = PandaPushTrajectoryEnv(render_mode="human", trajectory_type="straight")
    # For non-visual debugging, use:
    # env = PandaPushTrajectoryEnv(render_mode=None, trajectory_type="straight")

    test_actions = [
        ("neutral", np.array([0.0, 0.0, 0.3, 0.3], dtype=np.float32)),
        ("right_correction", np.array([0.0, 0.5, 0.3, 0.3], dtype=np.float32)),
        ("left_correction", np.array([0.0, -0.5, 0.3, 0.3], dtype=np.float32)),
        ("faster_forward", np.array([0.5, 0.0, 0.3, 0.3], dtype=np.float32)),
    ]

    results = []

    for label, action in test_actions:
        info = run_episode(env, action, max_steps=500, render=True, label=label)
        results.append((label, info))

    env.close()

    print("\n" + "#" * 70)
    print("SUMMARY")
    print("#" * 70)
    for label, info in results:
        print(
            f"{label:16s} | "
            f"progress={info.get('progress', 0.0):.1%} | "
            f"deviation={info.get('deviation', 0.0) * 100:.1f}cm | "
            f"tilt={info.get('bottle_tilt', 0.0):.4f} | "
            f"force={info.get('force_magnitude', 0.0):.2f}N"
        )


if __name__ == "__main__":
    main()