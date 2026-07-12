# """
# Test Trained Reach Model.

# Loads a trained reach model and runs test episodes.
# The reach phase is trajectory-agnostic — no trajectory type needed.

# Usage:
#     python reach_test.py
# """

# import os
# import time
# from stable_baselines3 import PPO
# from masterarbeit_force_learning.pick_and_place_task.franka_rl.Learning_Force_Profile.reach_phase.reach_env import PandaReachEnv


# # ==================== CONFIGURATION ====================
# MODEL_NAME = "reach_model_31.zip"
# NUM_EPISODES = 50
# VISUAL_DELAY = 0.1  # Seconds between frames (0 for max speed)

# # ==================== SETUP ====================
# print("=" * 60)
# print("TESTING: Panda Reach to Bottle")
# print("=" * 60)

# env = PandaReachEnv(render_mode="human")
# print("Environment created.")

# # ==================== LOAD MODEL ====================
# current_dir = os.path.dirname(os.path.realpath(__file__))
# model_path = os.path.join(current_dir, "saved_models", MODEL_NAME)

# if not os.path.exists(model_path):
#     print(f"Error: Model file not found at {model_path}")
#     save_dir = os.path.join(current_dir, "saved_models")
#     if os.path.exists(save_dir):
#         print("Available models:")
#         for f in sorted(os.listdir(save_dir)):
#             if f.endswith(".zip"):
#                 print(f"  - {f}")
#     env.close()
#     exit()

# try:
#     model = PPO.load(model_path, env=env)
#     print(f"Model loaded from {model_path}")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     env.close()
#     exit()

# # ==================== RUN TEST EPISODES ====================
# print(f"\nRunning {NUM_EPISODES} test episodes...")
# print("-" * 60)

# results = {
#     "successes": 0,
#     "total_rewards": [],
#     "steps_to_contact": [],
# }

# for episode in range(NUM_EPISODES):
#     obs, info = env.reset()
#     terminated = False
#     truncated = False
#     episode_reward = 0
#     step_count = 0

#     while not terminated and not truncated:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#         env.render()
#         episode_reward += reward
#         step_count += 1

#         if VISUAL_DELAY > 0:
#             time.sleep(VISUAL_DELAY)

#     success = info.get("reach_success", False)
#     results["total_rewards"].append(episode_reward)
#     results["steps_to_contact"].append(step_count)
#     if success:
#         results["successes"] += 1

#     status = "SUCCESS" if success else "FAILED"
#     fail_reason = ""
#     if not success:
#         if info.get("bottle_knocked"):
#             fail_reason = " (bottle knocked)"
#         else:
#             fail_reason = " (timeout)"

#     print(f"Episode {episode + 1}/{NUM_EPISODES}: {status}{fail_reason} | "
#           f"Steps: {step_count} | Reward: {episode_reward:.1f}")

# # ==================== SUMMARY ====================
# env.close()

# print("\n" + "=" * 60)
# print("TEST SUMMARY")
# print("=" * 60)
# print(f"Success Rate: {results['successes']}/{NUM_EPISODES} "
#       f"({100 * results['successes'] / NUM_EPISODES:.1f}%)")
# print(f"Avg Reward: {sum(results['total_rewards']) / NUM_EPISODES:.1f}")
# print(f"Avg Steps: {sum(results['steps_to_contact']) / NUM_EPISODES:.0f}")
# print("=" * 60)

# ########################################################################################################

"""
reach_test.py
(located at Learning_Force_Profile/reach_phase/)

Load a trained reach-phase model and run N test episodes with
visualisation.  The reach phase is trajectory-agnostic, so no
trajectory type needs to be passed.

By default this script finds the most recent training run under
reach_phase/training/ and loads best_model.zip from that folder.
Pass --run_dir, --model_type, or --checkpoint_step to override.

Usage from the reach_phase folder.

  # default: latest run, best model, 50 episodes, visualised
  python reach_test.py

  # explicit run folder
  python reach_test.py --run_dir training/2026-06-16_01-30-21

  # load the final model instead of the best
  python reach_test.py --model_type final

  # load a specific checkpoint
  python reach_test.py --model_type checkpoint --checkpoint_step 150000

  # fewer episodes for a quick check
  python reach_test.py --num_episodes 10

  # headless (no GUI), faster, suitable for scripted evaluation
  python reach_test.py --no_render

  # also write a per-episode CSV to the run folder for later analysis
  python reach_test.py --save_results
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from stable_baselines3 import PPO


# ---------------------------------------------------------------------
# Path setup so the local imports resolve regardless of which directory
# Python is launched from.
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
_PARENT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from reach_env import PandaReachEnv  # noqa: E402


# =====================================================================
def find_latest_run():
    """Find the most recent timestamp folder under training/."""
    training_dir = SCRIPT_DIR / "training"
    if not training_dir.exists():
        raise SystemExit(
            f"ERROR: no training folder at {training_dir}.\n"
            "Run reach_train.py first or pass --run_dir explicitly."
        )
    candidates = [
        d for d in training_dir.iterdir()
        if d.is_dir() and d.name[:4].isdigit()
    ]
    if not candidates:
        raise SystemExit(
            f"ERROR: no timestamp folders under {training_dir}.\n"
            "Run reach_train.py first or pass --run_dir explicitly."
        )
    return sorted(candidates, key=lambda d: d.name, reverse=True)[0]


def resolve_model_path(run_dir, model_type, checkpoint_step):
    """Return the path to the requested model file."""
    if model_type == "best":
        return run_dir / "best_model.zip"
    if model_type == "final":
        return run_dir / "final_model.zip"
    if model_type == "checkpoint":
        return run_dir / f"checkpoint_{checkpoint_step}_steps.zip"
    raise ValueError(f"unknown model_type: {model_type}")


def list_available_models(run_dir):
    """List all .zip files in the run folder, useful for error messages."""
    return sorted(p.name for p in run_dir.iterdir() if p.suffix == ".zip")


# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run N test episodes with a trained reach-phase policy."
        ),
    )
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help=(
            "Path to a specific training run folder under "
            "reach_phase/training/.  If omitted, the most recent "
            "folder is used automatically."
        ),
    )
    parser.add_argument(
        "--model_type", type=str, default="best",
        choices=["best", "final", "checkpoint"],
        help=(
            "Which model file to load (default: best).  "
            "checkpoint requires --checkpoint_step."
        ),
    )
    parser.add_argument(
        "--checkpoint_step", type=int, default=None,
        help=(
            "If --model_type checkpoint, the step number of the "
            "checkpoint to load, e.g. 150000 loads "
            "checkpoint_150000_steps.zip."
        ),
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50,
        help="Number of test episodes (default 50).",
    )
    parser.add_argument(
        "--no_render", action="store_true",
        help=(
            "Run without GUI.  Faster, suitable for scripted "
            "evaluation.  Default is human-visualised."
        ),
    )
    parser.add_argument(
        "--visual_delay", type=float, default=0.1,
        help=(
            "Seconds between frames when rendering (default 0.1).  "
            "Set to 0 for max speed.  Ignored if --no_render."
        ),
    )
    parser.add_argument(
        "--save_results", action="store_true",
        help=(
            "If set, write a per-episode CSV named "
            "test_results_<model_type>.csv into the run folder."
        ),
    )
    args = parser.parse_args()

    if args.model_type == "checkpoint" and args.checkpoint_step is None:
        parser.error(
            "--model_type checkpoint requires --checkpoint_step N"
        )
    return args


# =====================================================================
def main():
    args = parse_args()

    # ---------------- locate run folder and model file ----------------
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            raise SystemExit(f"ERROR: run folder not found: {run_dir}")
    else:
        run_dir = find_latest_run()

    model_path = resolve_model_path(
        run_dir, args.model_type, args.checkpoint_step
    )
    if not model_path.exists():
        print(f"ERROR: model file not found at {model_path}")
        print(f"Available zip files in {run_dir}:")
        for name in list_available_models(run_dir):
            print(f"  - {name}")
        sys.exit(1)

    print("=" * 60)
    print("Reach phase test")
    print("=" * 60)
    print(f"Run folder:   {run_dir}")
    print(f"Model file:   {model_path.name}")
    print(f"Episodes:     {args.num_episodes}")
    print(f"Render mode:  {'none' if args.no_render else 'human'}")
    print("=" * 60)

    # ---------------- env and model ----------------
    render_mode = None if args.no_render else "human"
    env = PandaReachEnv(render_mode=render_mode)
    print("Environment created.")

    try:
        model = PPO.load(str(model_path), env=env)
        print(f"Model loaded from {model_path.name}.")
    except Exception as e:
        print(f"ERROR loading model.\n  {e}")
        env.close()
        sys.exit(1)

    # ---------------- test loop ----------------
    print(f"\nRunning {args.num_episodes} test episodes.")
    print("-" * 60)

    results = []
    n_success = 0

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        step_count = 0

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if not args.no_render:
                env.render()
            episode_reward += float(reward)
            step_count += 1
            if not args.no_render and args.visual_delay > 0:
                time.sleep(args.visual_delay)

        success       = bool(info.get("reach_success", False))
        bottle_knocked = bool(info.get("bottle_knocked", False))
        timed_out      = (not success) and (not bottle_knocked)

        if success:
            n_success += 1
            status = "SUCCESS"
            reason = ""
        elif bottle_knocked:
            status = "FAILED"
            reason = " (bottle knocked)"
        else:
            status = "FAILED"
            reason = " (timeout)"

        results.append({
            "episode":         ep + 1,
            "success":         int(success),
            "bottle_knocked":  int(bottle_knocked),
            "timed_out":       int(timed_out),
            "n_steps":         step_count,
            "episode_reward":  float(episode_reward),
        })

        print(f"Episode {ep + 1:3d}/{args.num_episodes}.  {status}{reason}  "
              f"steps {step_count:4d}  reward {episode_reward:8.1f}")

    env.close()

    # ---------------- summary ----------------
    mean_reward = sum(r["episode_reward"] for r in results) / len(results)
    mean_steps  = sum(r["n_steps"]        for r in results) / len(results)
    success_rate = 100.0 * n_success / len(results)
    n_knocked = sum(r["bottle_knocked"] for r in results)
    n_timeout = sum(r["timed_out"]      for r in results)

    print()
    print("=" * 60)
    print("Test summary")
    print("=" * 60)
    print(f"Success rate:     {n_success}/{args.num_episodes}  "
          f"({success_rate:.1f}%)")
    print(f"Bottle knocked:   {n_knocked}/{args.num_episodes}")
    print(f"Timed out:        {n_timeout}/{args.num_episodes}")
    print(f"Mean reward:      {mean_reward:.1f}")
    print(f"Mean steps:       {mean_steps:.0f}")
    print("=" * 60)

    # ---------------- optional CSV ----------------
    if args.save_results:
        csv_name = f"test_results_{args.model_type}.csv"
        csv_path = run_dir / csv_name
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote per-episode results to: {csv_path}")


if __name__ == "__main__":
    main()