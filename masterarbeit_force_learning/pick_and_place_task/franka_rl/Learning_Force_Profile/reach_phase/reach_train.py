# """
# Training Script for Reach Phase.

# Trains a PPO agent to reach from home position to stable contact with bottle.
# This is trajectory-agnostic: the agent learns to approach and contact the
# bottle from the home position regardless of what trajectory will be pushed.

# Usage:
#     python reach_train.py
# """

# import os
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.callbacks import CheckpointCallback

# from masterarbeit_force_learning.pick_and_place_task.franka_rl.Learning_Force_Profile.reach_phase.reach_config import ReachConfig
# from masterarbeit_force_learning.pick_and_place_task.franka_rl.Learning_Force_Profile.reach_phase.reach_env import PandaReachEnv


# # ==================== CONFIGURATION ====================
# TOTAL_TIMESTEPS = 200_000
# MODEL_NAME = "reach_model_31"

# # ==================== SETUP ====================
# print("=" * 60)
# print("TRAINING: Panda Reach to Bottle Contact")
# print("=" * 60)
# print(f"Algorithm: PPO")
# print(f"Timesteps: {TOTAL_TIMESTEPS}")
# print(f"Goal: Reach bottle and establish stable contact")
# print("=" * 60)

# config = ReachConfig()
# env = PandaReachEnv(render_mode=None, config=config)
# print("Training environment created.")

# try:
#     check_env(env)
#     print("Environment check passed!")
# except Exception as e:
#     print(f"Environment check failed: {e}")
#     env.close()
#     exit()

# # ==================== CREATE MODEL ====================
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
#     ent_coef=0.01,
#     vf_coef=0.5,
#     max_grad_norm=0.5,
#     tensorboard_log="./ppo_reach_tensorboard/",
#     device="cpu"
# )

# # ==================== CALLBACKS ====================
# current_script_dir = os.path.dirname(os.path.realpath(__file__))
# save_folder = os.path.join(current_script_dir, "saved_models")
# os.makedirs(save_folder, exist_ok=True)

# checkpoint_callback = CheckpointCallback(
#     save_freq=50000,
#     save_path=save_folder,
#     name_prefix=MODEL_NAME
# )

# # ==================== TRAIN ====================
# print("\nStarting training...")
# print("Monitor with: tensorboard --logdir ./ppo_reach_tensorboard/")
# print("-" * 60)

# model.learn(
#     total_timesteps=TOTAL_TIMESTEPS,
#     callback=checkpoint_callback,
#     progress_bar=True
# )

# # ==================== SAVE ====================
# model_save_path = os.path.join(save_folder, MODEL_NAME)
# model.save(model_save_path)
# env.close()

# print("\n" + "=" * 60)
# print(f"Training complete!")
# print(f"Model saved to: {model_save_path}.zip")
# print("=" * 60)



###############################################################################################


"""
reach_train.py
(located at Learning_Force_Profile/reach_phase/)

Training script for the reach-phase policy.  Trains a PPO agent
to drive the Panda's end-effector from its home configuration
into stable contact with the object.  The reach phase is
trajectory-agnostic.  The agent learns to approach and contact
the object from the home position regardless of which trajectory
the push phase will eventually follow.

Run-folder layout.

  reach_phase/
      reach_train.py
      reach_config.py
      reach_env.py
      training/
          <YYYY-MM-DD_HH-MM-SS>/
              tensorboard_logs/
                  PPO_1/
                      events.out.tfevents.*
              eval_logs/
                  evaluations.npz
              train_monitor.csv
              eval_monitor.csv
              run_config.json
              checkpoint_<step>_steps.zip
              best_model.zip
              final_model.zip

Every training run creates a fresh timestamped subfolder under
training/ so previous runs are never overwritten.

Usage.

  # default 200 000 timesteps
  python reach_train.py

  # custom budget
  python reach_train.py --total_timesteps 500000

  # custom suffix appended to the timestamp folder
  python reach_train.py --suffix experiment_A

Monitor a live run from another terminal with.

  tensorboard --logdir reach_phase/training/<run-folder>/tensorboard_logs/
"""

import argparse
import datetime
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


# ---------------------------------------------------------------------
# Path setup.  Add the reach_phase folder to sys.path so the local
# imports of reach_config and reach_env work regardless of the
# current working directory the user launches the script from.
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, str(SCRIPT_DIR))

from reach_config import ReachConfig          # noqa: E402
from reach_env    import PandaReachEnv        # noqa: E402


# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the reach-phase PPO policy.  Every run writes "
            "to reach_phase/training/<timestamp>/."
        ),
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=200_000,
        help="Total training timesteps (default 200000).",
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=50_000,
        help="Save a checkpoint every N timesteps (default 50000).",
    )
    parser.add_argument(
        "--eval_freq", type=int, default=10_000,
        help=(
            "Run the evaluation callback every N timesteps "
            "(default 10000)."
        ),
    )
    parser.add_argument(
        "--n_eval_episodes", type=int, default=10,
        help="Episodes per evaluation pass (default 10).",
    )
    parser.add_argument(
        "--suffix", type=str, default=None,
        help=(
            "Optional name suffix appended to the timestamp folder "
            "(e.g. --suffix experiment_A produces "
            "<timestamp>_experiment_A/)."
        ),
    )
    return parser.parse_args()


# =====================================================================
def make_run_folder(suffix=None):
    """Create reach_phase/training/<timestamp>[_<suffix>]/."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{timestamp}_{suffix}" if suffix else timestamp
    run_dir = SCRIPT_DIR / "training" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# =====================================================================
def dump_run_config(args, config, run_dir):
    """Write a JSON snapshot of the training configuration.

    Records the CLI arguments and the ReachConfig dataclass so the
    exact hyperparameters of this run can be recovered later.
    """
    record = {
        "timestamp":  datetime.datetime.now().isoformat(),
        "script":     "reach_train.py",
        "cli_args": {
            "total_timesteps":  args.total_timesteps,
            "checkpoint_freq":  args.checkpoint_freq,
            "eval_freq":        args.eval_freq,
            "n_eval_episodes":  args.n_eval_episodes,
            "suffix":           args.suffix,
        },
        "ppo_hyperparameters": {
            "policy":             "MlpPolicy",
            "learning_rate":      3e-4,
            "n_steps":            2048,
            "batch_size":         64,
            "n_epochs":           10,
            "gamma":              0.99,
            "gae_lambda":         0.95,
            "clip_range":         0.2,
            "ent_coef":           0.01,
            "vf_coef":            0.5,
            "max_grad_norm":      0.5,
            "device":             "cpu",
        },
        "reach_config": (
            asdict(config) if is_dataclass(config) else str(config)
        ),
    }
    out = run_dir / "run_config.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2, default=str)


# =====================================================================
def main():
    args = parse_args()

    run_dir       = make_run_folder(args.suffix)
    tb_dir        = run_dir / "tensorboard_logs"
    eval_log_dir  = run_dir / "eval_logs"
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Reach phase PPO training")
    print("=" * 60)
    print(f"Run folder:        {run_dir}")
    print(f"Tensorboard log:   {tb_dir}")
    print(f"Total timesteps:   {args.total_timesteps}")
    print(f"Checkpoint freq:   {args.checkpoint_freq}")
    print(f"Eval freq:         {args.eval_freq}")
    print(f"Eval episodes:     {args.n_eval_episodes}")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Training environment with Monitor wrapper so episode rewards
    # and lengths are logged to a CSV next to the model files.
    # -----------------------------------------------------------------
    config = ReachConfig()
    train_env = PandaReachEnv(render_mode=None, config=config)
    train_env = Monitor(train_env,
                        filename=str(run_dir / "train_monitor"))
    print("Training environment created.")
    try:
        check_env(train_env)
        print("Environment check passed.")
    except Exception as e:
        print(f"Environment check failed.\n  {e}")
        train_env.close()
        sys.exit(1)

    # -----------------------------------------------------------------
    # Separate evaluation environment for the EvalCallback so the
    # deterministic eval rollouts do not pollute the training-state
    # statistics.
    # -----------------------------------------------------------------
    eval_env = PandaReachEnv(render_mode=None, config=ReachConfig())
    eval_env = Monitor(eval_env,
                       filename=str(run_dir / "eval_monitor"))

    # -----------------------------------------------------------------
    # Snapshot the run configuration to a JSON file inside the run
    # folder.  Useful for re-discovering hyperparameters months later.
    # -----------------------------------------------------------------
    dump_run_config(args, config, run_dir)

    # -----------------------------------------------------------------
    # PPO with the same hyperparameters as the previous reach script,
    # with the tensorboard log directed into the run folder and the
    # device pinned to CPU because the MLP policy is small enough
    # that the host-to-device transfer dominates any GPU speedup.
    # -----------------------------------------------------------------
    model = PPO(
        "MlpPolicy",
        train_env,
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
        tensorboard_log=str(tb_dir),
        device="cpu",
    )

    # -----------------------------------------------------------------
    # Callbacks.  Checkpoint every N steps and run a deterministic
    # evaluation every M steps so a best_model.zip is preserved
    # whenever the policy improves.
    # -----------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(run_dir),
        name_prefix="checkpoint",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(eval_log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks = [checkpoint_callback, eval_callback]

    # -----------------------------------------------------------------
    # Train.
    # -----------------------------------------------------------------
    print("\nStarting training.")
    print(f"Monitor with.\n  tensorboard --logdir {tb_dir}")
    print("-" * 60)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # -----------------------------------------------------------------
    # Save the final model and close environments.
    # -----------------------------------------------------------------
    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))
    train_env.close()
    eval_env.close()

    print()
    print("=" * 60)
    print("Training complete.")
    print(f"Run folder:    {run_dir}")
    print(f"Final model:   {final_model_path}.zip")
    print(f"Best model:    {run_dir / 'best_model.zip'} "
          f"(if EvalCallback found one)")
    print(f"Checkpoints:   checkpoint_<step>_steps.zip in run folder")
    print(f"Tensorboard:   {tb_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()