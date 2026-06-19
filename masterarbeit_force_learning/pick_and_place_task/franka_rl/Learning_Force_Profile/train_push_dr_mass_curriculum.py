"""
train_push_dr_mass_curriculum.py

Curriculum training with mass domain randomization.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import mujoco
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from env import PandaPushTrajectoryEnv


# =====================================================================
# Model name — bump version for each new run
# =====================================================================
MODEL_NAME = "trained_model_dr_mass_curriculum_v62"

TRAINING_MASS_KG = 0.5   # reference / nominal mass used in the XML

CURRICULUM = [
    # Phase 1: s_curve, fixed 0.5 kg — build the core skill from scratch
    {"traj_type": "s_curve",  "mass_range": (0.50, 0.50),  "timesteps": 500_000},
    # Phase 2: s_curve, slight mass variation — first exposure to DR
    {"traj_type": "s_curve",  "mass_range": (0.35, 0.70),  "timesteps": 200_000},
    # Phase 3: s_curve, full mass range
    {"traj_type": "s_curve",  "mass_range": (0.10, 1.50),  "timesteps": 300_000},
    # Phase 4: curved, moderate mass range
    {"traj_type": "curved",   "mass_range": (0.30, 0.80),  "timesteps": 200_000},
    # Phase 5: curved, full mass range
    {"traj_type": "curved",   "mass_range": (0.10, 1.50),  "timesteps": 150_000},
    # Phase 6: straight, moderate mass range
    {"traj_type": "straight", "mass_range": (0.30, 0.80),  "timesteps": 150_000},
    # Phase 7: straight, full mass range
    {"traj_type": "straight", "mass_range": (0.10, 1.50),  "timesteps": 150_000},
    # Phase 8: hardening — all 3 trajectories mixed + full mass range
    {"traj_type": "mixed",    "mass_range": (0.10, 1.50),  "timesteps": 500_000},
]

MIXED_TRAJECTORIES = ["straight", "curved", "s_curve"]


# =====================================================================
# Gym wrappers
# =====================================================================

class MassDomainRandomizationWrapper(gym.Wrapper):
    """Randomises bottle mass on every reset().

    Mutates env.model.body_mass directly (persists across MuJoCo
    resets because reset() restores qpos/qvel from the keyframe, not
    the model parameters).  Inertia is scaled proportionally so
    principal-axis inertia stays consistent with the new mass.
    """

    def __init__(self, env, mass_range=(0.40, 0.60)):
        super().__init__(env)
        self.mass_range = mass_range
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        mass = float(self._rng.uniform(*self.mass_range))
        self._apply_mass(mass)
        obs, info = self.env.reset(**kwargs)
        info["mass_kg"] = mass
        return obs, info

    def _apply_mass(self, mass):
        model    = self.env.model
        body_id  = self.env.bottle_body_id
        old_mass = float(model.body_mass[body_id])
        if old_mass > 1e-9:
            scale = mass / old_mass
            model.body_mass[body_id]    = mass
            model.body_inertia[body_id] = model.body_inertia[body_id] * scale


class TrajectoryMixerWrapper(gym.Wrapper):
    """Randomly picks one trajectory type on every reset().

    Injects the chosen trajectory into the options dict that env.reset()
    already supports, so no changes to env.py are needed.
    """

    def __init__(self, env, trajectories=None):
        super().__init__(env)
        self.trajectories = trajectories or MIXED_TRAJECTORIES
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        traj = str(self._rng.choice(self.trajectories))
        opts = kwargs.get("options") or {}
        opts["trajectory_type"] = traj
        kwargs["options"] = opts
        return self.env.reset(**kwargs)


# =====================================================================
# Environment factory
# =====================================================================

def make_train_env(traj_type, mass_range):
    """Create a training environment with mass DR (and trajectory mixing
    if traj_type == 'mixed')."""
    base_traj = MIXED_TRAJECTORIES[0] if traj_type == "mixed" else traj_type
    env = PandaPushTrajectoryEnv(render_mode=None, trajectory_type=base_traj)
    env = MassDomainRandomizationWrapper(env, mass_range=mass_range)
    if traj_type == "mixed":
        env = TrajectoryMixerWrapper(env, trajectories=MIXED_TRAJECTORIES)
    return env


def make_eval_env(traj_type, mass_range):
    """Create a monitored eval env matching the training phase.

    For the 'mixed' phase, eval runs on s_curve (the hardest trajectory)
    with the full mass range so EvalCallback's best_model is the most
    robust checkpoint.
    """
    if traj_type == "mixed":
        eval_traj = "s_curve"
    else:
        eval_traj = traj_type
    env = PandaPushTrajectoryEnv(render_mode=None, trajectory_type=eval_traj)
    env = MassDomainRandomizationWrapper(env, mass_range=mass_range)
    env = Monitor(env)
    return env


# =====================================================================
# Model loader
# =====================================================================

def load_or_create_model(finetune_model_dir, first_env, base_save_folder):
    """Load existing model for fine-tuning, or create a fresh PPO."""
    if finetune_model_dir:
        script_dir  = Path(os.path.dirname(os.path.realpath(__file__)))
        candidates  = [
            script_dir / "saved_models_new" / finetune_model_dir / "best_model.zip",
            script_dir / "saved_models_new" / finetune_model_dir / f"final_model_{finetune_model_dir}.zip",
            Path(finetune_model_dir) / "best_model.zip",
        ]
        model_path  = next((c for c in candidates if c.exists()), None)
        if model_path is None:
            raise SystemExit(
                f"ERROR: could not find model zip in {finetune_model_dir}.\n"
                f"Tried: {[str(c) for c in candidates]}"
            )
        print(f"Fine-tuning from: {model_path}")
        model = PPO.load(
            str(model_path),
            env=first_env,
            tensorboard_log=os.path.join(base_save_folder, "tensorboard_logs"),
            device="cpu",
        )
        # Keep the existing hyperparameters but allow learning rate reset.
        model.learning_rate = 1e-4   # lower LR for fine-tuning
        return model

    print("Starting fresh PPO model.")
    return PPO(
        "MlpPolicy",
        first_env,
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
        tensorboard_log=os.path.join(base_save_folder, "tensorboard_logs"),
        device="cpu",
    )


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum + mass DR training for the push policy.",
    )
    parser.add_argument(
        "--finetune_model", type=str, default=None,
        help=(
            "Directory name under saved_models_new/ of an existing model "
            "to fine-tune (recommended: v60).  "
            "Omit to start from scratch."
        ),
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run only 1000 steps per phase for a quick sanity check.",
    )
    parser.add_argument(
        "--model_name", type=str, default=MODEL_NAME,
        help=f"Output model folder name (default: {MODEL_NAME}).",
    )
    args = parser.parse_args()

    script_dir       = os.path.dirname(os.path.realpath(__file__))
    base_save_folder = os.path.join(script_dir, "saved_models_new", args.model_name)
    os.makedirs(base_save_folder, exist_ok=True)

    curriculum = CURRICULUM
    if args.smoke_test:
        curriculum = [
            {**p, "timesteps": 1_000} for p in curriculum
        ]

    total_steps = sum(p["timesteps"] for p in curriculum)

    print("=" * 60)
    print("TRAINING: Curriculum + Mass Domain Randomization")
    print("=" * 60)
    print(f"Model name:    {args.model_name}")
    print(f"Save dir:      {base_save_folder}")
    print(f"Fine-tune:     {args.finetune_model or 'no (fresh model)'}")
    print(f"Total steps:   {total_steps:,}")
    print(f"Smoke test:    {args.smoke_test}")
    print()
    print(f"{'Phase':<6} {'Trajectory':<14} {'Mass range [kg]':<20} {'Steps':>10}")
    print("-" * 55)
    for i, p in enumerate(curriculum, 1):
        lo, hi = p["mass_range"]
        print(f"  {i:<4} {p['traj_type']:<14} [{lo:.2f}, {hi:.2f}]{'':>12} {p['timesteps']:>10,}")
    print()

    # ---- Sanity check the env ----
    test_env = make_train_env("straight", (0.4, 0.6))
    try:
        check_env(test_env)
        print("Environment check passed.")
    except Exception as e:
        print(f"WARNING: environment check raised: {e}")
    finally:
        test_env.close()

    # ---- Build initial env and model ----
    first_phase = curriculum[0]
    first_env   = make_train_env(first_phase["traj_type"],
                                  first_phase["mass_range"])
    model = load_or_create_model(
        args.finetune_model, first_env, base_save_folder
    )
    first_env.close()

    # ---- Run curriculum ----
    print("\nStarting curriculum training...")
    print(f"TensorBoard:  tensorboard --logdir "
          f"{os.path.join(base_save_folder, 'tensorboard_logs')}\n")

    global_best_reward = -np.inf
    global_best_path   = os.path.join(base_save_folder, "best_model")

    for phase_idx, phase in enumerate(curriculum, 1):
        traj_type  = phase["traj_type"]
        mass_range = phase["mass_range"]
        steps      = phase["timesteps"]

        phase_label  = f"phase_{phase_idx}_{traj_type}"
        phase_folder = os.path.join(base_save_folder, phase_label)
        os.makedirs(phase_folder, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  PHASE {phase_idx}: {traj_type.upper()}"
              f"  |  mass [{mass_range[0]:.2f}, {mass_range[1]:.2f}] kg"
              f"  |  {steps:,} steps")
        print(f"{'=' * 60}")

        train_env = make_train_env(traj_type, mass_range)
        eval_env  = make_eval_env(traj_type, mass_range)

        model.set_env(train_env)

        checkpoint_cb = CheckpointCallback(
            save_freq=max(50_000, steps // 4),
            save_path=phase_folder,
            name_prefix=f"ckpt_{traj_type}",
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=phase_folder,
            log_path=phase_folder,
            eval_freq=max(20_000, steps // 8),
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )

        model.learn(
            total_timesteps=steps,
            callback=[checkpoint_cb, eval_cb],
            progress_bar=True,
            reset_num_timesteps=False,
        )

        final_path = os.path.join(phase_folder, f"final_{traj_type}")
        model.save(final_path)
        print(f"  Saved final phase model: {final_path}.zip")

        # ---- Track global best across all phases ----
        # Load the phase best_model (saved by EvalCallback) and compare
        # its eval reward against the global best.  This prevents later
        # phases from overwriting a high-performing earlier checkpoint.
        phase_best_path = os.path.join(phase_folder, "best_model.zip")
        if os.path.exists(phase_best_path):
            eval_npz = os.path.join(phase_folder, "evaluations.npz")
            if os.path.exists(eval_npz):
                d = np.load(eval_npz)
                phase_best_reward = float(d["results"].mean(axis=1).max())
                print(f"  Phase {phase_idx} best eval reward: {phase_best_reward:.2f}  "
                      f"(global best so far: {global_best_reward:.2f})")
                if phase_best_reward > global_best_reward:
                    global_best_reward = phase_best_reward
                    import shutil
                    shutil.copy2(phase_best_path, global_best_path + ".zip")
                    print(f"  *** New global best! Copied to best_model.zip ***")

        train_env.close()
        eval_env.close()

    print("\n" + "=" * 60)
    print("Curriculum + mass DR training complete!")
    print(f"Global best reward:  {global_best_reward:.2f}")
    print(f"Global best model:   {global_best_path}.zip")
    print(f"All phases:          {base_save_folder}/")
    print("=" * 60)
    print()
    print("Evaluate the trained model with:")
    print(f"  cd evaluation/07_Domain_randomization")
    print(f"  python evaluate_mass_fixed_friction.py \\")
    print(f"      --model_dir {args.model_name}")


if __name__ == "__main__":
    main()
