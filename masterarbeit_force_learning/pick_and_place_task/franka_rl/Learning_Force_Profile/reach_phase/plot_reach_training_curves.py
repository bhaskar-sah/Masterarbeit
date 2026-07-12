"""
plot_reach_training_curves.py
(located at Learning_Force_Profile/reach_phase/)

Read the tensorboard event files from a reach-phase training run
and produce a two-panel figure showing the rollout mean episode
reward and the periodic evaluation mean reward, both as a
function of training timesteps.  This is the headline Section
5.4 figure proving the reach policy converged during training.

By default the script picks the most recent timestamp folder
under reach_phase/training/.  Pass --run_dir to override.

Outputs (saved under <run_dir>/plots/):
  reach_training_curves.{pdf,png}

Usage from this folder.

  # default: latest run, default smoothing
  python plot_reach_training_curves.py

  # explicit run folder
  python plot_reach_training_curves.py \\
      --run_dir training/2026-06-16_01-36-21

  # disable EMA smoothing
  python plot_reach_training_curves.py --smooth 0
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))


# =====================================================================
# Thesis-matched style (Latin Modern, serif, LaTeX textwidth)
# =====================================================================
LATEX_TEXTWIDTH_INCHES = 5.8

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Latin Modern Roman", "Times New Roman",
                           "DejaVu Serif", "serif"],
    "mathtext.fontset":   "stix",
    "font.size":          11,
    "axes.titlesize":     11,
    "axes.labelsize":     11,
    "axes.linewidth":     0.8,
    "axes.edgecolor":     "black",
    "axes.facecolor":     "white",
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "grid.color":         "gray",
    "grid.alpha":         0.40,
    "legend.fontsize":    10,
    "legend.frameon":     True,
    "legend.framealpha":  1.0,
    "legend.edgecolor":   "black",
    "legend.fancybox":    False,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
})


# =====================================================================
def save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"    wrote: {name}.pdf + {name}.png")


# =====================================================================
def find_latest_run():
    """Pick the most recent timestamp folder under training/."""
    training_dir = SCRIPT_DIR / "training"
    if not training_dir.exists():
        raise SystemExit(f"ERROR: no training folder at {training_dir}")
    candidates = [
        d for d in training_dir.iterdir()
        if d.is_dir() and d.name[:4].isdigit()
    ]
    if not candidates:
        raise SystemExit(f"ERROR: no timestamp folders under {training_dir}")
    return sorted(candidates, key=lambda d: d.name, reverse=True)[0]


def find_event_files(run_dir):
    """Find any events.out.tfevents.* files under run_dir."""
    found = []
    for parent, _dirs, files in os.walk(run_dir):
        for f in files:
            if f.startswith("events.out.tfevents."):
                found.append(Path(parent))
                break
    return sorted(found)


def load_scalar(event_dir, tag):
    """Return (steps, values) arrays for one scalar tag, or (None, None)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator \
            import EventAccumulator
    except ImportError:
        raise SystemExit(
            "tensorboard is required.  Install with:  pip install tensorboard"
        )
    acc = EventAccumulator(str(event_dir),
                           size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None, None
    events = acc.Scalars(tag)
    if not events:
        return None, None
    steps = np.array([e.step for e in events])
    vals  = np.array([e.value for e in events])
    return steps, vals


def ema(values, alpha):
    """Exponential moving average smoothing."""
    if alpha <= 0 or alpha >= 1 or len(values) < 2:
        return values
    out = np.empty_like(values, dtype=float)
    last = float(values[0])
    for i, v in enumerate(values):
        last = alpha * float(v) + (1.0 - alpha) * last
        out[i] = last
    return out


def plot_panel(ax, steps, values, color, label, smooth_alpha,
               ylabel):
    """Plot a smoothed-on-top-of-faded-raw curve on a single panel."""
    if smooth_alpha > 0 and len(values) >= 5:
        smoothed = ema(values, smooth_alpha)
        ax.plot(steps, values, color=color, alpha=0.20, linewidth=0.9)
        ax.plot(steps, smoothed, color=color, linewidth=1.6,
                label=label)
    else:
        ax.plot(steps, values, color=color, linewidth=1.0,
                label=label)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5,
            color="gray", alpha=0.40)
    ax.legend(loc="best", fontsize=10)


# =====================================================================
def _fmt_k(x):
    """Format a step count as '25k', '100k', etc."""
    x = int(x)
    if x == 0:
        return "0"
    if x % 1_000 == 0:
        return f"{x // 1_000}k"
    return str(x)


def _nice_tick_spacing(max_steps):
    """Return a round tick spacing that gives ~5-8 ticks up to max_steps."""
    candidates = [
        5_000, 10_000, 25_000, 50_000, 100_000,
        200_000, 250_000, 500_000, 1_000_000,
    ]
    for c in candidates:
        if max_steps / c <= 8:
            return c
    return candidates[-1]


# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot reach-phase training convergence curves from a "
            "tensorboard event file."
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
        "--smooth", type=float, default=0.6,
        help=(
            "EMA smoothing alpha in [0, 1].  Default 0.6 matches "
            "TensorBoard's default.  Set 0 to disable."
        ),
    )
    args = parser.parse_args()

    # ----- locate run folder and event file -----
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            raise SystemExit(f"ERROR: run folder not found: {run_dir}")
    else:
        run_dir = find_latest_run()

    event_dirs = find_event_files(run_dir)
    if not event_dirs:
        raise SystemExit(
            f"ERROR: no events.out.tfevents.* under {run_dir}"
        )
    # If multiple, take the first (typically PPO_1).
    event_dir = event_dirs[0]

    outdir = run_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Reach training curves")
    print("=" * 60)
    print(f"Run folder:   {run_dir}")
    print(f"Event dir:    {event_dir}")
    print(f"Smoothing:    alpha = {args.smooth}")
    print(f"Output dir:   {outdir}")
    print()

    # ----- read tags -----
    train_steps, train_vals = load_scalar(event_dir, "rollout/ep_rew_mean")
    if train_vals is None:
        raise SystemExit(
            "ERROR: rollout/ep_rew_mean not found in event file."
        )

    eval_rew_steps, eval_rew_vals = load_scalar(event_dir, "eval/mean_reward")

    # ----- build the figure: 2 stacked panels -----
    fig, axes = plt.subplots(
        2, 1,
        figsize=(LATEX_TEXTWIDTH_INCHES, 6.0),
        sharex=True,
    )

    # Panel 1 — training mean episode reward
    plot_panel(
        axes[0], train_steps, train_vals,
        color="#2166AC",
        label="training mean episode reward",
        smooth_alpha=args.smooth,
        ylabel="Training Reward",
    )

    # Panel 2 — evaluation mean reward
    if eval_rew_vals is not None and len(eval_rew_vals) > 0:
        plot_panel(
            axes[1], eval_rew_steps, eval_rew_vals,
            color="#D6604D",
            label="evaluation mean reward",
            smooth_alpha=args.smooth,
            ylabel="Eval Reward",
        )
    else:
        axes[1].text(0.5, 0.5, "(no eval/mean_reward found)",
                     ha="center", va="center",
                     transform=axes[1].transAxes,
                     fontsize=10, color="gray")
        axes[1].set_ylabel("Eval Reward")

    # x-axis: auto-spaced ticks, range capped at 200k
    tick_spacing = _nice_tick_spacing(200_000)
    ticks = np.arange(0, 200_001, tick_spacing)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([_fmt_k(t) for t in ticks])
    axes[-1].set_xlim(0, 200_000)
    axes[-1].set_xlabel("Steps")

    fig.suptitle("Reach phase training convergence", fontsize=11)
    save(fig, outdir, "reach_training_curves")
    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()