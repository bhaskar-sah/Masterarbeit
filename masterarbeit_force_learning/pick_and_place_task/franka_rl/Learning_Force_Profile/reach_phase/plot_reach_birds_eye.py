"""
plot_reach_birds_eye.py
(located at Learning_Force_Profile/reach_phase/)

Load best_model.zip from a reach-phase training run and run one
deterministic episode to capture the hand trajectory.  Produce a
top-down bird's-eye view showing the bottle, the hand trajectory
from the home configuration to the contact point, the start
marker, and the contact-establishment marker.  This is the
Section 5.4 figure that visually demonstrates the reach policy
closing the gap and establishing stable contact.

By default the script picks the most recent timestamp folder
under reach_phase/training/.  Pass --run_dir to override.

Outputs (saved under <run_dir>/plots/):
  reach_birds_eye.{pdf,png}

Usage from this folder.

  # default: latest run, load best_model.zip
  python plot_reach_birds_eye.py

  # explicit run folder
  python plot_reach_birds_eye.py \\
      --run_dir training/2026-06-16_01-36-21

  # use final_model.zip instead of best_model.zip
  python plot_reach_birds_eye.py --model_type final
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO


SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
_PARENT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from reach_env import PandaReachEnv  # noqa: E402


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


def resolve_model_path(run_dir, model_type):
    if model_type == "best":
        return run_dir / "best_model.zip"
    if model_type == "final":
        return run_dir / "final_model.zip"
    raise ValueError(f"unknown model_type: {model_type}")


# =====================================================================
def run_episode(env, model):
    """Run one deterministic episode and capture spatial trajectories."""
    obs, _ = env.reset()
    hand_x = []
    hand_y = []
    bottle_x = []
    bottle_y = []

    contact_step = None
    step = 0
    done = False
    info = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        hand_pos = env.data.xpos[env.hand_body_id]
        bottle_pos = env.data.xpos[env.bottle_body_id]
        hand_x.append(float(hand_pos[0]))
        hand_y.append(float(hand_pos[1]))
        bottle_x.append(float(bottle_pos[0]))
        bottle_y.append(float(bottle_pos[1]))

        if contact_step is None and info.get("is_touching", False):
            contact_step = step

        step += 1

    return {
        "hand_x":         np.array(hand_x),
        "hand_y":         np.array(hand_y),
        "bottle_x":       np.array(bottle_x),
        "bottle_y":       np.array(bottle_y),
        "contact_step":   contact_step,
        "success":        bool(info.get("reach_success", False)),
        "n_steps":        step,
    }


# =====================================================================
def draw_figure(logs, env, outdir):
    """Compose the bird's-eye view figure."""
    fig, ax = plt.subplots(figsize=(LATEX_TEXTWIDTH_INCHES, LATEX_TEXTWIDTH_INCHES))

    bottle_radius = float(env.config.bottle_radius)

    # ----- bottle as a filled circle at its mean position -----
    bx = float(np.mean(logs["bottle_x"]))
    by = float(np.mean(logs["bottle_y"]))
    bottle_circle = plt.Circle(
        (bx, by), radius=bottle_radius,
        facecolor="#5B9BD5", edgecolor="#1F4E79",
        linewidth=1.5, alpha=0.55,
        zorder=4, label="object",
    )
    ax.add_patch(bottle_circle)

    # ----- hand trajectory as a line -----
    ax.plot(logs["hand_x"], logs["hand_y"],
            color="#1B998B", linewidth=2.0,
            label="hand trajectory", zorder=5)

    # ----- direction arrows along the trajectory -----
    n_arrows = 5
    n_pts = len(logs["hand_x"])
    if n_pts > 10:
        indices = np.linspace(3, n_pts - 5, n_arrows, dtype=int)
        for idx in indices:
            dx = logs["hand_x"][idx + 2] - logs["hand_x"][idx]
            dy = logs["hand_y"][idx + 2] - logs["hand_y"][idx]
            ax.annotate(
                "",
                xy=(logs["hand_x"][idx] + dx * 2.5,
                    logs["hand_y"][idx] + dy * 2.5),
                xytext=(logs["hand_x"][idx], logs["hand_y"][idx]),
                arrowprops=dict(
                    arrowstyle="-|>,head_length=0.8,head_width=0.55",
                    color="#1B998B", lw=1.2, alpha=0.75),
                zorder=5,
            )

    # ----- start marker -----
    ax.scatter(logs["hand_x"][0], logs["hand_y"][0],
               color="black", s=140, marker="o",
               zorder=6, label="start")

    # ----- contact establishment marker -----
    cs = logs["contact_step"]
    if cs is not None:
        ax.scatter(logs["hand_x"][cs], logs["hand_y"][cs],
                   s=180, marker="*",
                   color="#5C9E2B", edgecolor="black",
                   linewidth=0.8,
                   zorder=7, label="contact established")

    # ----- end of trajectory marker (if different from contact) -----
    if cs is None or cs != n_pts - 1:
        ax.scatter(logs["hand_x"][-1], logs["hand_y"][-1],
                   color="#1B998B", s=110, marker="X",
                   zorder=6, label="hand end")

    # ----- axes and title -----
    ax.set_xlabel(r"$x$ position [m]")
    ax.set_ylabel(r"$y$ position [m]")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5,
            color="gray", alpha=0.40)

    #ax.set_title("Reach trajectory", pad=8)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=9,
        frameon=True, edgecolor="black", fancybox=False,
        borderaxespad=0.,
    )

    save(fig, outdir, "reach_birds_eye")
    return outdir / "reach_birds_eye.pdf", outdir / "reach_birds_eye.png"


# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one deterministic reach episode with the trained "
            "policy and produce a bird's-eye view of the hand "
            "trajectory closing the gap to the object."
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
        choices=["best", "final"],
        help=(
            "Which model file to load (default: best)."
        ),
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            raise SystemExit(f"ERROR: run folder not found: {run_dir}")
    else:
        run_dir = find_latest_run()

    model_path = resolve_model_path(run_dir, args.model_type)
    if not model_path.exists():
        raise SystemExit(f"ERROR: model file not found: {model_path}")

    outdir = run_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Reach bird's-eye view")
    print("=" * 60)
    print(f"Run folder:   {run_dir}")
    print(f"Model file:   {model_path.name}")
    print(f"Output dir:   {outdir}")
    print()

    env = PandaReachEnv(render_mode=None)
    model = PPO.load(str(model_path), env=env)

    print(">>> Running one deterministic episode ...")
    logs = run_episode(env, model)
    print(f"    steps   = {logs['n_steps']}")
    print(f"    success = {logs['success']}")
    if logs["contact_step"] is not None:
        print(f"    contact step = {logs['contact_step']}")
    env.close()

    pdf_path, png_path = draw_figure(logs, env, outdir)
    print(f"\nWrote: {pdf_path}")
    print(f"Wrote: {png_path}")
    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()