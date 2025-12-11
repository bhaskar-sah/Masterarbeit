import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys

# --- 1. ROBUST FILE PATH ---
# This ensures we look in the same folder as the script,
# no matter where you run the command from.
current_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILENAME = os.path.join(current_dir, "experiment_log.txt")
SAVE_PLOT_AS = os.path.join(current_dir, "results_plot.png")


def parse_log_file(filename):
    if not os.path.exists(filename):
        print(f"CRITICAL ERROR: File not found at:\n{filename}")
        print("-> Please make sure 'experiment_log.txt' is in the same folder as this script.")
        return [], [], []

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"File found! Size: {len(content)} characters.")
    if len(content) < 10:
        print("-> WARNING: File seems empty! Did you save the text inside it?")
        return [], [], []

    # --- REGEX PARSING ---
    # Relaxed patterns to handle slight variations in copy-pasting
    # We look for "Step: <number>", "Distance 3D: <number>", etc.
    step_pattern = re.compile(r"Step:\s*(\d+)")
    dist_pattern = re.compile(r"Distance 3D:\s*([0-9.]+)")
    quat_pattern = re.compile(r"Quat error:\s*([0-9.]+)")

    steps, distances, quats = [], [], []

    # We don't split by "===" anymore because copy-pasting might break the line length.
    # Instead, we just find ALL matches in the file sequentially.
    all_steps = step_pattern.findall(content)
    all_dists = dist_pattern.findall(content)
    all_quats = quat_pattern.findall(content)

    # Convert to numbers
    # We take the minimum length to ensure we don't crash if data is mismatched
    min_len = min(len(all_steps), len(all_dists), len(all_quats))

    print(f"Debug: Found {len(all_steps)} steps, {len(all_dists)} distances, {len(all_quats)} quats.")

    if min_len == 0:
        return [], [], []

    # Slice arrays to match length
    steps = [int(s) for s in all_steps[:min_len]]
    distances = [float(d) for d in all_dists[:min_len]]
    quats = [float(q) for q in all_quats[:min_len]]

    return np.array(steps), np.array(distances), np.array(quats)


# --- MAIN EXECUTION ---
print(f"Reading logs from: {LOG_FILENAME}")
steps, dist_3d, quat_err = parse_log_file(LOG_FILENAME)

if len(steps) == 0:
    print("\nERROR: No valid data found.")
    print("Check your text file content. It should look like:")
    print("Step: 0")
    print("Distance 3D: 0.5971")
    print("Quat error: 0.0004")
    sys.exit()

print(f"Success! Plotting {len(steps)} data points...")

# Create the Dual-Axis Plot
fig, ax1 = plt.subplots(figsize=(12, 7))

color = 'tab:blue'
ax1.set_xlabel('Simulation Steps', fontsize=12)
ax1.set_ylabel('Distance to Target (m)', color=color, fontsize=12, fontweight='bold')
ax1.plot(steps, dist_3d, color=color, marker='o', linewidth=2, label='Position Error')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

# Add Success Line
ax1.axhline(y=0.03, color='green', linestyle=':', linewidth=2, label='3cm Threshold')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Quaternion Error (Orientation)', color=color, fontsize=12, fontweight='bold')
ax2.plot(steps, quat_err, color=color, marker='s', linestyle='--', linewidth=2, label='Orientation Error')
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f'Robot Performance Analysis', fontsize=14)
fig.tight_layout()

plt.savefig(SAVE_PLOT_AS)
print(f"Plot saved to: {SAVE_PLOT_AS}")
plt.show()