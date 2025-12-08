import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 1. Load the data
# Paste your log data into a file named 'log.txt' or change this variable
log_file_path = 'log.txt'

steps = []
hand_positions = []
targets = []
distances = []

# 2. Parse the log file using Regex
with open(log_file_path, 'r') as f:
    content = f.read()

    # Extract Steps
    steps = [int(s) for s in re.findall(r"Step:\s+(\d+)", content)]

    # Extract Hand Positions (x, y, z)
    # Finds strings inside brackets after "Hand pos:"
    hand_pos_strings = re.findall(r"Hand pos:\s+\[(.*?)\]", content)
    for pos in hand_pos_strings:
        # Convert "0.355, 0.000, 1.424" to floats
        coords = [float(x) for x in pos.split(',')]
        hand_positions.append(coords)

    # Extract Targets
    target_strings = re.findall(r"Reach target:\s+\[(.*?)\]", content)
    for tgt in target_strings:
        coords = [float(x) for x in tgt.split(',')]
        targets.append(coords)

    # Extract 3D Distance (for the error graph)
    distances = [float(d) for d in re.findall(r"Distance 3D:\s+([\d\.]+)", content)]

# Convert to numpy arrays for easier plotting
hand_positions = np.array(hand_positions)
targets = np.array(targets)

# 3. Plotting

fig = plt.figure(figsize=(14, 6))

# --- Plot 1: 3D Trajectory ---
ax1 = fig.add_subplot(121, projection='3d')

# Plot the Hand Path (Blue line/dots)
ax1.plot(hand_positions[:, 0], hand_positions[:, 1], hand_positions[:, 2],
         label='Hand Path', color='blue', alpha=0.6)
ax1.scatter(hand_positions[:, 0], hand_positions[:, 1], hand_positions[:, 2],
            c=steps, cmap='viridis', s=20, label='Hand Pos (Color=Step)')

# Plot the Target (Red X)
# Taking the first target found (assuming target is static per episode)
if len(targets) > 0:
    ax1.scatter(targets[0, 0], targets[0, 1], targets[0, 2],
                color='red', marker='x', s=100, label='Target')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Hand Trajectory')
ax1.legend()

# --- Plot 2: Convergence (Distance over Time) ---
ax2 = fig.add_subplot(122)
ax2.plot(steps, distances, marker='o', color='orange', linestyle='-')
ax2.set_xlabel('Simulation Steps')
ax2.set_ylabel('Distance to Target (m)')
ax2.set_title('Convergence Speed (Distance 3D)')
ax2.grid(True)

plt.tight_layout()
plt.show()