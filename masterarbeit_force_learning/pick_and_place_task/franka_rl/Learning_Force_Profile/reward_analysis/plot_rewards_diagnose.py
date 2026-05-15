# """
# Plot reward components and trajectory metrics for the Panda Push Environment.

# Usage:
#     python plot_rewards.py --model_path <path_to_model.zip> --trajectory_type straight

# This runs one episode and plots:
#     1. Individual reward components over time
#     2. Trajectory deviation + progress
#     3. Angle error θ + wrist rotation
#     4. Force magnitude + stiffness K
#     5. Bottle tilt
#     6. Actual bottle path vs desired trajectory (bird's eye view)
# """

# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import os
# import sys


# def run_episode_and_collect(env, model, trajectory_type="straight"):
#     """Run one episode and collect detailed logs."""

#     obs, _ = env.reset(options={"trajectory_type": trajectory_type})

#     logs = {
#         "step": [],
#         "progress": [],
#         "deviation": [],
#         "angle_error_deg": [],
#         "force_mag": [],
#         "tilt": [],
#         "stiffness": [],
#         "wrist_deg": [],
#         "target_z": [],
#         "is_touching": [],
#         "is_settling": [],
#         # Individual rewards
#         "r_progress": [],
#         "r_deviation": [],
#         "r_angle": [],
#         "r_stability": [],
#         "r_contact": [],
#         "r_total": [],
#         # Positions
#         "bottle_x": [],
#         "bottle_y": [],
#         "hand_x": [],
#         "hand_y": [],
#         "hand_z": [],
#     }

#     # Store trajectory for plotting
#     trajectory = env.trajectory.copy() if env.trajectory is not None else None

#     done = False
#     step = 0
#     prev_progress = 0.0
#     prev_action = np.zeros(env.action_space.shape, dtype=np.float32)  # ADD THIS

#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated

#         # Get current state
#         bottle_xy = env.data.xpos[env.bottle_body_id][:2]
#         hand_pos = env.data.xpos[env.hand_body_id]
#         tilt = env._get_bottle_tilt()
#         is_touching = env._is_touching()
#         force = env._get_contact_force()
#         force_mag = float(np.linalg.norm(force))
#         angle_error = env._get_angle_error(bottle_xy)
#         angle_deg = np.degrees(angle_error)
#         _, deviation_mag = env._get_path_deviation(bottle_xy)
#         progress = env._get_progress(bottle_xy)
#         K_avg = float(np.mean(env.current_K))
#         wrist_deg = float(np.degrees(env.wrist_offset))

#         # Adaptive target z
#         # Adaptive target z (safe for both old and new env)
#         bottle_y = env.data.xpos[env.bottle_body_id][1]
#         bottle_start_y = getattr(env, 'bottle_start_y', 0.2)
#         distance_from_start = abs(bottle_y - bottle_start_y)
#         if hasattr(env, 'bottle_start_y'):
#             current_target_z = env.target_z + 0.02 * distance_from_start
#             current_target_z = max(current_target_z, 0.88)
#         else:
#             current_target_z = env.target_z  # Old env: fixed height

#         # Compute individual reward components (mirror the reward function)
#         if env.in_approach:
#             dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)
#             r_progress = 0.0
#             r_deviation = 0.0
#             r_angle = 0.0
#             r_stability = 0.0
#             r_contact = 0.0
#             r_smooth = 0.0
#             r_total = -2.0 * dist_to_bottle - 5.0 * abs(hand_pos[2] - env.target_z)
#         else:
#             # Progress
#             progress_delta = progress - prev_progress
#             r_progress = 100.0 * max(progress_delta, 0)

#             # Deviation
#             r_deviation = 5.0 - 100.0 * deviation_mag
#             r_deviation = max(r_deviation, -15.0)

#             # Angle
#             # if angle_deg < 10:
#             #     r_angle = 2.0
#             # elif angle_deg < 30:
#             #     r_angle = 1.0
#             # elif angle_deg < 60:
#             #     r_angle = 0.0
#             # else:
#             #     r_angle = -2.0

#             # Angle (NEW CONTINUOUS LOGIC)
#             r_angle = float(np.clip(2.0 - (angle_deg / 15.0), -2.0, 2.0))

#             # Stability
#             if tilt > 0.995:
#                 r_stability = 3.0
#             elif tilt > 0.99:
#                 r_stability = 1.0
#             elif tilt > 0.98:
#                 r_stability = 0.0
#             else:
#                 r_stability = -15.0 * (1 - tilt)

#             # Contact
#             dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)
#             if is_touching:
#                 r_contact = 2.0
#             else:
#                 r_contact = -10.0 * dist_to_bottle
#                 if dist_to_bottle > 0.08:
#                     r_contact -= 5.0

#             # Action Smoothing (NEW LOGIC)
#             k_delta = np.linalg.norm(action[2:4] - prev_action[2:4])
#             r_smooth = -0.5 * k_delta  # NOTE: Match this multiplier to whatever you used in the env!

#             # Total
#             r_total = r_progress + r_deviation + r_angle + r_stability + r_contact + r_smooth - 0.005

#         prev_progress = progress
#         prev_action = action.copy()

#         # Log everything
#         logs["step"].append(step)
#         logs["progress"].append(progress * 100)
#         logs["deviation"].append(deviation_mag * 100)  # cm
#         logs["angle_error_deg"].append(angle_deg)
#         logs["force_mag"].append(force_mag)
#         logs["tilt"].append(tilt)
#         logs["stiffness"].append(K_avg)
#         logs["wrist_deg"].append(wrist_deg)
#         logs["target_z"].append(current_target_z)
#         logs["is_touching"].append(1.0 if is_touching else 0.0)
#         logs["is_settling"].append(1.0 if env.is_settling else 0.0)
#         logs["r_progress"].append(r_progress)
#         logs["r_deviation"].append(r_deviation)
#         logs["r_angle"].append(r_angle)
#         logs["r_stability"].append(r_stability)
#         logs["r_contact"].append(r_contact)
#         logs["r_total"].append(r_total)
#         logs["bottle_x"].append(float(bottle_xy[0]))
#         logs["bottle_y"].append(float(bottle_xy[1]))
#         logs["hand_x"].append(float(hand_pos[0]))
#         logs["hand_y"].append(float(hand_pos[1]))
#         logs["hand_z"].append(float(hand_pos[2]))

#         step += 1

#     # Convert to numpy
#     for key in logs:
#         logs[key] = np.array(logs[key])

#     return logs, trajectory, info


# def plot_rewards(logs, trajectory, info, save_path=None):
#     """Create comprehensive reward analysis plots."""

#     steps = logs["step"]

#     fig = plt.figure(figsize=(20, 24))
#     gs = GridSpec(6, 2, figure=fig, hspace=0.35, wspace=0.3)
#     fig.suptitle(
#         f"Episode Analysis | Progress: {logs['progress'][-1]:.1f}% | "
#         f"Final Dev: {logs['deviation'][-1]:.1f}cm | "
#         f"Success: {info.get('is_success', False)}",
#         fontsize=14, fontweight='bold'
#     )

#     # ==================== 1. Individual Reward Components ====================
#     ax1 = fig.add_subplot(gs[0, :])
#     ax1.plot(steps, logs["r_progress"], label="r_progress", color="#2ca02c", alpha=0.8, linewidth=1)
#     ax1.plot(steps, logs["r_deviation"], label="r_deviation", alpha=0.8, linewidth=1)
#     ax1.plot(steps, logs["r_angle"], label="r_angle", alpha=0.8, linewidth=1)
#     ax1.plot(steps, logs["r_stability"], label="r_stability", alpha=0.8, linewidth=1)
#     ax1.plot(steps, logs["r_contact"], label="r_contact", alpha=0.8, linewidth=1)
#     ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
#     ax1.set_xlabel("Step")
#     ax1.set_ylabel("Reward")
#     ax1.set_title("Individual Reward Components")
#     ax1.legend(loc='upper right', ncol=5, fontsize=8)
#     ax1.grid(True, alpha=0.3)

#     # ==================== 2. Total Reward ====================
#     ax2 = fig.add_subplot(gs[1, 0])
#     ax2.plot(steps, logs["r_total"], color='black', alpha=0.8, linewidth=1)
#     # Add cumulative reward on secondary axis
#     ax2b = ax2.twinx()
#     cumulative = np.cumsum(logs["r_total"])
#     ax2b.plot(steps, cumulative, color='blue', alpha=0.5, linewidth=1, linestyle='--')
#     ax2b.set_ylabel("Cumulative Reward", color='blue')
#     ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
#     ax2.set_xlabel("Step")
#     ax2.set_ylabel("Step Reward")
#     ax2.set_title("Total Reward (solid) + Cumulative (dashed)")
#     ax2.grid(True, alpha=0.3)

#     # ==================== 3. Progress + Deviation ====================
#     ax3 = fig.add_subplot(gs[1, 1])
#     ax3.plot(steps, logs["progress"], label="Progress (%)", color='green', linewidth=2)
#     ax3b = ax3.twinx()
#     ax3b.plot(steps, logs["deviation"], label="Deviation (cm)", color='red', linewidth=1.5)
#     ax3b.axhline(y=5.0, color='red', linestyle='--', alpha=0.3, label="5cm tolerance")
#     ax3.set_xlabel("Step")
#     ax3.set_ylabel("Progress (%)", color='green')
#     ax3b.set_ylabel("Deviation (cm)", color='red')
#     ax3.set_title("Progress & Deviation")
#     lines1, labels1 = ax3.get_legend_handles_labels()
#     lines2, labels2 = ax3b.get_legend_handles_labels()
#     ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
#     ax3.grid(True, alpha=0.3)

#     # ==================== 4. Angle Error + Wrist ====================
#     ax4 = fig.add_subplot(gs[2, 0])
#     ax4.plot(steps, logs["angle_error_deg"], label="θ (angle error)", color='purple', alpha=0.7, linewidth=1)
#     ax4b = ax4.twinx()
#     ax4b.plot(steps, logs["wrist_deg"], label="Wrist (deg)", color='orange', alpha=0.7, linewidth=1)
#     ax4.set_xlabel("Step")
#     ax4.set_ylabel("Angle Error θ (deg)", color='purple')
#     ax4b.set_ylabel("Wrist Rotation (deg)", color='orange')
#     ax4.set_title("Force Alignment & Wrist Rotation")
#     lines1, labels1 = ax4.get_legend_handles_labels()
#     lines2, labels2 = ax4b.get_legend_handles_labels()
#     ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
#     ax4.grid(True, alpha=0.3)

#     # ==================== 5. Force + Stiffness ====================
#     ax5 = fig.add_subplot(gs[2, 1])
#     ax5.plot(steps, logs["force_mag"], label="Force (N)", color='blue', alpha=0.7, linewidth=1)
#     ax5b = ax5.twinx()
#     ax5b.plot(steps, logs["stiffness"], label="K (N/m)", color='brown', alpha=0.7, linewidth=1)
#     ax5.set_xlabel("Step")
#     ax5.set_ylabel("Contact Force (N)", color='blue')
#     ax5b.set_ylabel("Stiffness K (N/m)", color='brown')
#     ax5.set_title("Force & Stiffness Profile")
#     lines1, labels1 = ax5.get_legend_handles_labels()
#     lines2, labels2 = ax5b.get_legend_handles_labels()
#     ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
#     ax5.grid(True, alpha=0.3)

#     # ==================== 6. Tilt + Contact + Settle ====================
#     ax6 = fig.add_subplot(gs[3, 0])
#     ax6.plot(steps, logs["tilt"], label="Tilt", color='darkgreen', linewidth=1.5)
#     ax6.axhline(y=0.99, color='green', linestyle='--', alpha=0.3, label="tilt_ok (0.99)")
#     ax6.axhline(y=0.96, color='orange', linestyle='--', alpha=0.3, label="tilt_stop (0.96)")
#     ax6.axhline(y=0.97, color='blue', linestyle='--', alpha=0.3, label="settle_exit (0.97)")
#     # Shade settle regions
#     settle = logs["is_settling"]
#     for i in range(len(settle) - 1):
#         if settle[i] > 0.5:
#             ax6.axvspan(steps[i], steps[i + 1], color='red', alpha=0.1)
#     ax6.set_xlabel("Step")
#     ax6.set_ylabel("Tilt (1.0 = upright)")
#     ax6.set_title("Bottle Tilt (red = settling)")
#     ax6.legend(loc='lower left', fontsize=8)
#     ax6.set_ylim(0.9, 1.01)
#     ax6.grid(True, alpha=0.3)

#     # ==================== 7. Hand Height + Adaptive Z ====================
#     ax7 = fig.add_subplot(gs[3, 1])
#     ax7.plot(steps, logs["hand_z"], label="Hand Z (actual)", color='blue', linewidth=1.5)
#     ax7.plot(steps, logs["target_z"], label="Target Z (adaptive)", color='red', linewidth=1, linestyle='--')
#     ax7.axhline(y=0.88, color='black', linestyle=':', alpha=0.3, label="min_safe_z (0.88)")
#     ax7.set_xlabel("Step")
#     ax7.set_ylabel("Height (m)")
#     ax7.set_title("Hand Height vs Adaptive Target")
#     ax7.legend(loc='upper left', fontsize=8)
#     ax7.grid(True, alpha=0.3)

#     # ==================== 8. Bird's Eye View: Trajectory ====================
#     ax8 = fig.add_subplot(gs[4:, :])
#     if trajectory is not None:
#         ax8.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label="Desired trajectory", zorder=1)
#     ax8.plot(logs["bottle_x"], logs["bottle_y"], 'b-', linewidth=1.5, alpha=0.8, label="Actual bottle path", zorder=2)
#     ax8.plot(logs["hand_x"], logs["hand_y"], 'g-', linewidth=0.8, alpha=0.5, label="Hand path", zorder=1)

#     # Mark start and end
#     ax8.scatter(logs["bottle_x"][0], logs["bottle_y"][0], c='blue', s=100, marker='o', zorder=3, label="Start")
#     ax8.scatter(logs["bottle_x"][-1], logs["bottle_y"][-1], c='blue', s=100, marker='*', zorder=3, label="End")

#     # Color dots by deviation magnitude
#     scatter = ax8.scatter(
#         logs["bottle_x"][::10], logs["bottle_y"][::10],
#         c=logs["deviation"][::10], cmap='RdYlGn_r', s=20,
#         vmin=0, vmax=5, zorder=4, alpha=0.8
#     )
#     plt.colorbar(scatter, ax=ax8, label="Deviation (cm)", shrink=0.6)

#     ax8.set_xlabel("X (m)")
#     ax8.set_ylabel("Y (m)")
#     ax8.set_title("Bird's Eye View: Desired vs Actual Trajectory")
#     ax8.legend(loc='upper right', fontsize=8)
#     ax8.set_aspect('equal')
#     ax8.grid(True, alpha=0.3)

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"Plot saved to: {save_path}")
#     else:
#         plt.show()

#     return fig


# def main():
#     parser = argparse.ArgumentParser(description="Plot reward components for push environment")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .zip file")
#     parser.add_argument("--trajectory_type", type=str, default="straight", choices=["straight", "curved", "s_curve"])
#     parser.add_argument("--save_path", type=str, default=None, help="Save plot to file (e.g., rewards.png)")
#     parser.add_argument("--env_path", type=str, default=None, help="Path to environment directory")
#     args = parser.parse_args()

#     # Import after parsing args so we can add to path if needed
#     if args.env_path:
#         sys.path.insert(0, args.env_path)

#     from stable_baselines3 import PPO
#     # from push_with_finger_learn_force_impedance_14_v01_straight import PandaPushTrajectoryEnv
#     from pick_and_place_task.franka_rl.Learning_Force_Profile.other_recent_files.push_with_finger_learn_force_impedance_14_v01_straight_reward_correction import PandaPushTrajectoryEnv

#     # Create environment (no rendering for data collection)
#     env = PandaPushTrajectoryEnv(render_mode=None, trajectory_type=args.trajectory_type)

#     # Load model
#     print(f"Loading model: {args.model_path}")
#     model = PPO.load(args.model_path, env=env)

#     # Run episode and collect data
#     print(f"Running episode with trajectory: {args.trajectory_type}")
#     logs, trajectory, info = run_episode_and_collect(env, model, args.trajectory_type)

#     print(f"\nEpisode complete:")
#     print(f"  Progress: {logs['progress'][-1]:.1f}%")
#     print(f"  Final deviation: {logs['deviation'][-1]:.1f}cm")
#     print(f"  Max deviation: {np.max(logs['deviation']):.1f}cm")
#     print(f"  Success: {info.get('is_success', False)}")
#     print(f"  Steps: {len(logs['step'])}")

#     # Generate save path if not specified
#     if args.save_path is None:
#         model_name = os.path.splitext(os.path.basename(args.model_path))[0]
#         args.save_path = f"reward_analysis_{model_name}_{args.trajectory_type}.png"

#     # Plot
#     plot_rewards(logs, trajectory, info, save_path=args.save_path)

#     env.close()


# if __name__ == "__main__":
#     main()

####################################################################################################

# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import os
# import sys

# # =========================================================================
# # PATH FIX: Tell Python to look one folder up to find env.py
# # =========================================================================
# current_dir = os.path.dirname(os.path.realpath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

# from stable_baselines3 import PPO
# from env import PandaPushTrajectoryEnv

# def run_episode_and_collect(env, model, trajectory_type="straight"):
#     """Run one episode and collect detailed logs based on pure velocity control."""

#     obs, _ = env.reset(options={"trajectory_type": trajectory_type})

#     logs = {
#         "step": [],
#         "progress": [],
#         "deviation": [],
#         "tilt": [],
#         "is_touching": [],
        
#         # Actions
#         "action_vx": [],
#         "action_vy": [],
#         "action_wz": [],
        
#         # Velocities
#         "v_des_mag": [],
#         "v_curr_mag": [],
#         "wz_des": [],
#         "wz_curr": [],
        
#         # Forces
#         "F_cmd_mag": [],
#         "F_meas_mag": [],
        
#         # Individual rewards
#         "r_progress": [],
#         "r_deviation": [],
#         "r_stability": [],
#         "r_contact": [],
#         "r_alignment": [],
#         "r_position": [],
#         "r_orientation": [],
#         "r_total": [],
        
#         # Positions for bird's eye view
#         "bottle_x": [],
#         "bottle_y": [],
#         "hand_x": [],
#         "hand_y": [],
#     }

#     # Store trajectory for plotting
#     trajectory = env.traj_manager.trajectory.copy() if env.traj_manager.trajectory is not None else None

#     done = False
#     step = 0

#     while not done:
#         # Predict action (deterministic=True strips out random noise)
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated

#         # Extract Physics / Tracking
#         bottle_xy = env.data.xpos[env.bottle_body_id][:2].copy()
#         hand_pos = env.data.xpos[env.hand_body_id].copy()
        
#         # Velocities
#         v_curr = env.push_controller.get_ee_velocity()
#         w_curr = env.push_controller.get_ee_angular_velocity()
        
#         v_des_mag = np.linalg.norm([action[0] * env.config.v_max, action[1] * env.config.v_max])
#         v_curr_mag = np.linalg.norm(v_curr[:2]) # Only XY for planar pushing
#         wz_des = action[2] * env.config.w_max
#         wz_curr = w_curr[2]
        
#         # Forces
#         F_cmd = env.push_controller.last_F_cmd
#         F_meas = env.contact_manager.get_contact_force()
#         F_cmd_mag = np.linalg.norm(F_cmd[:2]) # Planar force command
#         F_meas_mag = np.linalg.norm(F_meas[:2]) # Planar measured force

#         # Log everything
#         logs["step"].append(step)
#         logs["progress"].append(info.get("progress", 0.0) * 100)
#         logs["deviation"].append(info.get("deviation", 0.0) * 100)  # cm
#         logs["tilt"].append(info.get("bottle_tilt", 1.0))
#         logs["is_touching"].append(1.0 if info.get("is_touching", False) else 0.0)
        
#         logs["action_vx"].append(action[0])
#         logs["action_vy"].append(action[1])
#         logs["action_wz"].append(action[2])
        
#         logs["v_des_mag"].append(v_des_mag * 100)   # Convert to cm/s for readable plots
#         logs["v_curr_mag"].append(v_curr_mag * 100) # Convert to cm/s
#         logs["wz_des"].append(wz_des)
#         logs["wz_curr"].append(wz_curr)
        
#         logs["F_cmd_mag"].append(F_cmd_mag)
#         logs["F_meas_mag"].append(F_meas_mag)
        
#         logs["r_progress"].append(info.get("r_progress", 0.0))
#         logs["r_deviation"].append(info.get("r_deviation", 0.0))
#         logs["r_stability"].append(info.get("r_stability", 0.0))
#         logs["r_contact"].append(info.get("r_contact", 0.0))
#         logs["r_alignment"].append(info.get("r_alignment", 0.0))
#         logs["r_position"].append(info.get("r_position", 0.0))
#         # Safely grab orientation in case it wasn't added to info dict
#         logs["r_orientation"].append(info.get("r_orientation", 0.0)) 
#         logs["r_total"].append(reward)
        
#         logs["bottle_x"].append(float(bottle_xy[0]))
#         logs["bottle_y"].append(float(bottle_xy[1]))
#         logs["hand_x"].append(float(hand_pos[0]))
#         logs["hand_y"].append(float(hand_pos[1]))

#         step += 1

#     # Convert lists to numpy arrays
#     for key in logs:
#         logs[key] = np.array(logs[key])

#     return logs, trajectory, info


# def plot_rewards(logs, trajectory, info, save_path=None):
#     """Create comprehensive diagnostic dashboard."""

#     steps = logs["step"]
#     fig = plt.figure(figsize=(22, 26))
#     gs = GridSpec(7, 2, figure=fig, hspace=0.4, wspace=0.25)
    
#     fig.suptitle(
#         f"Diagnostic Dashboard | Final Progress: {logs['progress'][-1]:.1f}% | "
#         f"Final Dev: {logs['deviation'][-1]:.1f}cm | Success: {info.get('is_success', False)}",
#         fontsize=16, fontweight='bold'
#     )

#     # ==================== 1. Individual Rewards ====================
#     ax1 = fig.add_subplot(gs[0, :])
#     ax1.plot(steps, logs["r_progress"], label="Progress (+)", color="green", linewidth=1.5)
#     ax1.plot(steps, logs["r_alignment"], label="Force Align (+)", color="blue", linewidth=1.5)
#     ax1.plot(steps, logs["r_deviation"], label="Deviation (-)", color="red", linewidth=1.5, alpha=0.7)
#     ax1.plot(steps, logs["r_position"], label="Position Err (-)", color="purple", linewidth=1.5, alpha=0.7)
#     ax1.plot(steps, logs["r_orientation"], label="Orientation Err (-)", color="orange", linewidth=1.5, alpha=0.7)
#     ax1.plot(steps, logs["r_contact"], label="Contact/Dist (-)", color="brown", linewidth=1.5, alpha=0.7)
    
#     ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
#     ax1.set_title("Reward Components Breakdown")
#     ax1.set_ylabel("Reward Value")
#     ax1.legend(loc='lower left', ncol=6, fontsize=9)
#     ax1.grid(True, alpha=0.3)

#     # ==================== 2. Raw Actions ====================
#     ax2 = fig.add_subplot(gs[1, 0])
#     ax2.plot(steps, logs["action_vx"], label="vx (Forward)", color="blue")
#     ax2.plot(steps, logs["action_vy"], label="vy (Lateral)", color="green")
#     ax2.plot(steps, logs["action_wz"], label="wz (Yaw)", color="purple")
#     ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
#     ax2.set_ylim(-1.1, 1.1)
#     ax2.set_title("Raw Neural Network Actions [-1, 1]")
#     ax2.set_ylabel("Action Command")
#     ax2.legend(loc='upper right', fontsize=9)
#     ax2.grid(True, alpha=0.3)

#     # ==================== 3. Forces (Command vs Measured) ====================
#     ax3 = fig.add_subplot(gs[1, 1])
#     ax3.plot(steps, logs["F_cmd_mag"], label="F_cmd (from Impedance)", color="red", linestyle="--")
#     ax3.plot(steps, logs["F_meas_mag"], label="F_measured (Actual Contact)", color="blue", linewidth=1.5)
#     ax3.set_title("Force Profile: Command vs Reality (XY Plane)")
#     ax3.set_ylabel("Force (Newtons)")
#     ax3.legend(loc='upper right', fontsize=9)
#     ax3.grid(True, alpha=0.3)

#     # ==================== 4. Linear Velocity Tracking ====================
#     ax4 = fig.add_subplot(gs[2, 0])
#     ax4.plot(steps, logs["v_des_mag"], label="Desired v (cm/s)", color="red", linestyle="--")
#     ax4.plot(steps, logs["v_curr_mag"], label="Actual v (cm/s)", color="green", linewidth=1.5)
#     ax4.set_title("Planar Velocity Tracking (Mag)")
#     ax4.set_ylabel("Velocity (cm/s)")
#     ax4.legend(loc='upper right', fontsize=9)
#     ax4.grid(True, alpha=0.3)

#     # ==================== 5. Angular Velocity Tracking ====================
#     ax5 = fig.add_subplot(gs[2, 1])
#     ax5.plot(steps, logs["wz_des"], label="Desired wz (rad/s)", color="red", linestyle="--")
#     ax5.plot(steps, logs["wz_curr"], label="Actual wz (rad/s)", color="purple", linewidth=1.5)
#     ax5.set_title("Wrist Yaw Velocity Tracking")
#     ax5.set_ylabel("Angular Velocity (rad/s)")
#     ax5.legend(loc='upper right', fontsize=9)
#     ax5.grid(True, alpha=0.3)

#     # ==================== 6. Task Progress & Deviation ====================
#     ax6 = fig.add_subplot(gs[3, :])
#     ax6.plot(steps, logs["progress"], label="Progress (%)", color="green", linewidth=2)
#     ax6b = ax6.twinx()
#     ax6b.plot(steps, logs["deviation"], label="Deviation (cm)", color="red", linewidth=2)
#     ax6b.axhline(y=5.0, color='red', linestyle=':', alpha=0.5, label="Tolerance (5cm)")
    
#     ax6.set_title("Task Performance: Progress vs Deviation")
#     ax6.set_ylabel("Progress (%)", color="green")
#     ax6b.set_ylabel("Deviation (cm)", color="red")
    
#     lines1, labels1 = ax6.get_legend_handles_labels()
#     lines2, labels2 = ax6b.get_legend_handles_labels()
#     ax6.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)
#     ax6.grid(True, alpha=0.3)

#     # ==================== 7. Bird's Eye View Trajectory ====================
#     ax7 = fig.add_subplot(gs[4:, :])
#     if trajectory is not None:
#         ax7.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=3, label="Desired Trajectory", zorder=1)
    
#     ax7.plot(logs["bottle_x"], logs["bottle_y"], 'b-', linewidth=2, label="Actual Bottle Path", zorder=2)
#     ax7.plot(logs["hand_x"], logs["hand_y"], 'g-', linewidth=1, alpha=0.5, label="Hand Path", zorder=1)
    
#     ax7.scatter(logs["bottle_x"][0], logs["bottle_y"][0], c='cyan', s=100, marker='o', zorder=3, label="Start")
#     ax7.scatter(logs["bottle_x"][-1], logs["bottle_y"][-1], c='magenta', s=150, marker='*', zorder=3, label="End")

#     # Heatmap scatter of deviation
#     scatter = ax7.scatter(
#         logs["bottle_x"][::5], logs["bottle_y"][::5],
#         c=logs["deviation"][::5], cmap='Reds', s=30,
#         vmin=0, vmax=10, zorder=4, alpha=0.9
#     )
#     plt.colorbar(scatter, ax=ax7, label="Deviation (cm)", shrink=0.7)

#     ax7.set_title("Bird's Eye View: X-Y Trajectory Map")
#     ax7.set_xlabel("X Position (m)")
#     ax7.set_ylabel("Y Position (m)")
#     ax7.legend(loc='lower left', fontsize=10)
#     ax7.set_aspect('equal')
#     ax7.grid(True, alpha=0.4)

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"Plot saved to: {save_path}")
#     else:
#         plt.show()

#     # plt.show()

#     return fig



# def main():
#     parser = argparse.ArgumentParser(description="Plot diagnostic dashboard for pure velocity control.")
#     parser.add_argument("--model_dir", type=str, required=True, help="Name of the model directory")
#     parser.add_argument("--trajectory_type", type=str, default="s_curve", choices=["straight", "curved", "s_curve"])
#     parser.add_argument("--save", action="store_true", help="Save the plot to the reward_results directory")
#     args = parser.parse_args()

#     # Create environment
#     env = PandaPushTrajectoryEnv(render_mode=None, trajectory_type=args.trajectory_type)

#     # =========================================================================
#     # PATH FIX: Stepping in and out of folders correctly
#     # =========================================================================
#     # current_dir IS your 'reward_analysis' folder
#     current_dir = os.path.dirname(os.path.realpath(__file__)) 
    
#     # parent_dir IS your 'Learning_Force_Profile' folder
#     parent_dir = os.path.dirname(current_dir) 

#     # To find the model, we start at parent_dir, then go into saved_models_new
#     model_path = os.path.join(parent_dir, "saved_models_new", args.model_dir, "best_model.zip")

#     print(f"\nLoading best model from: {model_path}")
#     if not os.path.exists(model_path):
#         print(f"ERROR: Could not find best_model.zip at {model_path}")
#         exit()

#     model = PPO.load(model_path, env=env)

#     # Run episode
#     print(f"Running episode with trajectory: {args.trajectory_type}...")
#     logs, trajectory, info = run_episode_and_collect(env, model, args.trajectory_type)

#     print(f"\nEpisode complete! Generating plots...")
#     print(f"  Progress: {logs['progress'][-1]:.1f}%")
#     print(f"  Final deviation: {logs['deviation'][-1]:.1f}cm")

#     # Generate save path inside the nested reward_results folder
#     save_path = None
#     if args.save:
#         # Since current_dir is already 'reward_analysis', we just add 'reward_results'
#         analysis_dir = os.path.join(current_dir, "reward_results")
#         os.makedirs(analysis_dir, exist_ok=True) 
        
#         # Save the file here
#         save_path = os.path.join(analysis_dir, f"{args.model_dir}_{args.trajectory_type}.png")

#     # Plot
#     plot_rewards(logs, trajectory, info, save_path=save_path)
#     env.close()

# if __name__ == "__main__":
#     main()


################################################################################################
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import matplotlib.ticker as ticker

# =========================================================================
# PATH FIX: Tell Python to look one folder up to find env.py
# =========================================================================
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from stable_baselines3 import PPO
from env import PandaPushTrajectoryEnv

def run_episode_and_collect(env, model, trajectory_type="straight"):
    """Run one episode and collect detailed logs based on pure velocity control."""

    obs, _ = env.reset(options={"trajectory_type": trajectory_type})

    logs = {
        "step": [],
        "time": [],          # <--- Added time tracking key
        "progress": [],
        "deviation": [],
        "tilt": [],
        "is_touching": [],
        
        # Actions
        "action_vx": [],
        "action_vy": [],
        "action_wz": [],
        
        # Velocities
        "v_des_mag": [],
        "v_curr_mag": [],
        "wz_des": [],
        "wz_curr": [],
        
        # Forces
        "F_cmd_mag": [],
        "F_meas_mag": [],
        
        # Individual rewards
        "r_progress": [],
        "r_deviation": [],
        "r_stability": [],
        "r_contact": [],
        "r_alignment": [],
        "r_position": [],
        "r_orientation": [],
        "r_total": [],
        
        # Positions for bird's eye view
        "bottle_x": [],
        "bottle_y": [],
        "hand_x": [],
        "hand_y": [],
    }

    # Store trajectory for plotting
    trajectory = env.traj_manager.trajectory.copy() if env.traj_manager.trajectory is not None else None

    done = False
    step = 0

    while not done:
        # Predict action (deterministic=True strips out random noise)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Extract Physics / Tracking
        bottle_xy = env.data.xpos[env.bottle_body_id][:2].copy()
        hand_pos = env.data.xpos[env.hand_body_id].copy()
        
        # Velocities
        v_curr = env.push_controller.get_ee_velocity()
        w_curr = env.push_controller.get_ee_angular_velocity()
        
        v_des_mag = np.linalg.norm([action[0] * env.config.v_max, action[1] * env.config.v_max])
        v_curr_mag = np.linalg.norm(v_curr[:2]) # Only XY for planar pushing
        wz_des = action[2] * env.config.w_max
        wz_curr = w_curr[2]
        
        # Forces
        F_cmd = env.push_controller.last_F_cmd
        F_meas = env.contact_manager.get_contact_force()
        F_cmd_mag = np.linalg.norm(F_cmd[:2]) # Planar force command
        F_meas_mag = np.linalg.norm(F_meas[:2]) # Planar measured force

        # Log everything
        logs["step"].append(step)
        logs["time"].append(step * 0.002) # <--- Populated time scaling (Step * dt)
        logs["progress"].append(info.get("progress", 0.0) * 100)
        logs["deviation"].append(info.get("deviation", 0.0) * 100)  # cm
        logs["tilt"].append(info.get("bottle_tilt", 1.0))
        logs["is_touching"].append(1.0 if info.get("is_touching", False) else 0.0)
        
        logs["action_vx"].append(action[0])
        logs["action_vy"].append(action[1])
        logs["action_wz"].append(action[2])
        
        logs["v_des_mag"].append(v_des_mag * 100)   # Convert to cm/s for readable plots
        logs["v_curr_mag"].append(v_curr_mag * 100) # Convert to cm/s
        logs["wz_des"].append(wz_des)
        logs["wz_curr"].append(wz_curr)
        
        logs["F_cmd_mag"].append(F_cmd_mag)
        logs["F_meas_mag"].append(F_meas_mag)
        
        logs["r_progress"].append(info.get("r_progress", 0.0))
        logs["r_deviation"].append(info.get("r_deviation", 0.0))
        logs["r_stability"].append(info.get("r_stability", 0.0))
        logs["r_contact"].append(info.get("r_contact", 0.0))
        logs["r_alignment"].append(info.get("r_alignment", 0.0))
        logs["r_position"].append(info.get("r_position", 0.0))
        # Safely grab orientation in case it wasn't added to info dict
        logs["r_orientation"].append(info.get("r_orientation", 0.0)) 
        logs["r_total"].append(reward)
        
        logs["bottle_x"].append(float(bottle_xy[0]))
        logs["bottle_y"].append(float(bottle_xy[1]))
        logs["hand_x"].append(float(hand_pos[0]))
        logs["hand_y"].append(float(hand_pos[1]))

        step += 1

    # Convert lists to numpy arrays
    for key in logs:
        logs[key] = np.array(logs[key])

    return logs, trajectory, info


def plot_rewards(logs, trajectory, info, save_path=None):
    """Create comprehensive diagnostic dashboard."""

    # Changed: Extracted the newly made physics time tracker for standard x-axes
    time_axis = logs["time"]
    fig = plt.figure(figsize=(22, 26))
    gs = GridSpec(7, 2, figure=fig, hspace=0.4, wspace=0.25)
    
    fig.suptitle(
        f"Diagnostic Dashboard | Final Progress: {logs['progress'][-1]:.1f}% | "
        f"Final Dev: {logs['deviation'][-1]:.1f}cm | Success: {info.get('is_success', False)}",
        fontsize=16, fontweight='bold'
    )

    # ==================== 1. Individual Rewards ====================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, logs["r_progress"], label="Progress (+)", color="green", linewidth=1.5)
    ax1.plot(time_axis, logs["r_alignment"], label="Force Align (+)", color="blue", linewidth=1.5)
    ax1.plot(time_axis, logs["r_deviation"], label="Deviation (-)", color="red", linewidth=1.5, alpha=0.7)
    ax1.plot(time_axis, logs["r_position"], label="Position Err (-)", color="purple", linewidth=1.5, alpha=0.7)
    ax1.plot(time_axis, logs["r_orientation"], label="Orientation Err (-)", color="orange", linewidth=1.5, alpha=0.7)
    ax1.plot(time_axis, logs["r_contact"], label="Contact/Dist (-)", color="brown", linewidth=1.5, alpha=0.7)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title("Reward Components Breakdown")
    ax1.set_xlabel("Simulation Time (seconds)")
    ax1.set_ylabel("Reward Value")
    ax1.legend(loc='lower left', ncol=6, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ==================== 2. Raw Actions ====================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_axis, logs["action_vx"], label="vx (Forward)", color="blue")
    ax2.plot(time_axis, logs["action_vy"], label="vy (Lateral)", color="green")
    ax2.plot(time_axis, logs["action_wz"], label="wz (Yaw)", color="purple")
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_title("Raw Neural Network Actions [-1, 1]")
    ax2.set_xlabel("Simulation Time (seconds)")
    ax2.set_ylabel("Action Command")
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ==================== 3. Forces (Command vs Measured) ====================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_axis, logs["F_cmd_mag"], label="F_cmd (from Impedance)", color="red", linestyle="--")
    ax3.plot(time_axis, logs["F_meas_mag"], label="F_measured (Actual Contact)", color="blue", linewidth=1.5)
    ax3.set_title("Force Profile: Command vs Reality (XY Plane)")
    ax3.set_xlabel("Simulation Time (seconds)")
    ax3.set_ylabel("Force (Newtons)")
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ==================== 4. Linear Velocity Tracking ====================
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time_axis, logs["v_des_mag"], label="Desired v (cm/s)", color="red", linestyle="--")
    ax4.plot(time_axis, logs["v_curr_mag"], label="Actual v (cm/s)", color="green", linewidth=1.5)
    ax4.set_title("Planar Velocity Tracking (Mag)")
    ax4.set_xlabel("Simulation Time (seconds)")
    ax4.set_ylabel("Velocity (cm/s)")
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ==================== 5. Angular Velocity Tracking ====================
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time_axis, logs["wz_des"], label="Desired wz (rad/s)", color="red", linestyle="--")
    ax5.plot(time_axis, logs["wz_curr"], label="Actual wz (rad/s)", color="purple", linewidth=1.5)
    ax5.set_title("Wrist Yaw Velocity Tracking")
    ax5.set_xlabel("Simulation Time (seconds)")
    ax5.set_ylabel("Angular Velocity (rad/s)")
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ==================== 6. Task Progress & Deviation ====================
    ax6 = fig.add_subplot(gs[3, :])
    ax6.plot(time_axis, logs["progress"], label="Progress (%)", color="green", linewidth=2)
    ax6b = ax6.twinx()
    ax6b.plot(time_axis, logs["deviation"], label="Deviation (cm)", color="red", linewidth=2)
    ax6b.axhline(y=5.0, color='red', linestyle=':', alpha=0.5, label="Tolerance (5cm)")
    
    ax6.set_title("Task Performance: Progress vs Deviation")
    ax6.set_xlabel("Simulation Time (seconds)")
    ax6.set_ylabel("Progress (%)", color="green")
    ax6b.set_ylabel("Deviation (cm)", color="red")
    
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)
    ax6.grid(True, alpha=0.3)

    # ==================== 7. Bird's Eye View Trajectory ====================
    # (Remains based on meters position variables since it maps space instead of time)
    ax7 = fig.add_subplot(gs[4:, :])
    if trajectory is not None:
        ax7.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=3, label="Desired Trajectory", zorder=1)
    
    ax7.plot(logs["bottle_x"], logs["bottle_y"], 'b-', linewidth=2, label="Actual Bottle Path", zorder=2)
    ax7.plot(logs["hand_x"], logs["hand_y"], 'g-', linewidth=1, alpha=0.5, label="Hand Path", zorder=1)
    
    ax7.scatter(logs["bottle_x"][0], logs["bottle_y"][0], c='cyan', s=100, marker='o', zorder=3, label="Start")
    ax7.scatter(logs["bottle_x"][-1], logs["bottle_y"][-1], c='magenta', s=150, marker='*', zorder=3, label="End")

    # Heatmap scatter of deviation
    scatter = ax7.scatter(
        logs["bottle_x"][::5], logs["bottle_y"][::5],
        c=logs["deviation"][::5], cmap='Reds', s=30,
        vmin=0, vmax=10, zorder=4, alpha=0.9
    )
    plt.colorbar(scatter, ax=ax7, label="Deviation (cm)", shrink=0.7)

    ax7.set_title("Bird's Eye View: X-Y Trajectory Map")
    ax7.set_xlabel("X Position (m)")
    ax7.set_ylabel("Y Position (m)")
    ax7.legend(loc='lower left', fontsize=10)
    ax7.set_aspect('equal')
    ax7.grid(True, alpha=0.4)

    # Apply the 0.1-second numbering gap to all time-based charts
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.tick_params(axis='x', labelsize=8, rotation=45)
    # =================================================================

    plt.tight_layout()

    # if save_path:
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #     print(f"Plot saved to: {save_path}")
    # else:
    #     plt.show()
    plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot diagnostic dashboard for pure velocity control.")
    parser.add_argument("--model_dir", type=str, required=True, help="Name of the model directory")
    parser.add_argument("--trajectory_type", type=str, default="s_curve", choices=["straight", "curved", "s_curve"])
    parser.add_argument("--save", action="store_true", help="Save the plot to the reward_results directory")
    args = parser.parse_args()

    # Create environment
    env = PandaPushTrajectoryEnv(render_mode=None, trajectory_type=args.trajectory_type)

    # =========================================================================
    # PATH FIX: Stepping in and out of folders correctly
    # =========================================================================
    current_dir = os.path.dirname(os.path.realpath(__file__)) 
    parent_dir = os.path.dirname(current_dir) 

    model_path = os.path.join(parent_dir, "saved_models_new", args.model_dir, "best_model.zip")

    print(f"\nLoading best model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Could not find best_model.zip at {model_path}")
        exit()

    model = PPO.load(model_path, env=env)

    # Run episode
    print(f"Running episode with trajectory: {args.trajectory_type}...")
    logs, trajectory, info = run_episode_and_collect(env, model, args.trajectory_type)

    print(f"\nEpisode complete! Generating plots...")
    print(f"  Progress: {logs['progress'][-1]:.1f}%")
    print(f"  Final deviation: {logs['deviation'][-1]:.1f}cm")

        # Generate save path inside the nested reward_results folder
    save_path = None
    if args.save:
        analysis_dir = os.path.join(current_dir, "reward_results")
        os.makedirs(analysis_dir, exist_ok=True) 
        
        # Define the base filename structure
        base_name = f"{args.model_dir}_{args.trajectory_type}"
        save_path = os.path.join(analysis_dir, f"{base_name}.png")
        
        # Check if the file already exists, and append _1, _2, etc. if it does
        counter = 1
        while os.path.exists(save_path):
            save_path = os.path.join(analysis_dir, f"{base_name}_{counter}.png")
            counter += 1


    # Plot
    plot_rewards(logs, trajectory, info, save_path=save_path)
    env.close()

if __name__ == "__main__":
    main()
