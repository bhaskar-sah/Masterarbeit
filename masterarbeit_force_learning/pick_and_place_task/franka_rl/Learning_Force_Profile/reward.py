# """
# reward.py  (rebalanced)

# CHANGES from previous version (addressing "do nothing" local optimum):

#   1. r_deviation: QUADRATIC instead of linear, much lower cap (-3 vs -15),
#      and lower weight. Small deviations now nearly free; only catastrophic
#      deviations hurt. This lets the agent tolerate the messy contact
#      physics during early exploration.

#   2. r_alignment, r_position weights bumped (0.3 -> 0.5, 0.5 -> 1.0).
#      Doing the right thing now exceeds typical penalties.

#   3. off_path terminal penalty softened (-50 -> -10) so failed exploration
#      doesn't dominate gradient updates.

#   4. Sign convention: f_robot_on_bottle = -force (matches old working code).

# Reward components:
#   R = R_progress(gated by alignment) + R_deviation(quadratic) + R_stability +
#       R_contact + R_smoothness + R_alignment + R_position
# """

# import numpy as np


# class RewardComputer:
#     def __init__(self, config, traj_manager, contact_manager,
#                  bottle_body_id, hand_body_id, data, push_controller):
#         self.config = config
#         self.traj_manager = traj_manager
#         self.contact_manager = contact_manager
#         self.bottle_body_id = bottle_body_id
#         self.hand_body_id = hand_body_id
#         self.data = data
#         self.push_controller = push_controller
#         self.prev_progress = 0.0
#         # self.prev_force = np.zeros(3)

#     def reset(self):
#         self.prev_progress = 0.0
#         # self.prev_force = np.zeros(3)

#     def compute_reward(self):
#         info = {"is_success": False}

#         bottle_pos = self.data.xpos[self.bottle_body_id]
#         bottle_xy = bottle_pos[:2]
#         hand_pos = self.data.xpos[self.hand_body_id]
#         hand_xy = hand_pos[:2]

#         deviation_vec, deviation_mag = self.traj_manager.get_path_deviation(bottle_xy)
#         progress = self.traj_manager.get_progress(bottle_xy)
#         tilt = self.data.xmat[self.bottle_body_id].reshape(3, 3)[2, 2]
#         is_touching = self.contact_manager.is_touching()

#         # Raw measured contact force
#         force = self.contact_manager.get_contact_force()
#         force_mag = np.linalg.norm(force)

#         # Sign convention: MuJoCo gives bottle-on-robot reaction.
#         # Force the robot APPLIES to the bottle is the negative.
#         f_robot_on_bottle = force # was -force

#         dist_to_bottle = np.linalg.norm(hand_xy - bottle_xy)

#         # Push direction (blended tangent + correction toward lookahead)
#         push_dir_2d, _, _ = self.traj_manager.get_push_direction(bottle_xy)
#         push_dir_3d = np.array([push_dir_2d[0], push_dir_2d[1], 0.0])

#         # ============================================================
#         # ALIGNMENT FACTOR (used to gate progress)
#         # ============================================================
#         f_rb_mag = np.linalg.norm(f_robot_on_bottle)
#         if f_rb_mag > 0.1:
#             f_rb_dir = f_robot_on_bottle / f_rb_mag
#             alignment = max(float(np.dot(f_rb_dir, push_dir_3d)), 0.0)
#         else:
#             alignment = 0.0

#         # ============================================================
#         # 1. PROGRESS — gated by alignment
#         # ============================================================
#         progress_delta = progress - self.prev_progress
#         r_progress = self.config.w_progress * max(progress_delta, 0) # * alignment

#         # ============================================================
#         # 2. DEVIATION — QUADRATIC
#         # ============================================================
#         r_deviation = -self.config.w_deviation * deviation_mag ** 2
#         r_deviation = max(r_deviation, -3.0)

#         # ============================================================
#         # 3. STABILITY
#         # ============================================================
#         if tilt > 0.90: # was 0.97 - more flexibility
#             r_stability = 0.0
#         else:
#             r_stability = -5.0 * (1.0 - tilt) # Reduced penalty multiplier

#         # ============================================================
#         # 4. CONTACT
#         # ============================================================
#         if is_touching:
#             r_contact = 0.0
#         else:
#             dist_penalty = self.config.w_contact_loss * dist_to_bottle
#             r_contact = max(dist_penalty, -5.0) # Cap the bleeding
            
#             if dist_to_bottle > 0.08:
#                 r_contact -= 2.0

#         # ============================================================
#         # 5. ALIGNMENT (Force in the direction of the path)
#         # ============================================================
#         f_along_tangent = float(np.dot(f_robot_on_bottle, push_dir_3d))
        
#         # Cap the rewardable force at 5.0N so the agent doesn't smash it for points
#         f_along_tangent_clipped = float(np.clip(f_along_tangent, 0.0, 5.0))
#         if progress_delta > 0.0001:
#             r_alignment = self.config.w_alignment * f_along_tangent_clipped
#         else:
#             r_alignment = 0.0

#         # check tangents also in the world frame. (to be sure double check)

#         # ============================================================
#         # 6. POSITION — CHANGED TO A PENALTY!
#         # Max reward is 0.0 (perfectly aligned). Penalizes being skewed.
#         # ============================================================
#         # bottle_to_flange = hand_xy - bottle_xy
#         # b2f_norm = np.linalg.norm(bottle_to_flange)
#         # if b2f_norm > 1e-6:
#         #     b2f_unit = bottle_to_flange / b2f_norm
#         #     push_dir_2d_unit = push_dir_2d / (np.linalg.norm(push_dir_2d) + 1e-6)
#         #     side_dot = float(np.dot(b2f_unit, push_dir_2d_unit))
#         #     position_alignment = max(-side_dot, 0.0)  # 1 when perfectly behind
#         # else:
#         #     side_dot = 0.0
#         #     position_alignment = 0.0
            
#         # # # FIX: Subtract 1.0 so the best score is 0, and worse alignments are negative
#         # # r_position = self.config.w_position * (position_alignment - 1.0)

#         # # ============================================================
#         # # 7. POSITION — DYNAMIC CROSS-TRACK TARGETING
#         # # ============================================================
#         # # 1. We know the exact direction we want the bottle to go
#         # push_dir_2d_unit = push_dir_2d / (np.linalg.norm(push_dir_2d) + 1e-6)
        
#         # # 2. Calculate the exact IDEAL hand position (opposite the push direction)
#         # # Assuming bottle radius is roughly 3.5cm (0.035m)
#         # bottle_radius = 0.039
#         # ideal_hand_xy = bottle_xy - (push_dir_2d_unit * bottle_radius)
        
#         # # 3. Calculate how far the actual hand is from this ideal spot
#         # hand_position_error = np.linalg.norm(hand_xy - ideal_hand_xy)

#         # # Give the agent a 2cm "safe zone" behind the bottle where there is ZERO penalty.
#         # # This stops the agent from micro-panicking.
#         # error_outside_safezone = max(hand_position_error - 0.02, 0.0)
        
#         # # 4. Penalize the agent for being far from the ideal spot
#         # # Subtracting an offset so it only penalizes if it's way off center
#         # # r_position = -self.config.w_position * max(hand_position_error - 0.02, 0.0)

#         # # Use a QUADRATIC penalty instead of linear. 
#         # # Being 1mm outside the zone is a tiny penalty. Being 5cm outside is a massive penalty.
#         # r_position = -self.config.w_position * (error_outside_safezone ** 2)

#         # ============================================================
#         # 8. ORIENTATION ALIGNMENT (Wrist Yaw)
#         # Penalize the agent if the pushing face is not aligned with the path
#         # ============================================================
#         # flange_axis_3d = self.push_controller.get_flange_axis_world()
#         # flange_axis_2d = flange_axis_3d[:2]
        
#         # if np.linalg.norm(flange_axis_2d) > 1e-6:
#         #     flange_axis_2d = flange_axis_2d / np.linalg.norm(flange_axis_2d)
#         #     # 1.0 = perfectly aligned, 0.0 = 90 degrees off
#         #     cos_flange_align = float(np.dot(flange_axis_2d, push_dir_2d_unit))
#         # else:
#         #     cos_flange_align = 0.0

#         # # Quadratic penalty so small deviations are fine, but large ones hurt
#         # r_orientation = -1.0 * ((1.0 - cos_flange_align) ** 2) # was -1.0

#         # ============================================================
#         # 9. VELOCITY PENALTY
#         # Penalize the agent if the end-effector moves faster than the limit
#         # As Marko mentioned in the chat <-ask him how to do it tomorrow.
#         # Is the way you implemented is actually correct.
#         # ============================================================
#         # Fetch current end-effector velocity from the controller
#         v_current_3d = self.push_controller.get_ee_velocity()
#         v_mag = np.linalg.norm(v_current_3d[:2])  # Only care about XY speed
        
#         if v_mag > self.config.v_target_limit:
#             # Quadratic penalty: a little bit over is a small fine, 
#             # a massive slapshot is a huge fine.
#             speed_excess = v_mag - self.config.v_target_limit
#             r_velocity = -self.config.w_velocity_penalty * (speed_excess ** 2)
#         else:
#             r_velocity = 0.0
        

#         # ============================================================
#         # TOTAL + TIME PENALTY FIX
#         # ============================================================
#         total_reward = (r_progress + r_deviation + r_stability +
#                         r_contact  + r_alignment + r_velocity) # +  + r_position + r_orientation
        
#         # ACTUALLY APPLY THE TIME PENALTY
#         total_reward -= self.config.time_penalty

#         info['r_progress'] = r_progress
#         info['r_deviation'] = r_deviation
#         info['r_stability'] = r_stability
#         info['r_contact'] = r_contact
#         info['r_alignment'] = r_alignment
#         # info['r_position'] = r_position
#         # info['alignment'] = alignment
#         # info['side_dot'] = side_dot
#         info['f_along_tangent'] = f_along_tangent
#         info['r_velocity'] = r_velocity

#         # ============================================================
#         # TERMINAL CONDITIONS
#         # ============================================================
#         if progress > 0.95 and deviation_mag < self.config.path_tolerance:
#             total_reward += self.config.success_bonus
#             info["is_success"] = True

#         if tilt < 0.5:
#             total_reward += self.config.failure_penalty
#             info["bottle_fallen"] = True

#         if deviation_mag > 0.08: # was 0.15
#             total_reward += self.config.off_path_penalty
#             info["off_path"] = True

#         if dist_to_bottle > 0.30:
#             total_reward += self.config.failure_penalty
#             info["off_path"] = True

#         self.prev_progress = progress
#         # self.prev_force = force.copy()

#         info.update({
#             "progress": progress,
#             "deviation": deviation_mag,
#             "force_magnitude": force_mag,
#             "bottle_tilt": tilt,
#             "is_touching": is_touching,
#         })

#         return float(total_reward), info




#################################################################################################################################
"""
reward.py  (rebalanced)

CHANGES from previous version:
  1. Removed progress_delta gate from r_alignment to reward force intent before static friction breaks.
  2. Added "Sticky Contact" reward to encourage leaning into the bottle safely.
  3. r_deviation: QUADRATIC instead of linear, much lower cap (-3 vs -15).
  4. r_alignment, r_position weights bumped.
  5. off_path terminal penalty softened (-50 -> -10).
  6. Sign convention: f_robot_on_bottle = -force (matches old working code).
"""

import numpy as np


class RewardComputer:
    def __init__(self, config, traj_manager, contact_manager,
                 bottle_body_id, hand_body_id, data, push_controller):
        self.config = config
        self.traj_manager = traj_manager
        self.contact_manager = contact_manager
        self.bottle_body_id = bottle_body_id
        self.hand_body_id = hand_body_id
        self.data = data
        self.push_controller = push_controller
        self.prev_progress = 0.0

    def reset(self):
        self.prev_progress = 0.0

    def compute_reward(self):
        info = {"is_success": False}

        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]
        hand_pos = self.data.xpos[self.hand_body_id]
        hand_xy = hand_pos[:2]

        deviation_vec, deviation_mag = self.traj_manager.get_path_deviation(bottle_xy)
        progress = self.traj_manager.get_progress(bottle_xy)
        tilt = self.data.xmat[self.bottle_body_id].reshape(3, 3)[2, 2]
        is_touching = self.contact_manager.is_touching()

        # Raw measured contact force
        force = self.contact_manager.get_contact_force()
        force_mag = np.linalg.norm(force)

        # Sign convention: MuJoCo gives bottle-on-robot reaction.
        # Force the robot APPLIES to the bottle is the negative.
        f_robot_on_bottle = force 

        dist_to_bottle = np.linalg.norm(hand_xy - bottle_xy)

        # Push direction (blended tangent + correction toward lookahead)
        push_dir_2d, _, _ = self.traj_manager.get_push_direction(bottle_xy)
        push_dir_3d = np.array([push_dir_2d[0], push_dir_2d[1], 0.0])

        # ============================================================
        # ALIGNMENT FACTOR & TANGENT FORCE CALCULATIONS
        # ============================================================
        f_rb_mag = np.linalg.norm(f_robot_on_bottle)
        if f_rb_mag > 0.1:
            f_rb_dir = f_robot_on_bottle / f_rb_mag
            alignment = max(float(np.dot(f_rb_dir, push_dir_3d)), 0.0)
        else:
            alignment = 0.0

        # Calculate force applied strictly along the desired path
        f_along_tangent = float(np.dot(f_robot_on_bottle, push_dir_3d))
        # Cap the rewardable force at 5.0N so the agent doesn't smash it for points
        f_along_tangent_clipped = float(np.clip(f_along_tangent, 0.0, 5.0))

        # ============================================================
        # 1. PROGRESS
        # ============================================================
        progress_delta = progress - self.prev_progress
        r_progress = self.config.w_progress * max(progress_delta, 0) 

        # ============================================================
        # 2. DEVIATION — QUADRATIC
        # ============================================================
        r_deviation = -self.config.w_deviation * deviation_mag ** 2
        r_deviation = max(r_deviation, -3.0)

        # ============================================================
        # 3. STABILITY
        # ============================================================
        if tilt > 0.90: 
            r_stability = 0.0
        else:
            r_stability = -5.0 * (1.0 - tilt) 

        # ============================================================
        # 4. CONTACT (The "Sticky Contact" Fix)
        # ============================================================
        if is_touching:
            # Reward leaning into the bottle safely (max +0.5 reward per step)
            # This teaches the agent that touching the bottle is a safe haven.
            r_contact = 0.5 * (f_along_tangent_clipped / 5.0)
        else:
            dist_penalty = self.config.w_contact_loss * dist_to_bottle
            r_contact = max(dist_penalty, -5.0) # Cap the bleeding
            
            if dist_to_bottle > 0.08:
                r_contact -= 2.0

        # ============================================================
        # 5. ALIGNMENT (Force in the direction of the path)
        # ============================================================
        # Removed progress_delta gate. Agent is now rewarded for pushing 
        # correctly even before static friction breaks.
        r_alignment = self.config.w_alignment * f_along_tangent_clipped

        # ============================================================
        # 6. VELOCITY PENALTY
        # ============================================================
        v_current_3d = self.push_controller.get_ee_velocity()
        v_mag = np.linalg.norm(v_current_3d[:2])  # Only care about XY speed
        
        if v_mag > self.config.v_target_limit:
            speed_excess = v_mag - self.config.v_target_limit
            r_velocity = -self.config.w_velocity_penalty * (speed_excess ** 2)
        else:
            r_velocity = 0.0
        
        # ============================================================
        # TOTAL + TIME PENALTY FIX
        # ============================================================
        total_reward = (r_progress + r_deviation + r_stability +
                        r_contact  + r_alignment + r_velocity)
        
        # ACTUALLY APPLY THE TIME PENALTY
        total_reward -= self.config.time_penalty

        info['r_progress'] = r_progress
        info['r_deviation'] = r_deviation
        info['r_stability'] = r_stability
        info['r_contact'] = r_contact
        info['r_alignment'] = r_alignment
        info['f_along_tangent'] = f_along_tangent
        info['r_velocity'] = r_velocity

        # ============================================================
        # TERMINAL CONDITIONS
        # ============================================================
        if progress > 0.95 and deviation_mag < self.config.path_tolerance:
            total_reward += self.config.success_bonus
            info["is_success"] = True

        if tilt < 0.5:
            total_reward += self.config.failure_penalty
            info["bottle_fallen"] = True

        if deviation_mag > 0.08: 
            total_reward += self.config.off_path_penalty
            info["off_path"] = True

        if dist_to_bottle > 0.30:
            total_reward += self.config.failure_penalty
            info["off_path"] = True

        self.prev_progress = progress

        info.update({
            "progress": progress,
            "deviation": deviation_mag,
            "force_magnitude": force_mag,
            "bottle_tilt": tilt,
            "is_touching": is_touching,
        })

        return float(total_reward), info