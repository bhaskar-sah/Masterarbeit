"""
reward.py  (rebalanced)

CHANGES from previous version (addressing "do nothing" local optimum):

  1. r_deviation: QUADRATIC instead of linear, much lower cap (-3 vs -15),
     and lower weight. Small deviations now nearly free; only catastrophic
     deviations hurt. This lets the agent tolerate the messy contact
     physics during early exploration.

  2. r_alignment, r_position weights bumped (0.3 -> 0.5, 0.5 -> 1.0).
     Doing the right thing now exceeds typical penalties.

  3. off_path terminal penalty softened (-50 -> -10) so failed exploration
     doesn't dominate gradient updates.

  4. Sign convention: f_robot_on_bottle = -force (matches old working code).

Reward components (Bhaskar's contributions):
  R = R_progress(gated by alignment) + R_deviation(quadratic) + R_stability +
      R_contact + R_smoothness + R_alignment + R_position
"""

import numpy as np


class RewardComputer:
    def __init__(self, config, traj_manager, contact_manager,
                 bottle_body_id, hand_body_id, data):
        self.config = config
        self.traj_manager = traj_manager
        self.contact_manager = contact_manager
        self.bottle_body_id = bottle_body_id
        self.hand_body_id = hand_body_id
        self.data = data
        self.prev_progress = 0.0
        # self.prev_force = np.zeros(3)

    def reset(self):
        self.prev_progress = 0.0
        # self.prev_force = np.zeros(3)

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
        f_robot_on_bottle = -force

        dist_to_bottle = np.linalg.norm(hand_xy - bottle_xy)

        # Push direction (blended tangent + correction toward lookahead)
        push_dir_2d, _, _ = self.traj_manager.get_push_direction(bottle_xy)
        push_dir_3d = np.array([push_dir_2d[0], push_dir_2d[1], 0.0])

        # ============================================================
        # ALIGNMENT FACTOR (used to gate progress)
        # ============================================================
        f_rb_mag = np.linalg.norm(f_robot_on_bottle)
        if f_rb_mag > 0.1:
            f_rb_dir = f_robot_on_bottle / f_rb_mag
            alignment = max(float(np.dot(f_rb_dir, push_dir_3d)), 0.0)
        else:
            alignment = 0.0

        # ============================================================
        # 1. PROGRESS — gated by alignment
        # ============================================================
        progress_delta = progress - self.prev_progress
        r_progress = self.config.w_progress * max(progress_delta, 0) * alignment

        # ============================================================
        # 2. DEVIATION — QUADRATIC
        # ============================================================
        r_deviation = -self.config.w_deviation * deviation_mag ** 2
        r_deviation = max(r_deviation, -3.0)

        # ============================================================
        # 3. STABILITY
        # ============================================================
        if tilt > 0.97:
            r_stability = 0.0
        else:
            r_stability = -15.0 * (1.0 - tilt)

        # ============================================================
        # 4. CONTACT
        # ============================================================
        if is_touching:
            r_contact = 0.0
        else:
            dist_penalty = self.config.w_contact_loss * dist_to_bottle
            r_contact = max(dist_penalty, -5.0) # Cap the bleeding
            
            if dist_to_bottle > 0.08:
                r_contact -= 2.0

        # ============================================================
        # 5. ALIGNMENT (Force in the direction of the path)
        # ============================================================
        f_along_tangent = float(np.dot(f_robot_on_bottle, push_dir_3d))
        
        # Cap the rewardable force at 5.0N so the agent doesn't smash it for points
        f_along_tangent_clipped = float(np.clip(f_along_tangent, 0.0, 5.0))
        r_alignment = self.config.w_alignment * f_along_tangent_clipped

        # ============================================================
        # 6. POSITION — CHANGED TO A PENALTY!
        # Max reward is 0.0 (perfectly aligned). Penalizes being skewed.
        # ============================================================
        bottle_to_flange = hand_xy - bottle_xy
        b2f_norm = np.linalg.norm(bottle_to_flange)
        if b2f_norm > 1e-6:
            b2f_unit = bottle_to_flange / b2f_norm
            push_dir_2d_unit = push_dir_2d / (np.linalg.norm(push_dir_2d) + 1e-6)
            side_dot = float(np.dot(b2f_unit, push_dir_2d_unit))
            position_alignment = max(-side_dot, 0.0)  # 1 when perfectly behind
        else:
            side_dot = 0.0
            position_alignment = 0.0
            
        # FIX: Subtract 1.0 so the best score is 0, and worse alignments are negative
        r_position = self.config.w_position * (position_alignment - 1.0)

        # ============================================================
        # TOTAL + TIME PENALTY FIX
        # ============================================================
        total_reward = (r_progress + r_deviation + r_stability +
                        r_contact + r_alignment + r_position)
        
        # ACTUALLY APPLY THE TIME PENALTY
        total_reward -= self.config.time_penalty

        info['r_progress'] = r_progress
        info['r_deviation'] = r_deviation
        info['r_stability'] = r_stability
        info['r_contact'] = r_contact
        info['r_alignment'] = r_alignment
        info['r_position'] = r_position
        info['alignment'] = alignment
        info['side_dot'] = side_dot
        info['f_along_tangent'] = f_along_tangent

        # ============================================================
        # TERMINAL CONDITIONS
        # ============================================================
        if progress > 0.95 and deviation_mag < self.config.path_tolerance:
            total_reward += self.config.success_bonus
            info["is_success"] = True

        if tilt < 0.5:
            total_reward += self.config.failure_penalty
            info["bottle_fallen"] = True

        if deviation_mag > 0.15:
            total_reward += self.config.off_path_penalty
            info["off_path"] = True

        if dist_to_bottle > 0.30:
            total_reward += self.config.failure_penalty
            info["off_path"] = True

        self.prev_progress = progress
        # self.prev_force = force.copy()

        info.update({
            "progress": progress,
            "deviation": deviation_mag,
            "force_magnitude": force_mag,
            "bottle_tilt": tilt,
            "is_touching": is_touching,
        })

        return float(total_reward), info