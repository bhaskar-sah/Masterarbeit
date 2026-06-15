# """
# reward.py
# """
import numpy as np


class RewardComputer:
    def __init__(self, config, traj_manager, contact_manager,
                 bottle_body_id, gripper_site_id, data, push_controller):
        self.config = config
        self.traj_manager = traj_manager
        self.contact_manager = contact_manager
        self.bottle_body_id = bottle_body_id
        self.gripper_site_id = gripper_site_id
        self.data = data
        self.push_controller = push_controller
        self.prev_progress = 0.0

    def reset(self):
        self.prev_progress = 0.0

    def compute_reward(self, action):
        info = {"is_success": False}

        bottle_pos = self.data.xpos[self.bottle_body_id]
        bottle_xy = bottle_pos[:2]

        # ee_pos = self.push_controller.get_ee_position()
        ee_pos_world = self.data.site_xpos[self.gripper_site_id].copy()
        ee_xy = ee_pos_world[:2]

        deviation_vec, deviation_mag = self.traj_manager.get_path_deviation(bottle_xy)
        progress = self.traj_manager.get_progress(bottle_xy)
        z_cap = self.data.xmat[self.bottle_body_id].reshape(3,3)[:,2]
        n_hat = np.array([0.0, 0.0, 1.0])
        tilt = float(np.dot(z_cap, n_hat))

        is_touching = self.contact_manager.is_touching()

        # Raw measured contact force
        force = self.contact_manager.get_contact_force()
        force_mag = np.linalg.norm(force)

        # Sign convention: MuJoCo gives bottle-on-robot reaction.
        f_robot_on_bottle = force
        f_robot_on_bottle[2] = 0.0 # TODO: not sure if z-comp should just be removed but otherwise alignment with push_dir which is only calculated in 2D is bad.

        dist_to_bottle = np.linalg.norm(ee_xy - bottle_xy)

        # Push direction (blended tangent + correction toward lookahead)
        push_dir_2d, _, _ = self.traj_manager.get_push_direction(bottle_xy)
        push_dir_3d = np.array([push_dir_2d[0], push_dir_2d[1], 0.0])

        # ============================================================
        # ALIGNMENT FACTOR & TANGENT FORCE CALCULATIONS
        # ============================================================
        f_rb_mag = np.linalg.norm(f_robot_on_bottle)
        
        # Pure Normalized Alignment Score [-1 to 1]
        if f_rb_mag > 1e-6:
            f_rb_dir = f_robot_on_bottle / f_rb_mag
            alignment_score = float(np.dot(f_rb_dir, push_dir_3d))
        else:
            alignment_score = 0.0

        # Raw Force Projection (Still needed ONLY for the sticky contact logic)
        f_along_tangent = float(np.dot(f_robot_on_bottle, push_dir_3d))

        # ============================================================
        # 1. PROGRESS
        # ============================================================
        progress_delta = progress - self.prev_progress
        r_progress = self.config.w_progress * max(progress_delta, 0) 

        # ============================================================
        # 2. DEVIATION — QUADRATIC
        # ============================================================
        r_deviation = -self.config.w_deviation * deviation_mag ** 2

        # ============================================================
        # 3. STABILITY
        # ============================================================
        r_stability = - self.config.w_stability * (1.0 - tilt)

        # ============================================================
        # 4. CONTACT (The "Sticky Contact" Fix)
        # ============================================================
        if is_touching:
            r_contact = self.config.w_in_contact
        else: 
            r_contact = self.config.w_contact_loss * dist_to_bottle
            
        if dist_to_bottle > 0.08:
            r_contact -= 2.0

        # ============================================================
        # 5. ALIGNMENT (Pure Direction, No Magnitude)
        # ============================================================
        r_alignment = self.config.w_alignment * alignment_score

        # ============================================================
        # 6. VELOCITY PENALTY
        # ============================================================
        v_current_3d = (self.push_controller.get_ee_vel_world())
        v_mag = np.linalg.norm(v_current_3d)
        
        if v_mag > self.config.v_target_limit: #penalise fast movement
            speed_excess = v_mag - self.config.v_target_limit
            r_velocity = -self.config.w_velocity_penalty * (speed_excess ** 2)
        elif v_mag < self.config.v_minimum: # penalise no movement
            r_velocity = -0.2
        else:
            r_velocity = 0.0

        # ============================================================
        # TOTAL + TIME PENALTY FIX
        # ============================================================
        total_reward = (r_progress + 
                        r_deviation + 
                        r_stability +
                        r_contact  + 
                        r_alignment + 
                        r_velocity)
                        
        # TIME PENALTY
        total_reward -= self.config.time_penalty

        info['r_progress'] = r_progress
        info['r_deviation'] = r_deviation
        info['r_stability'] = r_stability
        info['r_contact'] = r_contact
        info['r_alignment'] = r_alignment
        info['f_along_tangent'] = f_along_tangent
        info['r_velocity'] = r_velocity
        info['total_reward'] = total_reward
        info['time_penalty'] = -self.config.time_penalty

        # ============================================================
        # TERMINAL CONDITIONS
        # ============================================================
        if progress > 0.95 and deviation_mag < self.config.path_tolerance:
            total_reward += self.config.success_bonus
            info["is_success"] = True

        if tilt < 0.5:
            total_reward += self.config.failure_penalty
            info["bottle_fallen"] = True

        if deviation_mag > 0.05: 
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