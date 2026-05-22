"""
push_controller.py

CHANGES from previous version: (PURE MOTION CONTROL — Marko's correction)

Old equation (mixed motion/force, conflicting):
    F_cmd = Kp(p_des - p) + Kd(v_des - v) + F_des + Kf(F_des - F_meas)
 
New equation (pure motion control):
    F_cmd = Kp(p_des - p) + Kd(v_des - v)

    ACTION = [vx, vy, wz] (3D action)  all in [-1, 1]. No force action.
    Force on the bottle EMERGES naturally:
    - When the gripper pushes against the bottle, position error grows in the push direction (because bottle resists)
    - PD term outputs Kp * p_error in that direction
    - That IS the contact force

  1. p_error magnitude is now CAPPED at P_ERROR_MAX (3cm). Without this,
     when the agent commands sustained max velocity, p_des races ahead of
     the actual EE position and the position-control term (Kp * p_error)
     can grow to 10+ N — drowning out the agent's commanded F_des.
     Capping p_error gives the agent meaningful authority over force.

  2. yaw_des is WRAPPED to [-pi, pi] each step. Without this, sustained
     wrist commands cause yaw_des to grow unboundedly across an episode,
     even though the joint can only physically rotate to +-2.9 rad. The
     wrap doesn't change R_desired (cosine and sine are periodic) but
     keeps the state bounded.

  3. F_robot_on_bottle = -F_measured_raw used for force feedback
     (sign convention matches reward.py and old working code).

The agent learns velocity profiles; force profiles emerge from interaction.
"""

import numpy as np
import mujoco


# ============================================================
# Set from visualize_flange.py output
# ============================================================
FLANGE_PUSH_AXIS_LOCAL = 0   # 0=x, 1=y, 2=z
FLANGE_PUSH_SIGN = -1         # Positive X-axis pushes the bottle # +1 (positive x axis) or -1 (negative x-axis)


class PushController:
    def __init__(self, model, data, traj_manager, contact_manager,
                 hand_body_id, bottle_body_id, config):
        self.model = model
        self.data = data
        self.traj_manager = traj_manager
        self.contact_manager = contact_manager
        self.hand_body_id = hand_body_id
        self.bottle_body_id = bottle_body_id
        self.config = config

        self.gripper_site_id = model.site("gripper_center").id

        # Gains
        self.Kp = config.Kp
        self.Kd = config.Kd
        # self.Kf = config.Kf
        self.Kp_rot = config.Kp_rot
        self.Kd_rot = config.Kd_rot

        # Limits
        self.v_max = config.v_max
        self.w_max = config.w_max
        # self.f_max = config.f_max
        # self.F_FLOOR = config.F_FLOOR
        self.dt = config.dt
        self.control_dt = config.control_dt

        self.n_joints = 7
        self.qvel_start = 6  # set to 0 for no-bottle test, 6 with bottle
        self.logger = None

        # Debug
        self.last_F_cmd = np.zeros(3)
        self.last_push_dir = np.zeros(2)

    def reset(self):
        pass  # no state to reset — pure velocity control has no integrators

    # ---------- kinematics (SITE-BASED) ----------
    def get_ee_position(self):
        return self.data.site_xpos[self.gripper_site_id].copy()

    def get_ee_orientation(self):
        return self.data.site_xmat[self.gripper_site_id].reshape(3, 3).copy()
    
    # transform the base frame to the ee frame. start with the velocity. 

    def get_ee_velocity(self):
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.gripper_site_id)
        return jacp @ self.data.qvel

    def get_ee_angular_velocity(self):
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, None, jacr, self.gripper_site_id)
        return jacr @ self.data.qvel

    def get_jacobian_full(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.gripper_site_id)
        J_full = np.vstack([jacp, jacr])
        return J_full[:, self.qvel_start:self.qvel_start + self.n_joints]

    def get_gravity_compensation(self):
        return self.data.qfrc_bias[self.qvel_start:self.qvel_start + self.n_joints].copy()

    def get_bottle_tilt(self):
        return float(self.data.xmat[self.bottle_body_id].reshape(3, 3)[2, 2])

    def get_flange_axis_world(self):
        R_current = self.get_ee_orientation()
        axis = R_current[:, FLANGE_PUSH_AXIS_LOCAL] * FLANGE_PUSH_SIGN
        axis[2] = 0.0
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            axis /= norm
        return axis

    def compute_orientation_error(self, R_current, R_desired):
        R_error = R_desired @ R_current.T
        trace = np.clip(np.trace(R_error), -1.0, 3.0)
        angle = np.arccos((trace - 1) / 2)
        if angle < 1e-6:
            return np.zeros(3)
        elif angle > np.pi - 1e-6:
            axis = np.sqrt((np.diag(R_error) + 1) / 2)
            return axis * angle
        else:
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1],
            ]) / (2 * np.sin(angle))
            return axis * angle

    # def compute_torque(self, action, tilt=None):
    #     # =======================================================================================
    #     # Action: [vx, vy, wz] - Strictly in the EE frame (pure velocity control)
    #     # =======================================================================================
    #     wz_cmd = action[2] * self.w_max

    #     # =======================================================================================
    #     # 1. CURRENT STATE (Base Frame)
    #     # =======================================================================================
    #     v_current_base = self.get_ee_velocity()
    #     omega_current_base = self.get_ee_angular_velocity()
    #     R_curr = self.get_ee_orientation()  # 3x3 rotation matrix from EE to Base

    #     # =======================================================================================
    #     # 2. DESIRED VELOCITY (EE Frame)
    #     # action[0] (vx) = push forward out of the flange
    #     # action[1] (vy) = slide laterally left/right
    #     # =======================================================================================
    #     v_des_ee = np.array([
    #         action[0] * self.v_max * FLANGE_PUSH_SIGN,
    #         action[1] * self.v_max,
    #         0.0
    #     ])

    #     # =======================================================================================
    #     # 3. FRAME CONSISTENCY
    #     # Convert base velocity to EE frame to compute error in local coordinates
    #     # =======================================================================================
    #     v_current_ee = R_curr.T @ v_current_base

    #     # =======================================================================================
    #     # 4. FORCE COMMAND
    #     # XY: Pure velocity damping in the EE frame, as requested by supervisor
    #     # Z:  Position hold (controller-managed in base frame to prevent dropping)
    #     # =======================================================================================
    #     F_cmd_ee = self.Kd * (v_des_ee - v_current_ee)
        
    #     # Rotate the locally-computed force command back to the Base frame
    #     F_cmd_base = R_curr @ F_cmd_ee

    #     # Apply Z-axis position control directly in the base frame since Z is universally "up"
    #     p_current = self.get_ee_position()
    #     F_cmd_base[2] = self.Kp * (0.84 - p_current[2]) - self.Kd * v_current_base[2]

    #     # Clip force
    #     F_cmd_mag = np.linalg.norm(F_cmd_base)
    #     if F_cmd_mag > 100.0:
    #         F_cmd_base = F_cmd_base / F_cmd_mag * 100.0

    #     # =======================================================================================
    #     # 5. ORIENTATION COMMAND (Pure angular velocity damping)
    #     # =======================================================================================
    #     omega_des_ee = np.array([0.0, 0.0, wz_cmd])
    #     omega_des_base = R_curr @ omega_des_ee  # Convert desired angular velocity to base frame

    #     tau_rot_cmd = self.Kd_rot * (omega_des_base - omega_current_base)
    #     tau_rot_mag = np.linalg.norm(tau_rot_cmd)
    #     if tau_rot_mag > 10.0:
    #         tau_rot_cmd = tau_rot_cmd / tau_rot_mag * 10.0

    #     # =======================================================================================
    #     # 6. JOINT TORQUES
    #     # =======================================================================================
    #     wrench_cmd = np.concatenate([F_cmd_base, tau_rot_cmd])
    #     J_full = self.get_jacobian_full()
    #     tau_task = J_full.T @ wrench_cmd
    #     tau = tau_task + self.get_gravity_compensation()

    #     tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
    #     tau = np.clip(tau, -tau_max, tau_max)

    #     # =======================================================================================
    #     # Debug storage
    #     # =======================================================================================
    #     self.last_F_cmd = F_cmd_base.copy()
    #     bottle_xy = self.data.xpos[self.bottle_body_id][:2]
    #     push_dir_2d, _, _ = self.traj_manager.get_push_direction(bottle_xy)
    #     self.last_push_dir = push_dir_2d.copy()

    #     return tau.astype(np.float32)

    def compute_torque(self, action, tilt=None):
        # ====================================================================
        # Action: [a0, a1, wz] - meaning depends on action_mode
        #   velocity mode: a0=vx, a1=vy (in EE frame)
        #   force mode:    a0=Fx, a1=Fy (in EE frame)
        # ====================================================================
        wz_cmd = action[2] * self.w_max

        # 1. CURRENT STATE (Base Frame)
        v_current_base = self.get_ee_velocity()
        omega_current_base = self.get_ee_angular_velocity()
        R_curr = self.get_ee_orientation()

        # 2. + 3. + 4. COMPUTE F_cmd_ee BASED ON ACTION MODE
        if self.config.action_mode == "velocity":
            # Existing impedance behavior
            v_des_ee = np.array([
                action[0] * self.v_max * FLANGE_PUSH_SIGN,
                action[1] * self.v_max,
                0.0
            ])
            v_current_ee = R_curr.T @ v_current_base
            F_cmd_ee = self.Kd * (v_des_ee - v_current_ee)
        
        elif self.config.action_mode == "force":
            # Direct force command — no velocity middleman
            F_cmd_ee = np.array([
                action[0] * self.config.f_max * FLANGE_PUSH_SIGN,
                action[1] * self.config.f_max,
                0.0
            ])
            # SAFETY: cap force if EE is already moving too fast
            # (prevents runaway if contact is suddenly lost)
            v_current_ee = R_curr.T @ v_current_base
            v_mag = np.linalg.norm(v_current_ee[:2])
            v_safe = self.config.v_max * 1.5   # 50% above nominal v_max = safety threshold
            if v_mag > v_safe:
                scale = v_safe / v_mag
                F_cmd_ee[:2] *= scale
        
        else:
            raise ValueError(f"Unknown action_mode: {self.config.action_mode}")

        # Rotate to base frame
        F_cmd_base = R_curr @ F_cmd_ee

        # 4b. Z-axis position hold (unchanged for both modes)
        p_current = self.get_ee_position()
        F_cmd_base[2] = self.Kp * (0.84 - p_current[2]) - self.Kd * v_current_base[2]

        # Clip total force magnitude
        F_cmd_mag = np.linalg.norm(F_cmd_base)
        if F_cmd_mag > 100.0:
            F_cmd_base = F_cmd_base / F_cmd_mag * 100.0

        # 5. ORIENTATION COMMAND (unchanged)
        omega_des_ee = np.array([0.0, 0.0, wz_cmd])
        omega_des_base = R_curr @ omega_des_ee
        tau_rot_cmd = self.Kd_rot * (omega_des_base - omega_current_base)
        tau_rot_mag = np.linalg.norm(tau_rot_cmd)
        if tau_rot_mag > 10.0:
            tau_rot_cmd = tau_rot_cmd / tau_rot_mag * 10.0

        # 6. JOINT TORQUES (unchanged)
        wrench_cmd = np.concatenate([F_cmd_base, tau_rot_cmd])
        J_full = self.get_jacobian_full()
        tau_task = J_full.T @ wrench_cmd
        tau = tau_task + self.get_gravity_compensation()

        tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
        tau = np.clip(tau, -tau_max, tau_max)

        # Debug storage (unchanged)
        self.last_F_cmd = F_cmd_base.copy()
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        push_dir_2d, _, _ = self.traj_manager.get_push_direction(bottle_xy)
        self.last_push_dir = push_dir_2d.copy()

        return tau.astype(np.float32)