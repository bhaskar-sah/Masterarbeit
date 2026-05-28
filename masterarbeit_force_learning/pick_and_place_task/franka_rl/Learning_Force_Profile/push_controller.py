"""
push_controller.py

CHANGES from previous version: (PURE FORCE CONTROL)
"""

import numpy as np
import mujoco


# ============================================================
# Set from visualize_flange.py output
# ============================================================
# FLANGE_PUSH_AXIS_LOCAL = 0   # 0=x, 1=y, 2=z
# FLANGE_PUSH_SIGN = -1         # Positive X-axis pushes the bottle # +1 (positive x axis) or -1 (negative x-axis)


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

        # self.Kp = config.Kp
        # self.Kd = config.Kd
        # # self.Kf = config.Kf
        # self.Kp_rot = config.Kp_rot

        self.Kd_rot = config.Kd_rot

        # Limits

        # self.v_max = config.v_max
        
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
        pass  # no state to reset — pure force control has no integrators

    # ---------- kinematics (SITE-BASED) ----------
    def get_ee_position(self):
        return self.data.site_xpos[self.gripper_site_id].copy()

    def get_ee_orientation(self):
        return self.data.site_xmat[self.gripper_site_id].reshape(3, 3).copy()
    
    def get_ee_velocity(self):
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.gripper_site_id)
        return jacp @ self.data.qvel

    def get_ee_angular_velocity(self):
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, None, jacr, self.gripper_site_id)
        return jacr @ self.data.qvel

    def get_jacobian_full(self): # The  Link: https://mujoco.readthedocs.io/en/3.2.0/APIreference/APIfunctions.html#mj-jacsite (see mj_jac)
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
        # get local X-Axis of the flange
        axis = R_current[:, 0]
        axis[2] = 0.0
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            axis /= norm
        return axis

    def compute_torque(self, action, tilt=None):
        # ====================================================================
        # 1. READ ACTIONS
        # Action: [Fx, Fy, Wz] (Normalized -1 to 1)
        # Action: [a0, a1, wz] - no differen modes (JUST force - as MARKO suggested)
        #         a0=Fx, a1=Fy (in EE frame)
        # ====================================================================
        # wz_cmd = action[2] * self.w_max

        # ====================================================================
        # 2. PURE PLANAR FORCE IN END-EFFECTOR FRAME
        # No negative sign, No Z-axis forces.
        # use ee local body frame
        # ====================================================================
        F_cmd_ee = np.array([
            action[0] * self.config.f_max, 
            action[1] * self.config.f_max, 
            0.0  # Z-force is strictly 0
        ])

        tau_rot_cmd = np.array([
            0.0,
            0.0,
            action[2] * self.config.tau_rot_max
        ])

        # ====================================================================
        # 3. ROTATE FORCE TO BASE FRAME (Required for MuJoCo Jacobian) - mj_jac is converts in base(/space/word)
        # ====================================================================
        R_curr = self.get_ee_orientation()
        F_cmd_base = R_curr @ F_cmd_ee
        tau_rot_cmd_base = R_curr @ tau_rot_cmd

        # Hold the exact Z-height (0.84m) using impedance control
        current_z = self.get_ee_position()[2]
        current_vz = self.get_ee_velocity()[2]

        target_z = 0.84
        kp_z = 1000.0
        kd_z = 50.0

        # Override the floating Z-command with the holding force
        F_cmd_base[2] = kp_z * (target_z - current_z) - kd_z * current_vz

        # # ====================================================================
        # # 4. ORIENTATION DAMPING (Keep wrist stable)
        # # ====================================================================
        # omega_current_base = self.get_ee_angular_velocity()
        # omega_des_ee = np.array([0.0, 0.0, wz_cmd])
        # omega_des_base = R_curr @ omega_des_ee
        
        # tau_rot_cmd = self.Kd_rot * (omega_des_base - omega_current_base)

        # ====================================================================
        # 5. ASSEMBLE WRENCH AND COMPUTE JOINT TORQUES
        # ====================================================================
        wrench_cmd = np.concatenate([F_cmd_base, tau_rot_cmd_base])
        
        J_full = self.get_jacobian_full()
        tau_task = J_full.T @ wrench_cmd
        tau = tau_task + self.get_gravity_compensation()

        # Clip to hardware limits
        tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
        tau = np.clip(tau, -tau_max, tau_max)

        # Debug storage
        self.last_F_cmd = F_cmd_base.copy()

        return tau.astype(np.float32)