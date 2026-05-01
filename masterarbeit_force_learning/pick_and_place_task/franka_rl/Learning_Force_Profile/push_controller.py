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
FLANGE_PUSH_SIGN = -1        # +1 or -1


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

        self.p_des = None
        self.yaw_des = None
        self.n_joints = 7
        self.qvel_start = 6 # set to 0 for no-bottle test, 6 with bottle
        self.logger = None

        # Debug
        self.last_F_cmd = np.zeros(3)
        # self.last_F_des = np.zeros(3)
        # self.last_f_magnitude = 0.0
        self.last_push_dir = np.zeros(2)
        self.last_yaw_des = 0.0
        self.last_flange_axis_world = np.zeros(3)

    def reset(self, initial_ee_pos=None):
        if initial_ee_pos is not None:
            self.p_des = initial_ee_pos.copy()
        else:
            self.p_des = self.get_ee_position().copy()
        R_initial = self.get_ee_orientation()
        self.yaw_des = float(np.arctan2(R_initial[1, 0], R_initial[0, 0]))

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

    def compute_torque(self, action, tilt=None):
        # =======================================================================================
        # Action is now 3D: [vx, vy, wz]
        # =======================================================================================
        v_des = np.array([action[0] * self.v_max,
                          action[1] * self.v_max,
                          0.0])
        wz_cmd = action[2] * self.w_max
 
        # =======================================================================================
        # 2. INTEGRATE v -> p_des
        # =======================================================================================
        if self.p_des is None:
            self.p_des = self.get_ee_position()
        self.p_des = self.p_des + v_des * self.control_dt
        self.p_des[0] = np.clip(self.p_des[0], 0.0, 0.7)
        self.p_des[1] = np.clip(self.p_des[1], -0.6, 0.6)
        self.p_des[2] = 0.84  # locked z height
 
        # =======================================================================================
        # 3. INTEGRATE wz -> yaw_des
        # =======================================================================================
        self.yaw_des += wz_cmd * self.control_dt
        self.yaw_des = (self.yaw_des + np.pi) % (2 * np.pi) - np.pi
 
        c, s = np.cos(self.yaw_des), np.sin(self.yaw_des)
        R_desired = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, -1.0],
        ])
 
        # =======================================================================================
        # 4. CURRENT STATE
        # =======================================================================================
        p_current = self.get_ee_position()
        v_current = self.get_ee_velocity()
        omega_current = self.get_ee_angular_velocity()
        R_current = self.get_ee_orientation()
 
        # =======================================================================================
        # 5. POSITION CONTROL ONLY (no F_des, no force feedback)
        # =======================================================================================
        p_error = self.p_des - p_current
 
        # Cap XY error magnitude (Z stays raw)
        xy_error = p_error[:2]
        xy_error_mag = np.linalg.norm(xy_error)
        limit = self.config.p_error_max
        if xy_error_mag > limit:
            xy_error = xy_error * (limit / xy_error_mag)
        p_error_capped = np.array([xy_error[0], xy_error[1], p_error[2]])
 
        v_error = v_des - v_current
 
        # PURE PD control law — Marko's correction
        F_cmd = self.Kp * p_error_capped + self.Kd * v_error
 
        F_cmd_mag = np.linalg.norm(F_cmd)
        if F_cmd_mag > 100.0:
            F_cmd = F_cmd / F_cmd_mag * 100.0
 
        # =======================================================================================
        # 6. ORIENTATION
        # =======================================================================================
        theta_error = self.compute_orientation_error(R_current, R_desired)
        tau_rot_cmd = self.Kp_rot * theta_error - self.Kd_rot * omega_current
        tau_rot_mag = np.linalg.norm(tau_rot_cmd)
        if tau_rot_mag > 10.0:
            tau_rot_cmd = tau_rot_cmd / tau_rot_mag * 10.0
 
        # 7. JOINT TORQUES
        wrench_cmd = np.concatenate([F_cmd, tau_rot_cmd])
        J_full = self.get_jacobian_full()
        tau_task = J_full.T @ wrench_cmd
        tau = tau_task + self.get_gravity_compensation()
 
        tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
        tau = np.clip(tau, -tau_max, tau_max)
 
        # Debug storage
        self.last_F_cmd = F_cmd.copy()
        self.last_yaw_des = self.yaw_des
        # flange_axis_world = R_current[:, FLANGE_PUSH_AXIS_LOCAL] * FLANGE_PUSH_SIGN
        # flange_axis_world[2] = 0.0
        # norm = np.linalg.norm(flange_axis_world)
        # if norm > 1e-6:
        #     flange_axis_world /= norm
        # self.last_flange_axis_world = flange_axis_world.copy()
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        push_dir_2d, _, _ = self.traj_manager.get_push_direction(bottle_xy)
        self.last_push_dir = push_dir_2d.copy()
 
        return tau.astype(np.float32)