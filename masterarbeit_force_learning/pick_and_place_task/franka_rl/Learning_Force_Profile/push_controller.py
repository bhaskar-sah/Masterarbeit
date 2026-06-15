"""
push_controller.py

CHANGES from previous version: (PURE FORCE CONTROL)
"""

import numpy as np
import mujoco


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
        self.link0_id = model.body("link0").id

        # Gains
        self.Kd_rot = config.Kd_rot
        
        self.w_max = config.w_max
        
        self.dt = config.dt
        self.control_dt = config.control_dt

        self.n_joints = 7
        self.qvel_start = 6  # set to 0 for no-bottle test, 6 with bottle
        self.logger = None

        # Debug
        self.last_F_cmd = np.zeros(3)
        
        self.last_push_dir = np.zeros(2)
        self.last_t_hat = np.zeros(3)
        self.last_b_hat = np.zeros(3)

    def reset(self):
        pass  # no state to reset — pure force control has no integrator

    def get_base_pos(self):
        return self.data.xpos[self.link0_id].copy()

    def get_base_orientation(self):
        return self.data.xmat[self.link0_id].reshape(3,3).copy()

    # ---------- kinematics (SITE-BASED) ----------
    def get_ee_position(self):
        pos_in_world = self.data.site_xpos[self.gripper_site_id].copy()
        base_pos_in_world = self.get_base_pos()
        w_R_b = self.get_base_orientation()

        translation_vector = pos_in_world - base_pos_in_world

        # DEBUG PRINT:
        # print(f"w_R_b shape: {w_R_b.shape}, pos_in_world shape: {pos_in_world.shape}")
        ee_pos_base_frame = w_R_b.T @ translation_vector

        # print(f"DEBUG: pos in the word: {pos_in_world}")
        # print(f"DEBUG: pos in the word: {translation_vector}")
        # print(f"DEBUG: fRAME ROTATION orientation: {w_R_b}")
        # print(f"DEBUG: Gripper position relative to base: {ee_pos_base_frame}")

        return ee_pos_base_frame

    def get_ee_orientation(self):
        w_R_EE = self.data.site_xmat[self.gripper_site_id].reshape(3, 3).copy()
        w_R_b = self.get_base_orientation()
        return w_R_b.T @ w_R_EE
    
    def get_ee_velocity(self):
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.gripper_site_id)
        vel_world = jacp @ self.data.qvel
        w_R_b = self.get_base_orientation()
        return w_R_b.T @ vel_world

    def get_ee_angular_velocity(self):
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, None, jacr, self.gripper_site_id)
        w_world = jacr @ self.data.qvel
        w_R_b = self.get_base_orientation()
        return w_R_b.T @ w_world

    # def get_jacobian_full(self):
    #     jacp = np.zeros((3, self.model.nv))
    #     jacr = np.zeros((3, self.model.nv))
    #     mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.gripper_site_id)
    #     J_full = np.vstack([jacp, jacr])
    #     return J_full[:, self.qvel_start:self.qvel_start + self.n_joints]

    def get_jacobian_full(self): 
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.gripper_site_id)
        
        # Rotate the MuJoCo World-Jacobian into the Base Frame
        w_R_b = self.get_base_orientation()
        jacp_base = w_R_b.T @ jacp
        jacr_base = w_R_b.T @ jacr
        
        J_full = np.vstack([jacp_base, jacr_base])
        return J_full[:, self.qvel_start:self.qvel_start + self.n_joints]

    def get_gravity_compensation(self):
        return self.data.qfrc_bias[self.qvel_start:self.qvel_start + self.n_joints].copy()

    def get_bottle_tilt(self):
        return float(self.data.xmat[self.bottle_body_id].reshape(3, 3)[2, 2])

    def get_flange_axis_world(self):
        R_current = self.get_ee_orientation()
        # get local X-Axis of the flange
        axis = R_current[:, 0].copy()
        axis[2] = 0.0
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            axis /= norm
        return axis
    
    def get_rpy(self):
        """Calculate Roll, Pitch, Yaw from the rotation matrix."""
        R = self.get_ee_orientation()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        roll = np.arctan2(R[2, 1], R[2, 2])
        return np.array([roll, pitch, yaw])

    def get_measured_wrench(self):
        """Calculate ACTUAL measured forces/torques from motor outputs."""
        J_full = self.get_jacobian_full()
        # Read the actual motor torque applied, minus gravity
        tau_meas_joints = self.data.qfrc_actuator[self.qvel_start:self.qvel_start + self.n_joints] - self.get_gravity_compensation()
        W_meas_base = np.linalg.pinv(J_full.T) @ tau_meas_joints
        return W_meas_base[:3], W_meas_base[3:]
    
    # ====================================================================
    # PATH FRAME — single source of truth
    # ====================================================================
    def compute_path_frame_base(self, bottle_xy_world):
        w_R_b = self.get_base_orientation()
        base_pos_world = self.get_base_pos()
 
        # --- world-frame path queries ---
        push_dir_2d_world, _, _ = self.traj_manager.get_push_direction(bottle_xy_world)
        deviation_vec_world, _ = self.traj_manager.get_path_deviation(bottle_xy_world)
 
        # get_path_deviation returns (closest_pt - bottle_xy), i.e. bottle -> path.
        # Therefore closest_pt = bottle_xy + deviation_vec.  (Sign fixed.)
        closest_pt_2d_world = bottle_xy_world + deviation_vec_world
 
        target_z = getattr(self.config, "target_z", 0.84)
        p_path_world = np.array([closest_pt_2d_world[0], closest_pt_2d_world[1], target_z])
        t_hat_world = np.array([push_dir_2d_world[0], push_dir_2d_world[1], 0.0])
        n_hat_world = np.array([0.0, 0.0, 1.0])
 
        # --- into base frame ---
        p_path = w_R_b.T @ (p_path_world - base_pos_world)   # position: rotate + translate
        t_hat = w_R_b.T @ t_hat_world                        # direction: rotate only
        n_hat = w_R_b.T @ n_hat_world
 
        # --- orthonormalise ---
        nt = np.linalg.norm(t_hat)
        t_hat = t_hat / nt if nt > 1e-6 else np.array([1.0, 0.0, 0.0])
        n_hat = n_hat / np.linalg.norm(n_hat)
        b_hat = np.cross(n_hat, t_hat)
        b_hat /= np.linalg.norm(b_hat)
 
        R_path = np.column_stack((t_hat, b_hat, n_hat))
        return R_path, t_hat, b_hat, n_hat, p_path


    def compute_torque_from_wrench(self, action, tilt=None):
            # ====================================================================
            # 1. GET ROBOT STATE (Already in Base Frame)
            # ====================================================================
            p_ee = self.get_ee_position()
            v_ee = self.get_ee_velocity()
            
            # ====================================================================
            # 2. GET PATH STATE (Query in World, Convert to Base)
            # ====================================================================
            bottle_xy_world = self.data.xpos[self.bottle_body_id][:2].copy()
            R_path, t_hat, b_hat, n_hat, p_path = self.compute_path_frame_base(bottle_xy_world)

            # ====================================================================
            # 3. HEIGHT CONTROL (Generalized Dot Product)
            # ====================================================================
            delta_p = p_ee - p_path
            d = np.dot(delta_p, n_hat)
            d_dot = np.dot(v_ee, n_hat)
            
            # d_des is usually 0.0 if you want to be exactly on the path
            d_des = 0.0 
            
            F_normal = (
                self.config.Kz * (d_des - d) 
                - self.config.Dz * d_dot
            )

            # ====================================================================
            # 4. RL POLICY IN PATH FRAME
            # ====================================================================
            F_path = np.array([
                action[0] * self.config.f_max,
                action[1] * self.config.f_max,
                F_normal,
            ])
            
            Tau_path = np.array([
                0.0,
                0.0,
                action[2] * self.config.tau_rot_max,
            ])

            # ====================================================================
            # 5. TRANSFORM TO BASE FRAME {p} -> {B}
            # ====================================================================
            F_base = R_path @ F_path
            Tau_task_base = R_path @ Tau_path


            # 6. Assemble base-frame wrench
            wrench_base = np.concatenate([F_base, Tau_task_base])

            # ============================================================
            # 7. MANUAL WRIST-ORIENTATION PD  (roll/pitch only; yaw stays with RL)
            # Idea from your document.
            # Holds the gripper upright so contact reaction can't tip it flat.
            # Everything here is in the base frame {B}, matching wrench_base.
            # The spring e_rot = z_ee x z_target is automatically free of yaw
            # about n_hat, and the damper is projected to roll/pitch only, so
            # this NEVER fights the RL's Tz action (which is purely along n_hat).
            # ============================================================
            R_ee = self.get_ee_orientation()        # gripper orientation in {B}
            z_ee = R_ee[:, 2]                        # gripper local z-axis in {B}
            w_ee = self.get_ee_angular_velocity()    # EE angular velocity in {B}
    
            # one-shot sign check: upright must read ~[0, 0, -1]
            if not getattr(self, "_z_ee_checked", False):
                print(f"[ORIENT PD] z_ee at first step = {np.round(z_ee, 3)} "
                    f"(upright should be ~[0,0,-1]; if ~[0,0,+1], flip z_target sign)")
                self._z_ee_checked = True
    
            # Target: gripper z points "down" toward the table, i.e. along -n_hat.
            z_target = -n_hat
    
            # Restoring spring (cross product => no component along n_hat => yaw-free)
            e_rot = np.cross(z_ee, z_target)
    
            # Damp ONLY roll/pitch: strip the n_hat component of angular velocity
            w_rollpitch = w_ee - np.dot(w_ee, n_hat) * n_hat
    
            Tau_align = self.config.Kp_rot * e_rot - self.config.Kd_rot * w_rollpitch
    
            # Add to the moment part of the wrench (do NOT replace the RL yaw)
            wrench_base[3:] += Tau_align

            # ====================================================================
            # 7. CALCULATE JOINT TORQUES
            # ====================================================================     
            J_full = self.get_jacobian_full()
            tau_task = J_full.T @ wrench_base
            tau = tau_task + self.get_gravity_compensation()

            tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
            tau = np.clip(tau, -tau_max, tau_max)

            # Update logger variables so your plots stay accurate
            self.last_F_cmd_ee = F_path
            self.last_tau_cmd_ee = Tau_path
            self.last_F_cmd_base_pure = F_base
            self.last_tau_cmd_base = Tau_task_base
            self.last_F_cmd_base_total = F_base
            self.last_R_path = R_path
            self.last_t_hat = t_hat
            self.last_b_hat = b_hat
            self.last_push_dir = t_hat[:2].copy()

            return tau.astype(np.float32)


    def compute_torque_from_motion(self, action, tilt=None):
        """
        MOTION-BASED controller (impedance / velocity-tracking), as a baseline
        to compare against compute_torque_from_wrench.

        Parallel structure to the wrench controller:
            - same path frame {P} via compute_path_frame_base()
            - same height regulation along n_hat (here as a velocity command)
            - same manual wrist roll/pitch PD (holds the DOF the RL can't)
            - same logger handoff

        Difference: the RL action is a DESIRED TWIST in {P} (planar velocity +
        yaw rate), and the joint torque comes from an impedance law that drives
        the measured twist toward that desired twist:
            F   = kv * (v_des_base - v_ee)
            Tau = kw * (w_des_base - w_ee)
        """
        # ====================================================================
        # 1. ROBOT STATE (base frame {B})
        # ====================================================================
        p_ee = self.get_ee_position()
        v_ee = self.get_ee_velocity()
        w_ee = self.get_ee_angular_velocity()

        # ====================================================================
        # 2. PATH FRAME {P}  (same single-source builder as the wrench ctrl)
        # ====================================================================
        bottle_xy_world = self.data.xpos[self.bottle_body_id][:2].copy()
        R_path, t_hat, b_hat, n_hat, p_path = self.compute_path_frame_base(bottle_xy_world)

        # ====================================================================
        # 3. HEIGHT CONTROL  (velocity command along n_hat, with damping)
        # ====================================================================
        delta_p = p_ee - p_path
        d = np.dot(delta_p, n_hat)        # signed height error along n_hat
        d_dot = np.dot(v_ee, n_hat)       # current normal velocity
        d_des = 0.0                       # stay on the path plane
        v_normal = self.config.kp_normal * (d_des - d) - self.config.kd_normal * d_dot

        # ====================================================================
        # 4. RL POLICY = DESIRED TWIST IN PATH FRAME {P}
        # ====================================================================
        linear_vel_path = np.array([
            action[0] * self.config.v_max,    # along t_hat
            action[1] * self.config.v_max,    # along b_hat
            v_normal,                         # along n_hat (controller, not RL)
        ])
        angular_vel_path = np.array([
            0.0,
            0.0,
            action[2] * self.config.w_max,    # yaw rate about n_hat
        ])

        # ====================================================================
        # 5. {P} -> {B}
        # ====================================================================
        linear_vel_base = R_path @ linear_vel_path
        angular_vel_base = R_path @ angular_vel_path

        # ====================================================================
        # 6. IMPEDANCE LAW: torque from twist error
        # ====================================================================
        v_err = linear_vel_base - v_ee
        w_err = angular_vel_base - w_ee

        F_base = self.config.kv * v_err
        Tau_task_base = self.config.kw * w_err

        wrench_base = np.concatenate([F_base, Tau_task_base])

        # ====================================================================
        # 7. MANUAL WRIST-ORIENTATION PD  (roll/pitch only; yaw stays with RL)
        # Identical to the wrench controller — holds the gripper upright so
        # contact reaction can't tip it flat. Orthogonal to the RL yaw axis.
        # ====================================================================
        R_ee = self.get_ee_orientation()
        z_ee = R_ee[:, 2]

        if not getattr(self, "_z_ee_checked", False):
            print(f"[ORIENT PD] z_ee at first step = {np.round(z_ee, 3)} "
                  f"(upright should be ~[0,0,-1]; if ~[0,0,+1], flip z_target sign)")
            self._z_ee_checked = True

        z_target = -n_hat
        e_rot = np.cross(z_ee, z_target)
        w_rollpitch = w_ee - np.dot(w_ee, n_hat) * n_hat
        Tau_align = self.config.Kp_rot * e_rot - self.config.Kd_rot * w_rollpitch

        wrench_base[3:] += Tau_align

        # ====================================================================
        # 8. JOINT TORQUES  (J is base-frame, wrench is base-frame)
        # ====================================================================
        J_full = self.get_jacobian_full()
        tau = J_full.T @ wrench_base + self.get_gravity_compensation()
        tau_max = np.array([87, 87, 87, 87, 12, 12, 12])

        tau = np.clip(tau, -tau_max, tau_max)

        # ====================================================================
        # 9. LOGGER HANDOFF  (same fields as the wrench controller, so that the
        #    plots/CSV keep working. F_cmd_path here is the COMMANDED FORCE
        #    that the impedance law produced -- not the raw action.)
        # ====================================================================
        self.last_F_cmd_ee = R_path.T @ F_base          # commanded force expressed in {P}
        self.last_tau_cmd_ee = R_path.T @ Tau_task_base
        self.last_F_cmd_base_pure = F_base
        self.last_tau_cmd_base = Tau_task_base
        self.last_F_cmd_base_total = F_base
        self.last_R_path = R_path
        self.last_t_hat = t_hat
        self.last_b_hat = b_hat
        self.last_push_dir = t_hat[:2].copy()

        return tau.astype(np.float32)
