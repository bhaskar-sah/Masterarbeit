"""
push_controller.py

CHANGES from previous version: (PURE FORCE CONTROL)
"""

import numpy as np
import mujoco


class PushController:
    def __init__(self, model, data, traj_manager, contact_manager,
                 gripper_site_id, bottle_body_id, config):
        self.model = model
        self.data = data
        self.traj_manager = traj_manager
        self.contact_manager = contact_manager
        self.gripper_site_id = gripper_site_id
        self.bottle_body_id = bottle_body_id
        self.config = config

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
    def get_ee_pos_base(self):
        ee_pos_in_world = self.data.site_xpos[self.gripper_site_id].copy()
        base_pos_in_world = self.get_base_pos()
        w_R_b = self.get_base_orientation()

        translation_vector = ee_pos_in_world - base_pos_in_world

        # DEBUG PRINT:
        # print(f"w_R_b shape: {w_R_b.shape}, ee_pos_in_world shape: {ee_pos_in_world.shape}")
        ee_pos_in_base = w_R_b.T @ translation_vector

        # print(f"DEBUG: pos in the word: {ee_pos_in_world}")
        # print(f"DEBUG: pos in the word: {translation_vector}")
        # print(f"DEBUG: fRAME ROTATION orientation: {w_R_b}")
        # print(f"DEBUG: Gripper position relative to base: {ee_pos_in_base}")

        return ee_pos_in_base

    def get_ee_pos_world(self):
        return self.data.site_xpos[self.gripper_site_id].copy()

    def get_ee_orientation_world(self):
        return self.data.site_xmat[self.gripper_site_id].reshape(3, 3).copy()

    def get_ee_orientation_base(self):
        w_R_EE = self.data.site_xmat[self.gripper_site_id].reshape(3, 3).copy()
        w_R_b = self.get_base_orientation()
        return w_R_b.T @ w_R_EE
    
    def get_ee_vel_base(self):
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.gripper_site_id)
        vel_world = jacp @ self.data.qvel
        w_R_b = self.get_base_orientation()
        return w_R_b.T @ vel_world

    def get_ee_vel_world(self):
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.gripper_site_id)
        return jacp @ self.data.qvel

    def get_ee_ang_vel_base(self):
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, None, jacr, self.gripper_site_id)
        w_world = jacr @ self.data.qvel
        w_R_b = self.get_base_orientation()
        return w_R_b.T @ w_world

    def get_ee_ang_vel_world(self):
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, None, jacr, self.gripper_site_id)
        return jacr @ self.data.qvel

    def get_jacobian_full_world(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.gripper_site_id)
        J_full = np.vstack([jacp, jacr])
        return J_full[:, self.qvel_start:self.qvel_start + self.n_joints]

    def get_jacobian_full_base(self):
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
        z_axis_bottle = self.data.xmat[self.bottle_body_id].reshape(3, 3)[:, 2]
        n_hat = self.get_path_normal()
        tilt = float(np.dot(z_axis_bottle, n_hat))
        return tilt

    def get_ee_x_dir_world(self):
        R_current = self.get_ee_orientation_world()
        # get local X-Axis of the flange
        axis = R_current[:, 0].copy()
        axis[2] = 0.0
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            axis /= norm
        return axis
    
    def get_rpy(self):
        """Calculate Roll, Pitch, Yaw from the rotation matrix."""
        R = self.get_ee_orientation_world()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        roll = np.arctan2(R[2, 1], R[2, 2])
        return np.array([roll, pitch, yaw])

    def get_measured_wrench_base(self):
        """Calculate ACTUAL measured forces/torques from motor outputs."""
        J_full = self.get_jacobian_full_base()
        # Read the actual motor torque applied, minus gravity
        tau_meas_joints = self.data.qfrc_actuator[self.qvel_start:self.qvel_start + self.n_joints] - self.get_gravity_compensation()
        W_meas_base = np.linalg.pinv(J_full.T) @ tau_meas_joints
        return W_meas_base[:3], W_meas_base[3:]

    def get_measured_wrench_world(self):
        """Calculate ACTUAL measured forces/torques from motor outputs."""
        J_full = self.get_jacobian_full_world()
        # Read the actual motor torque applied, minus gravity
        tau_meas_joints = self.data.qfrc_actuator[self.qvel_start:self.qvel_start + self.n_joints] - self.get_gravity_compensation()
        W_meas_world = np.linalg.pinv(J_full.T) @ tau_meas_joints
        return W_meas_world[:3], W_meas_world[3:]
    
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

 # TODO: trajectory calculations should be in 3D so that we don't need this manual table height here
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

    def compute_path_frame_world(self, bottle_xy_world):
        # --- world-frame path queries ---
        push_dir_2d_world, _, _ = self.traj_manager.get_push_direction(bottle_xy_world)
        deviation_vec_world, _ = self.traj_manager.get_path_deviation(bottle_xy_world)

        # get_path_deviation returns (closest_pt - bottle_xy), i.e. bottle -> path.
        # Therefore closest_pt = bottle_xy + deviation_vec.  (Sign fixed.)
        closest_pt_2d_world = bottle_xy_world + deviation_vec_world

        # TODO: trajectory calculations should be in 3D so that we don't need this manual table height here
        table_height_world = 0.8
        p_path_world = np.array([closest_pt_2d_world[0], closest_pt_2d_world[1], table_height_world])
        t_hat_world = np.array([push_dir_2d_world[0], push_dir_2d_world[1], 0.0])
        n_hat_world = np.array([0.0, 0.0, 1.0])

        # --- orthonormalise ---
        nt = np.linalg.norm(t_hat_world)
        t_hat = t_hat_world / nt if nt > 1e-6 else np.array([0.0, 0.0, 0.0])
        n_hat = n_hat_world / np.linalg.norm(n_hat_world)
        b_hat = np.cross(n_hat, t_hat)
        b_hat /= np.linalg.norm(b_hat)

        R_path = np.column_stack((t_hat, b_hat, n_hat))
        return R_path, t_hat, b_hat, n_hat, p_path_world

    # ===== retunr the path normal =====
    # TODO: for now this function returns a fixed normal direction
    def get_path_normal(self):
        return np.array([0.0, 0.0, 1.0])


    def compute_torque_from_wrench(self, action, tilt=None):
        # ====================================================================
        # 1. GET ROBOT STATE (Already in Base Frame)
        # ====================================================================
        p_ee_world = self.get_ee_pos_world()
        v_ee_world = self.get_ee_vel_world()

        # ====================================================================
        # 2. GET PATH STATE (Query in World, Convert to Base)
        # ====================================================================
        p_bottle_world = self.data.xpos[self.bottle_body_id].copy()
        R_path_world, t_hat_world, b_hat_world, n_hat_world, p_path_world = self.compute_path_frame_world(p_bottle_world[:2])

        # ====================================================================
        # 3. HEIGHT CONTROL (Generalized Dot Product)
        # ====================================================================
        delta_p = p_ee_world - p_path_world
        d = np.dot(delta_p, n_hat_world)
        d_dot = np.dot(v_ee_world, n_hat_world)

        # d_des is object-height * 0.4 -> push a little bit below (0.4) the objects center
        # TODO: Do that programatically by reading height from mujoco geometry (for now manual)
        object_height = 0.12
        d_des =  object_height*0.4

        F_normal = (
                self.config.Kfz * (d_des - d)
                - self.config.Dfz * d_dot
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
        F_world = R_path_world @ F_path
        Tau_task_world = R_path_world @ Tau_path
        wrench_world = np.concatenate([F_world, Tau_task_world])

        #===============
        # Add orientation control
        #===============
        w_ee_world = self.get_ee_ang_vel_world()
        R_ee_world = self.get_ee_orientation_world()
        z_ee_world = R_ee_world[:,2]
        e_rot = np.cross(z_ee_world, -n_hat_world)
        tau_align = (
                self.config.Kp_rot * e_rot
                #- self.config.Kd_rot * w_ee_world
        )
        wrench_world[3:] += tau_align

        # ====================================================================
        # 7. CALCULATE JOINT TORQUES
        # ====================================================================
        J_full = self.get_jacobian_full_world()
        tau_task = J_full.T @ wrench_world
        tau = tau_task + self.get_gravity_compensation()

        tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
        tau = np.clip(tau, -tau_max, tau_max)

        # Update logger variables so your plots stay accurate
        self.last_F_path_cmd = F_path
        self.last_tau_path_cmd = Tau_path
        self.last_F_world_cmd = F_world
        self.last_tau_world_cmd = Tau_task_world
        self.last_R_path = R_path_world
        self.last_t_hat = t_hat_world
        self.last_b_hat = b_hat_world

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
        # 1. ROBOT STATE {W} (in world frame)
        # ====================================================================
        p_ee_world = self.get_ee_pos_world()
        v_ee_world = self.get_ee_vel_world()
        w_ee_world = self.get_ee_ang_vel_world()

        # ====================================================================
        # 2. PATH FRAME {P}  (same single-source builder as the wrench ctrl)
        # ====================================================================
        p_bottle_world = self.data.xpos[self.bottle_body_id].copy()
        R_path_world, t_hat_world, b_hat_world, n_hat_world, p_path_world = self.compute_path_frame_world(
            p_bottle_world[:2])

        # ====================================================================
        # 3. HEIGHT CONTROL  (velocity command along n_hat, with damping)
        # ====================================================================
        delta_p = p_ee_world - p_path_world
        d = np.dot(delta_p, n_hat_world)
        d_dot = np.dot(v_ee_world, n_hat_world)

        # d_des is object-height * 0.4 -> push a little bit below (0.4) the objects center
        # TODO: Do that programatically by reading height from mujoco geometry (for now manual)
        object_height = 0.12
        d_des = object_height * 0.4

        v_normal = self.config.Kvz * (d_des - d) - self.config.Dvz * d_dot

        # ====================================================================
        # 4. RL POLICY = DESIRED TWIST IN PATH FRAME {P}
        # ====================================================================
        linear_vel_path = np.array([
            action[0] * self.config.v_max,  # along t_hat
            action[1] * self.config.v_max,  # along b_hat
            v_normal,  # along n_hat (controller, not RL)
        ])
        angular_vel_path = np.array([
            0.0,
            0.0,
            action[2] * self.config.w_max,  # yaw rate about n_hat
        ])

        # ====================================================================
        # 5. {P} -> {W}
        # ====================================================================
        linear_vel_world = R_path_world @ linear_vel_path
        angular_vel_world = R_path_world @ angular_vel_path

        # ====================================================================
        # 6. IMPEDANCE LAW: torque from twist error
        # ====================================================================
        v_err = linear_vel_world - v_ee_world
        w_err = angular_vel_world - w_ee_world

        K_v = R_path_world @ np.diag([self.config.kv_t, self.config.kv_b, self.config.kv_n]) @ R_path_world.T
        F_world = K_v @ v_err
        Tau_task_world = self.config.Dw * w_err

        wrench_world = np.concatenate([F_world, Tau_task_world])

        # ===============
        # Add orientation control
        # ===============
        R_ee_world = self.get_ee_orientation_world()
        z_ee_world = R_ee_world[:, 2]
        e_rot = np.cross(z_ee_world, -n_hat_world)
        tau_align = (
                self.config.Kp_rot * e_rot
            # - self.config.Kd_rot * w_ee_world
        )
        wrench_world[3:] += tau_align

        # ====================================================================
        # 8. JOINT TORQUES  (J is base-frame, wrench is base-frame)
        # ====================================================================
        J_full = self.get_jacobian_full_world()
        tau = J_full.T @ wrench_world + self.get_gravity_compensation()
        tau_max = np.array([87, 87, 87, 87, 12, 12, 12])

        tau = np.clip(tau, -tau_max, tau_max)

        # ====================================================================
        # 9. LOGGER HANDOFF  (same fields as the wrench controller, so that the
        #    plots/CSV keep working. F_cmd_path here is the COMMANDED FORCE
        #    that the impedance law produced -- not the raw action.)
        # ====================================================================
        # Update logger variables so your plots stay accurate
        self.last_F_path_cmd = R_path_world.T @ F_world
        self.last_tau_path_cmd = R_path_world.T @ Tau_task_world
        self.last_F_world_cmd = F_world
        self.last_tau_world_cmd = Tau_task_world
        self.last_R_path = R_path_world
        self.last_t_hat = t_hat_world
        self.last_b_hat = b_hat_world

        return tau.astype(np.float32)

    def compute_torque_from_manual_motion(self, action, tilt=None):
        """
        MOTION-BASED controller without RL actions. Input is only a tangential desired velocity vx.
        Rest of the motion is manually created to follow the trajectory.
        """
        # ====================================================================
        # 1. ROBOT STATE {W} (in world frame)
        # ====================================================================
        p_ee_world = self.get_ee_pos_world()
        v_ee_world = self.get_ee_vel_world()
        w_ee_world = self.get_ee_ang_vel_world()

        # ====================================================================
        # 2. PATH FRAME {P}  (same single-source builder as the wrench ctrl)
        # ====================================================================
        p_bottle_world = self.data.xpos[self.bottle_body_id].copy()
        R_path_world, t_hat_world, b_hat_world, n_hat_world, p_path_world = self.compute_path_frame_world(
            p_bottle_world[:2])

        # ====================================================================
        # 3. HEIGHT CONTROL  (velocity command along n_hat, with damping)
        # ====================================================================
        delta_p = p_ee_world - p_path_world
        d = np.dot(delta_p, n_hat_world)
        d_dot = np.dot(v_ee_world, n_hat_world)

        # d_des is object-height * 0.4 -> push a little bit below (0.4) the objects center
        # TODO: Do that programatically by reading height from mujoco geometry (for now manual)
        object_height = 0.12
        d_des = object_height * 0.4

        v_normal = self.config.Kvz * (d_des - d) - self.config.Dvz * d_dot

        # ======================
        # Angular Alignment (end-effector keeps facing the path tangent)
        # ======================
        R_ee_world = self.get_ee_orientation_world()
        y_ee_world = R_ee_world[:,1]

        e_rot_y_world = np.cross(y_ee_world, b_hat_world)

        # ====================================================================
        # 4. RL POLICY = DESIRED TWIST IN PATH FRAME {P}
        # ====================================================================
        linear_vel_path = np.array([
            action[0] * self.config.v_max,  # along t_hat
            0.0,  # along b_hat
            v_normal,  # along n_hat (controller, not RL)
        ])
        angular_vel_path = 10.0*R_path_world.T @ e_rot_y_world

        # ====================================================================
        # 5. {P} -> {W}
        # ====================================================================
        linear_vel_world = R_path_world @ linear_vel_path
        angular_vel_world = R_path_world @ angular_vel_path

        # ====================================================================
        # 6. IMPEDANCE LAW: torque from twist error
        # ====================================================================
        v_err = linear_vel_world - v_ee_world
        w_err = angular_vel_world - w_ee_world

        K_v = R_path_world @ np.diag([self.config.kv_t, self.config.kv_b, self.config.kv_n]) @ R_path_world.T
        F_world = K_v @ v_err
        Tau_task_world = self.config.Dw * w_err

        wrench_world = np.concatenate([F_world, Tau_task_world])

        # ===============
        # Add orientation control
        # ===============
        z_ee_world = R_ee_world[:, 2]
        e_rot = np.cross(z_ee_world, -n_hat_world)
        tau_align = (
                self.config.Kp_rot * e_rot
            # - self.config.Kd_rot * w_ee_world
        )
        wrench_world[3:] += tau_align

        # ====================================================================
        # 8. JOINT TORQUES  (J is base-frame, wrench is base-frame)
        # ====================================================================
        J_full = self.get_jacobian_full_world()
        tau = J_full.T @ wrench_world + self.get_gravity_compensation()
        tau_max = np.array([87, 87, 87, 87, 12, 12, 12])

        tau = np.clip(tau, -tau_max, tau_max)

        # ====================================================================
        # 9. LOGGER HANDOFF  (same fields as the wrench controller, so that the
        #    plots/CSV keep working. F_cmd_path here is the COMMANDED FORCE
        #    that the impedance law produced -- not the raw action.)
        # ====================================================================
        # Update logger variables so your plots stay accurate
        self.last_F_path_cmd = R_path_world.T @ F_world
        self.last_tau_path_cmd = R_path_world.T @ Tau_task_world
        self.last_F_world_cmd = F_world
        self.last_tau_world_cmd = Tau_task_world
        self.last_R_path = R_path_world
        self.last_t_hat = t_hat_world
        self.last_b_hat = b_hat_world

        return tau.astype(np.float32)
