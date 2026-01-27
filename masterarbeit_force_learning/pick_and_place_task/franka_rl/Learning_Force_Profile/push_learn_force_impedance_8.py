"""
Panda Push Environment - STOP AND SETTLE when bottle tilts

THE PROBLEM:
- Bottle has round bottom
- When tilted, pushing causes rotation
- Robot keeps pushing → Bottle falls

THE SOLUTION:
1. Monitor bottle tilt continuously
2. If tilt exceeds threshold → STOP pushing
3. Wait for bottle to settle (become upright again)
4. Resume pushing slowly

This mimics how a human would push a bottle:
- Push gently
- If bottle starts to wobble → Stop!
- Wait for it to stabilize
- Continue pushing
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os


class PandaPushTrajectoryEnv(gym.Env):
    """
    Environment that stops and waits when bottle tilts.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, trajectory_type="straight"):
        super().__init__()

        # ==================== LOAD MODEL ====================
        current_dir = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(current_dir, "robot_panda_push_force.xml")
        xml_path = os.path.abspath(xml_path)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Could not find XML file at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ==================== IDs ====================
        self.home_key_id = self.model.key("home").id
        self.bottle_body_id = self.model.body("bottle").id
        self.hand_body_id = self.model.body("hand").id
        self.goal_site_id = self.model.site("goal").id
        self.left_finger_body_id = self.model.body("left_finger").id
        self.right_finger_body_id = self.model.body("right_finger").id

        self.robot_contact_bodies = {
            self.hand_body_id,
            self.left_finger_body_id,
            self.right_finger_body_id
        }

        # ==================== CONTROL ====================
        self.actuator_ranges = self.model.actuator_ctrlrange[:7, :]
        self.act_low = self.actuator_ranges[:, 0]
        self.act_high = self.actuator_ranges[:, 1]
        self.current_qpos_target = np.zeros(7)

        # ==================== PUSH PARAMETERS ====================
        self.push_speed = 0.010  # SLOWER push speed (10mm/s)
        self.correction_speed = 0.008  # Even slower for correction
        self.behind_distance = 0.04  # 4cm behind bottle

        # ==================== TILT THRESHOLDS (KEY!) ====================
        self.tilt_ok = 0.995  # Above this: Push normally (almost perfectly upright)
        self.tilt_slow = 0.985  # Above this: Push slowly
        self.tilt_stop = 0.975  # Above this: Stop pushing, maintain position
        self.tilt_backup = 0.96  # Below this: Back off slightly!
        self.tilt_settle = 0.99  # Must reach this before resuming push

        # ==================== SETTLING STATE ====================
        self.is_settling = False  # Are we waiting for bottle to settle?
        self.settle_counter = 0  # How long have we been settling?
        self.settle_required = 30  # Steps to wait after settling

        # ==================== STIFFNESS ====================
        self.K_min = 100.0
        self.K_max = 800.0
        self.current_K = np.array([300.0, 300.0])

        # ==================== CARTESIAN ====================
        self.target_z = 0.93
        self.z_gain = 10.0
        self.damping = 0.01

        # ==================== TRAJECTORY ====================
        self.trajectory_type = trajectory_type
        self.trajectory = None
        self.total_arc_length = 0.0
        self.path_tolerance = 0.05

        # ==================== ACTION SPACE ====================
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ==================== OBSERVATION ====================
        obs_dim = 35  # Added: is_settling flag
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ==================== STATE ====================
        self.render_mode = render_mode
        self.viewer = None
        self.episode_length = 0
        self.max_episode_length = 3000  # More time since we stop/wait
        self.prev_progress = 0.0
        self.in_approach_phase = True

        self.force_profile_log = []
        self.stiffness_profile_log = []

        print(f"\n{'=' * 60}")
        print("PUSH WITH STOP-AND-SETTLE")
        print(f"{'=' * 60}")
        print(f"Tilt OK (full speed): > {self.tilt_ok}")
        print(f"Tilt Slow: > {self.tilt_slow}")
        print(f"Tilt STOP: > {self.tilt_stop}")
        print(f"Tilt BACKUP: > {self.tilt_backup}")
        print(f"Must settle to: > {self.tilt_settle}")
        print(f"{'=' * 60}")

    # ==================================================================
    #           TILT MONITORING AND SETTLING
    # ==================================================================

    def _get_bottle_tilt(self):
        """Get bottle uprightness (1.0 = perfectly upright)."""
        bottle_mat = self.data.xmat[self.bottle_body_id].reshape(3, 3)
        return bottle_mat[2, 2]

    def _get_bottle_angular_velocity(self):
        """Get bottle's angular velocity magnitude."""
        # Bottle is first body (free joint), so its angular velocity is at index 3:6
        bottle_angvel = self.data.qvel[3:6]
        return np.linalg.norm(bottle_angvel)

    def _check_bottle_stable(self):
        """Check if bottle is stable enough to push."""
        tilt = self._get_bottle_tilt()
        angvel = self._get_bottle_angular_velocity()

        # Stable = upright AND not rotating
        return tilt > self.tilt_settle and angvel < 0.1

    def _get_push_state(self):
        """
        Determine what the robot should do based on bottle tilt.

        Returns: 'push', 'slow', 'stop', 'backup', 'settle'
        """
        tilt = self._get_bottle_tilt()

        if self.is_settling:
            # Currently settling - check if done
            if self._check_bottle_stable():
                self.settle_counter += 1
                if self.settle_counter >= self.settle_required:
                    # Done settling!
                    self.is_settling = False
                    self.settle_counter = 0
                    return 'push'
            else:
                self.settle_counter = 0  # Reset if not stable
            return 'settle'

        # Not settling - check tilt
        if tilt > self.tilt_ok:
            return 'push'  # Full speed
        elif tilt > self.tilt_slow:
            return 'slow'  # Slow down
        elif tilt > self.tilt_stop:
            return 'stop'  # Stop, but don't back off yet
        elif tilt > self.tilt_backup:
            return 'backup'  # Back off slightly
        else:
            # Tilt is bad - start settling
            self.is_settling = True
            self.settle_counter = 0
            return 'settle'

    # ==================================================================
    #           POSITION CALCULATION
    # ==================================================================

    def _get_push_direction(self, bottle_xy):
        """
        Get direction to push bottle.

        IMMEDIATE CORRECTION: As soon as deviation > threshold,
        blend in correction direction!

        The correction weight increases with deviation:
        - Small deviation (< 1cm): Push forward only
        - Medium deviation (1-3cm): Push forward + some correction
        - Large deviation (> 3cm): Mostly correction, less forward
        """
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)

        # Threshold for starting correction
        correction_start = 0.01  # Start correcting at 1cm deviation!

        if deviation_mag < correction_start:
            # On trajectory - push straight forward
            return tangent
        else:
            # IMMEDIATE CORRECTION!
            # Correction weight increases with deviation
            # At 1cm: 20% correction, 80% forward
            # At 3cm: 60% correction, 40% forward
            # At 5cm+: 80% correction, 20% forward

            correction_weight = min(0.2 + (deviation_mag - correction_start) * 20, 0.8)
            forward_weight = 1.0 - correction_weight

            # Correction direction (towards trajectory)
            correction_dir = deviation_vec / (deviation_mag + 1e-6)

            # Combined push direction
            push_dir = forward_weight * tangent + correction_weight * correction_dir
            push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)

            # Debug: Show when correcting
            if self.episode_length % 25 == 0 and deviation_mag > 0.015:
                print(f"    → CORRECTING: dev={deviation_mag:.3f}m, "
                      f"correction={correction_weight:.0%}, forward={forward_weight:.0%}")

            return push_dir

    def _get_hand_target_position(self, bottle_xy, push_state):
        """
        Get where hand should be.

        KEY: Hand must be BEHIND the bottle relative to push direction!
        When correcting, hand moves to push bottle back to trajectory.
        """
        push_dir = self._get_push_direction(bottle_xy)
        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)

        # Base distance behind bottle
        if push_state == 'backup':
            distance = self.behind_distance + 0.02  # Extra 2cm
        elif push_state in ['stop', 'settle']:
            distance = self.behind_distance + 0.01
        else:
            distance = self.behind_distance

        # Hand position: behind bottle in the direction opposite to push
        hand_target_xy = bottle_xy - push_dir * distance

        # IMPORTANT: When deviating, offset hand to push towards trajectory
        # This ensures the hand is positioned to push the bottle BACK to path
        if deviation_mag > 0.01 and push_state in ['push', 'slow']:
            # Additional offset perpendicular to trajectory, away from it
            # So hand can push bottle TOWARDS trajectory
            correction_dir = deviation_vec / (deviation_mag + 1e-6)
            # Move hand slightly opposite to correction (to push towards correction)
            hand_offset = -correction_dir * min(deviation_mag * 0.5, 0.02)
            hand_target_xy = hand_target_xy + hand_offset

        return np.array([hand_target_xy[0], hand_target_xy[1], self.target_z])

    # ==================================================================
    #           APPROACH PHASE
    # ==================================================================

    def _compute_approach_velocity(self):
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        target_pos = self._get_hand_target_position(bottle_xy, 'push')
        error = target_pos - hand_pos
        dist = np.linalg.norm(error[:2])

        v_desired = error * 1.5
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.04:
            v_desired[:2] = v_desired[:2] / v_mag * 0.04

        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])
        return v_desired, dist

    def _check_approach_complete(self):
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_pos = self.data.xpos[self.bottle_body_id]

        dist_xy = np.linalg.norm(hand_pos[:2] - bottle_pos[:2])
        height_ok = abs(hand_pos[2] - self.target_z) < 0.03
        return dist_xy < 0.08 and height_ok

    # ==================================================================
    #           PUSH PHASE - WITH STOP AND SETTLE
    # ==================================================================

    def _compute_push_velocity(self, action, push_state):
        """
        Compute velocity based on push_state.
        """
        hand_pos = self.data.xpos[self.hand_body_id]
        bottle_xy = self.data.xpos[self.bottle_body_id][:2]

        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)
        push_dir = self._get_push_direction(bottle_xy)

        # Parse stiffness from action
        self.current_K[0] = self.K_min + action[2] * (self.K_max - self.K_min)
        self.current_K[1] = self.K_min + action[3] * (self.K_max - self.K_min)

        # Get target hand position for this state
        target_hand = self._get_hand_target_position(bottle_xy, push_state)

        v_desired = np.zeros(3)

        # === BEHAVIOR BASED ON STATE ===

        if push_state == 'push':
            # FULL PUSH: Move to position + push forward
            pos_error = target_hand[:2] - hand_pos[:2]

            # Faster repositioning when deviation is large
            reposition_gain = 2.0 + min(deviation_mag * 20, 3.0)  # 2.0 to 5.0
            v_reposition = pos_error * reposition_gain
            v_reposition = np.clip(v_reposition, -0.03, 0.03)

            speed = self.push_speed * (1.0 + action[0] * 0.2)
            v_push = push_dir * speed

            v_desired[0] = v_reposition[0] + v_push[0]
            v_desired[1] = v_reposition[1] + v_push[1]

        elif push_state == 'slow':
            # SLOW PUSH: Same as push but slower, still responsive to correction
            pos_error = target_hand[:2] - hand_pos[:2]

            # Still need good repositioning even when slow
            reposition_gain = 1.5 + min(deviation_mag * 15, 2.0)  # 1.5 to 3.5
            v_reposition = pos_error * reposition_gain
            v_reposition = np.clip(v_reposition, -0.02, 0.02)

            speed = self.push_speed * 0.5  # Half speed
            v_push = push_dir * speed

            v_desired[0] = v_reposition[0] + v_push[0]
            v_desired[1] = v_reposition[1] + v_push[1]

        elif push_state == 'stop':
            # STOP: Just maintain position, no pushing
            pos_error = target_hand[:2] - hand_pos[:2]
            v_desired[0] = pos_error[0] * 1.0
            v_desired[1] = pos_error[1] * 1.0
            v_desired[:2] = np.clip(v_desired[:2], -0.01, 0.01)

        elif push_state == 'backup':
            # BACKUP: Move away from bottle
            pos_error = target_hand[:2] - hand_pos[:2]
            v_desired[0] = pos_error[0] * 2.0
            v_desired[1] = pos_error[1] * 2.0
            v_desired[:2] = np.clip(v_desired[:2], -0.02, 0.02)

        elif push_state == 'settle':
            # SETTLE: Hold position, wait for bottle
            pos_error = target_hand[:2] - hand_pos[:2]
            v_desired[0] = pos_error[0] * 0.5
            v_desired[1] = pos_error[1] * 0.5
            v_desired[:2] = np.clip(v_desired[:2], -0.005, 0.005)

        # Height control
        v_desired[2] = self.z_gain * (self.target_z - hand_pos[2])

        # Final limits
        v_desired[:2] = np.clip(v_desired[:2], -0.03, 0.03)
        v_desired[2] = np.clip(v_desired[2], -0.1, 0.1)

        return v_desired

    def _cartesian_to_joint_velocity(self, cart_vel):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)

        J = jacp[:, 6:13]
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.damping ** 2 * np.eye(3))

        return J_pinv @ cart_vel

    # ==================================================================
    #                      TRAJECTORY
    # ==================================================================

    def _generate_trajectory(self, bottle_start_xy, traj_type):
        start = bottle_start_xy.copy()
        n_points = 50

        if traj_type == "straight":
            end = start + np.array([0.0, -0.4])
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, end)
        elif traj_type == "straight_short":
            end = start + np.array([0.0, -0.25])
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, end)
        elif traj_type == "diagonal":
            end = start + np.array([0.15, -0.3])
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, end)
        elif traj_type == "curved":
            t = np.linspace(0, np.pi / 2, n_points)
            radius = 0.2
            x = start[0] + radius * np.sin(t)
            y = start[1] - radius * (1 - np.cos(t))
            traj = np.stack([x, y], axis=1)
        elif traj_type == "s_curve":
            t = np.linspace(0, 1, n_points)
            x = start[0] + 0.08 * np.sin(2 * np.pi * t)
            y = start[1] - 0.35 * t
            traj = np.stack([x, y], axis=1)
        else:
            end = start + np.array([0.0, -0.3])
            t = np.linspace(0, 1, n_points)
            traj = np.outer(1 - t, start) + np.outer(t, end)

        return traj.astype(np.float32)

    def _compute_arc_length(self):
        if self.trajectory is None or len(self.trajectory) < 2:
            self.total_arc_length = 0.0
            return
        diffs = np.diff(self.trajectory, axis=0)
        self.total_arc_length = np.sum(np.linalg.norm(diffs, axis=1))

    def _get_closest_point_on_trajectory(self, pos_xy):
        if self.trajectory is None:
            return 0, pos_xy.copy(), 0.0
        distances = np.linalg.norm(self.trajectory - pos_xy, axis=1)
        idx = np.argmin(distances)
        closest_pt = self.trajectory[idx].copy()
        arc_len = np.sum(np.linalg.norm(np.diff(self.trajectory[:idx + 1], axis=0), axis=1)) if idx > 0 else 0.0
        return idx, closest_pt, arc_len

    def _get_path_deviation(self, bottle_xy):
        _, closest_pt, _ = self._get_closest_point_on_trajectory(bottle_xy)
        deviation_vec = closest_pt - bottle_xy
        return deviation_vec.astype(np.float32), float(np.linalg.norm(deviation_vec))

    def _get_path_tangent(self, bottle_xy):
        if self.trajectory is None or len(self.trajectory) < 2:
            return np.array([0.0, -1.0], dtype=np.float32)
        idx, _, _ = self._get_closest_point_on_trajectory(bottle_xy)
        if idx < len(self.trajectory) - 1:
            tangent = self.trajectory[idx + 1] - self.trajectory[idx]
        else:
            tangent = self.trajectory[idx] - self.trajectory[idx - 1]
        norm = np.linalg.norm(tangent)
        return (tangent / norm).astype(np.float32) if norm > 1e-6 else np.array([0.0, -1.0], dtype=np.float32)

    def _get_progress(self, bottle_xy):
        if self.total_arc_length < 1e-6:
            return 0.0
        _, _, arc_len = self._get_closest_point_on_trajectory(bottle_xy)
        return float(np.clip(arc_len / self.total_arc_length, 0.0, 1.0))

    # ==================================================================
    #                      CONTACT
    # ==================================================================

    def _get_contact_force(self):
        total_force = np.zeros(3, dtype=np.float32)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            robot_touch = body1 in self.robot_contact_bodies or body2 in self.robot_contact_bodies
            bottle_touch = body1 == self.bottle_body_id or body2 == self.bottle_body_id
            if robot_touch and bottle_touch:
                c_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_force)
                frame = contact.frame.reshape(3, 3)
                force_world = frame.T @ c_force[:3]
                total_force += force_world.astype(np.float32)
        return total_force

    def _is_touching(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            robot_touch = body1 in self.robot_contact_bodies or body2 in self.robot_contact_bodies
            bottle_touch = body1 == self.bottle_body_id or body2 == self.bottle_body_id
            if robot_touch and bottle_touch:
                return True
        return False

    # ==================================================================
    #                      OBSERVATION
    # ==================================================================

    def _get_obs(self):
        qpos = self.data.qpos[7:14].astype(np.float32)
        qvel = self.data.qvel[6:13].astype(np.float32)
        hand_pos = self.data.xpos[self.hand_body_id].astype(np.float32)
        bottle_pos = self.data.xpos[self.bottle_body_id].astype(np.float32)
        bottle_xy = bottle_pos[:2]

        hand_to_bottle = bottle_pos[:2] - hand_pos[:2]
        dist_to_bottle = np.linalg.norm(hand_to_bottle)
        dir_to_bottle = hand_to_bottle / (dist_to_bottle + 1e-6)

        deviation_vec, _ = self._get_path_deviation(bottle_xy)
        tangent = self._get_path_tangent(bottle_xy)
        progress = self._get_progress(bottle_xy)
        contact_force = self._get_contact_force()
        is_touching = np.array([1.0 if self._is_touching() else 0.0], dtype=np.float32)

        K_normalized = (self.current_K - self.K_min) / (self.K_max - self.K_min)

        # NEW: Is settling flag
        settling_flag = np.array([1.0 if self.is_settling else 0.0], dtype=np.float32)

        obs = np.concatenate([
            qpos,  # 7
            qvel,  # 7
            hand_pos,  # 3
            bottle_pos,  # 3
            dir_to_bottle,  # 2
            [dist_to_bottle],  # 1
            deviation_vec,  # 2
            tangent,  # 2
            [progress],  # 1
            contact_force,  # 3
            is_touching,  # 1
            K_normalized,  # 2
            settling_flag,  # 1 (NEW!)
        ])

        return obs.astype(np.float32)

    # ==================================================================
    #                      REWARD
    # ==================================================================

    def _get_reward(self):
        info = {"is_success": False}

        bottle_xy = self.data.xpos[self.bottle_body_id][:2]
        hand_pos = self.data.xpos[self.hand_body_id]

        deviation_vec, deviation_mag = self._get_path_deviation(bottle_xy)
        progress = self._get_progress(bottle_xy)
        tilt = self._get_bottle_tilt()
        is_touching = self._is_touching()
        force = self._get_contact_force()
        force_mag = np.linalg.norm(force)
        dist_to_bottle = np.linalg.norm(hand_pos[:2] - bottle_xy)

        push_state = self._get_push_state()

        if is_touching:
            self.in_approach_phase = False

        if self.in_approach_phase:
            total_reward = -2.0 * dist_to_bottle
            total_reward += -5.0 * abs(hand_pos[2] - self.target_z)
            if is_touching:
                total_reward += 10.0

            if self.episode_length % 50 == 0:
                print(f"Step {self.episode_length} [APPROACH]: dist={dist_to_bottle:.3f}m")
        else:
            # Progress
            progress_delta = progress - self.prev_progress
            r_progress = 80.0 * max(progress_delta, 0)

            # Deviation
            if deviation_mag < 0.02:
                r_deviation = 3.0
            elif deviation_mag < self.path_tolerance:
                r_deviation = 1.0
            else:
                r_deviation = -8.0 * deviation_mag

            # Contact
            r_contact = 0.5 if is_touching else -1.0 * dist_to_bottle

            # STABILITY - Most important!
            if tilt > 0.995:
                r_stability = 5.0  # Very stable - big reward!
            elif tilt > 0.99:
                r_stability = 3.0
            elif tilt > 0.98:
                r_stability = 1.0
            elif tilt > 0.97:
                r_stability = 0.0
            elif tilt > 0.95:
                r_stability = -3.0
            else:
                r_stability = -15.0

            # Reward for correct behavior during settling
            if push_state == 'settle':
                if tilt > 0.98:  # Settling and bottle recovering
                    r_settling = 2.0
                else:
                    r_settling = 0.0
            else:
                r_settling = 0.0

            total_reward = (
                    r_progress +
                    r_deviation +
                    r_contact +
                    r_stability +
                    r_settling
            )

            # Print state
            if self.episode_length % 50 == 0:
                state_str = push_state.upper()
                if push_state == 'settle':
                    state_str += f" ({self.settle_counter}/{self.settle_required})"
                print(f"Step {self.episode_length} [{state_str}]: "
                      f"prog={progress:.1%}, "
                      f"dev={deviation_mag:.3f}m, "
                      f"F={force_mag:.1f}N, "
                      f"tilt={tilt:.4f}")

        total_reward -= 0.005  # Small time penalty

        # Success
        if progress > 0.95 and deviation_mag < self.path_tolerance:
            total_reward += 100.0
            info["is_success"] = True
            print(f"SUCCESS at step {self.episode_length}!")

        # Failures
        if tilt < 0.5:
            total_reward -= 100.0
            info["bottle_fallen"] = True

        if deviation_mag > 0.25:
            total_reward -= 30.0
            info["off_path"] = True

        self.prev_progress = progress

        info["progress"] = progress
        info["deviation"] = deviation_mag
        info["bottle_tilt"] = tilt
        info["push_state"] = push_state

        return total_reward, info

    # ==================================================================
    #                      RESET / STEP
    # ==================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)

        self.current_qpos_target = self.data.qpos[7:14].copy()
        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(100):
            self.data.ctrl[:7] = self.current_qpos_target
            mujoco.mj_step(self.model, self.data)

        bottle_start = self.data.xpos[self.bottle_body_id].copy()

        traj_type = self.trajectory_type
        if options and "trajectory_type" in options:
            traj_type = options["trajectory_type"]

        self.trajectory = self._generate_trajectory(bottle_start[:2], traj_type)
        self._compute_arc_length()

        self.prev_progress = 0.0
        self.in_approach_phase = True
        self.episode_length = 0
        self.current_K = np.array([300.0, 300.0])
        self.is_settling = False
        self.settle_counter = 0

        self.force_profile_log = []
        self.stiffness_profile_log = []

        print(f"\n{'=' * 50}")
        print(f"EPISODE: {traj_type}")
        print(f"Bottle: ({bottle_start[0]:.2f}, {bottle_start[1]:.2f})")
        print(f"{'=' * 50}")

        return self._get_obs(), {}

    def step(self, action):
        push_state = self._get_push_state()

        if self.in_approach_phase:
            cart_vel, _ = self._compute_approach_velocity()
            if self._check_approach_complete() or self._is_touching():
                self.in_approach_phase = False
                print(f"Step {self.episode_length}: → PUSH phase")
        else:
            cart_vel = self._compute_push_velocity(action, push_state)

        q_dot = self._cartesian_to_joint_velocity(cart_vel)

        dt = 0.02
        self.current_qpos_target = np.clip(
            self.current_qpos_target + q_dot * dt,
            self.act_low, self.act_high
        )
        self.data.ctrl[:7] = self.current_qpos_target

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Log
        force = self._get_contact_force()
        self.force_profile_log.append(force.copy())
        self.stiffness_profile_log.append(self.current_K.copy())

        obs = self._get_obs()
        reward, info = self._get_reward()
        self.episode_length += 1

        terminated = bool(info.get("is_success") or info.get("bottle_fallen") or info.get("off_path"))
        truncated = bool(self.episode_length >= self.max_episode_length)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
