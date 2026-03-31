import numpy as np


class RewardManager:
    """
    Computes the reward signal for the Panda push environment.

    Reward components:
    - Progress along trajectory
    - Low deviation from trajectory
    - Small angle error θ (force aligned with push direction)
    - Maintain contact
    - Keep bottle upright
    """

    def __init__(self, path_tolerance, target_z):
        self.path_tolerance = path_tolerance
        self.target_z = target_z
        self.prev_progress = 0.0
        self.in_approach = True

    def reset(self):
        self.prev_progress = 0.0
        self.in_approach = True

    def compute(self, deviation_mag, progress, tilt, is_touching,
                force_mag, dist_to_bottle, angle_error,
                hand_z, K_avg, is_settling, episode_length):
        """
        Compute reward from pre-gathered metrics.

        Returns: (total_reward, info)
        """
        info = {"is_success": False}

        if self.in_approach:
            total_reward = -2.0 * dist_to_bottle
            total_reward += -5.0 * abs(hand_z - self.target_z)
            if is_touching or dist_to_bottle < 0.06:
                total_reward += 10.0
                self.in_approach = False

            if episode_length % 50 == 0:
                print(f"Step {episode_length} [APPROACH]: dist={dist_to_bottle:.3f}m")

        else:
            # 1. Progress reward
            progress_delta = progress - self.prev_progress
            r_progress = 100.0 * max(progress_delta, 0)

            # 2. Deviation reward (stay on trajectory)
            max_dev_reward = 5.0
            dev_slope = 100.0
            r_deviation = max_dev_reward - dev_slope * deviation_mag
            r_deviation = max(r_deviation, -15.0)

            # 3. Angle error reward (minimize θ)
            angle_deg = np.degrees(angle_error)
            if angle_deg < 10:
                r_angle = 2.0
            elif angle_deg < 30:
                r_angle = 1.0
            elif angle_deg < 60:
                r_angle = 0.0
            else:
                r_angle = -2.0

            # 4. Stability reward
            if tilt > 0.995:
                r_stability = 3.0
            elif tilt > 0.99:
                r_stability = 1.0
            elif tilt > 0.98:
                r_stability = 0.0
            else:
                r_stability = -15.0 * (1 - tilt)

            # 5. Contact reward
            if is_touching:
                r_contact = 2.0
            else:
                r_contact = -10.0 * dist_to_bottle
                if dist_to_bottle > 0.08:
                    r_contact -= 5.0

            total_reward = r_progress + r_deviation + r_angle + r_stability + r_contact

            if episode_length % 50 == 0:
                mode = "SETTLE" if is_settling else "PUSH"
                contact_status = "CONTACT" if is_touching else "NO CONTACT!"
                print(f"Step {episode_length} [{mode}]: "
                      f"prog={progress:.1%}, dev={deviation_mag * 100:.1f}cm, "
                      f"θ={angle_deg:.1f}°, K={K_avg:.0f}, F={force_mag:.1f}N, "
                      f"tilt={tilt:.4f}, {contact_status}")

        # Time penalty
        total_reward -= 0.005

        # Success
        if progress > 0.95 and deviation_mag < self.path_tolerance:
            total_reward += 100.0
            info["is_success"] = True
            print(f"SUCCESS at step {episode_length}!")

        # Failures
        if tilt < 0.5:
            total_reward -= 100.0
            info["bottle_fallen"] = True

        if deviation_mag > 0.15:
            total_reward -= 50.0
            info["off_path"] = True

        self.prev_progress = progress

        info["progress"] = progress
        info["deviation"] = deviation_mag
        info["angle_error"] = angle_error
        info["force_magnitude"] = force_mag
        info["bottle_tilt"] = tilt
        info["stiffness"] = K_avg

        return total_reward, info