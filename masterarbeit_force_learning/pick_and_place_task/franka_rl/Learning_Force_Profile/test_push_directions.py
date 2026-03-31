"""
Test script to verify the robot can push the bottle in ALL directions.
This bypasses the trajectory following and directly commands specific push directions.
"""

import numpy as np
import mujoco
import mujoco.viewer
import os
import time

# Load model
current_dir = os.path.dirname(os.path.realpath(__file__))
xml_path = os.path.join(current_dir, "robot_panda_push_force.xml")

if not os.path.exists(xml_path):
    # Try the regular one
    xml_path = os.path.join(current_dir, "robot_panda_push_force.xml")

print(f"Loading: {xml_path}")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# IDs
home_key_id = model.key("home").id
bottle_body_id = model.body("bottle").id
hand_body_id = model.body("hand").id

# Reset to home
mujoco.mj_resetDataKeyframe(model, data, home_key_id)
mujoco.mj_forward(model, data)

# Get initial positions
current_qpos_target = data.qpos[7:14].copy()
data.ctrl[:7] = current_qpos_target

# Stabilize
for _ in range(100):
    data.ctrl[:7] = current_qpos_target
    mujoco.mj_step(model, data)

# Control parameters
actuator_ranges = model.actuator_ctrlrange[:7, :]
act_low = actuator_ranges[:, 0]
act_high = actuator_ranges[:, 1]
damping = 0.01
target_z = 0.92
z_gain = 10.0


def cartesian_to_joint_velocity(cart_vel):
    """Convert Cartesian velocity to joint velocities."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_body_id)

    J = jacp[:, 6:13]
    JJT = J @ J.T
    J_pinv = J.T @ np.linalg.inv(JJT + damping ** 2 * np.eye(3))

    return J_pinv @ cart_vel


def run_push_test(push_direction, direction_name, steps=300):
    """Push the bottle in a specific direction."""
    global current_qpos_target

    # Reset
    mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    mujoco.mj_forward(model, data)
    current_qpos_target = data.qpos[7:14].copy()

    for _ in range(100):
        data.ctrl[:7] = current_qpos_target
        mujoco.mj_step(model, data)

    bottle_start = data.xpos[bottle_body_id].copy()
    print(f"\n{'=' * 60}")
    print(f"TEST: Push {direction_name}")
    print(f"Push direction: [{push_direction[0]:.2f}, {push_direction[1]:.2f}]")
    print(f"Bottle start: [{bottle_start[0]:.3f}, {bottle_start[1]:.3f}]")
    print(f"{'=' * 60}")

    behind_distance = 0.03

    # Phase 1: Approach (100 steps)
    print("Phase 1: Approach...")
    for step in range(100):
        hand_pos = data.xpos[hand_body_id]
        bottle_xy = data.xpos[bottle_body_id][:2]

        # Target: behind bottle along push direction
        hand_target = bottle_xy - push_direction * behind_distance
        hand_target_3d = np.array([hand_target[0], hand_target[1], target_z])

        error = hand_target_3d - hand_pos
        v_desired = error * 3.0
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.08:
            v_desired[:2] = v_desired[:2] / v_mag * 0.08
        v_desired[2] = z_gain * (target_z - hand_pos[2])

        q_dot = cartesian_to_joint_velocity(v_desired)
        dt = 0.02

        current_qpos_target[:6] = np.clip(
            current_qpos_target[:6] + q_dot[:6] * dt,
            act_low[:6], act_high[:6]
        )
        data.ctrl[:7] = current_qpos_target

        for _ in range(20):
            mujoco.mj_step(model, data)

    hand_pos = data.xpos[hand_body_id]
    bottle_xy = data.xpos[bottle_body_id][:2]
    print(f"After approach - Hand: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}], "
          f"Bottle: [{bottle_xy[0]:.3f}, {bottle_xy[1]:.3f}]")

    # Phase 2: Push (200 steps)
    print("Phase 2: Push...")
    for step in range(200):
        hand_pos = data.xpos[hand_body_id]
        bottle_xy = data.xpos[bottle_body_id][:2]
        bottle_tilt = data.xmat[bottle_body_id].reshape(3, 3)[2, 2]

        # Target: behind bottle along push direction
        hand_target = bottle_xy - push_direction * behind_distance
        hand_target_3d = np.array([hand_target[0], hand_target[1], target_z])

        # Tracking velocity
        pos_error = hand_target_3d[:2] - hand_pos[:2]
        v_tracking = pos_error * 10.0
        v_tracking = np.clip(v_tracking, -0.15, 0.15)

        # Push velocity
        push_speed = 0.02
        v_push = push_direction * push_speed

        # Combined
        v_desired = np.zeros(3)
        v_desired[0] = v_push[0] + v_tracking[0]
        v_desired[1] = v_push[1] + v_tracking[1]

        # Limit
        v_mag = np.linalg.norm(v_desired[:2])
        if v_mag > 0.12:
            v_desired[:2] = v_desired[:2] / v_mag * 0.12

        v_desired[2] = z_gain * (target_z - hand_pos[2])

        q_dot = cartesian_to_joint_velocity(v_desired)
        dt = 0.02

        current_qpos_target[:6] = np.clip(
            current_qpos_target[:6] + q_dot[:6] * dt,
            act_low[:6], act_high[:6]
        )
        data.ctrl[:7] = current_qpos_target

        for _ in range(20):
            mujoco.mj_step(model, data)

        if step % 50 == 0:
            print(f"  Step {step}: Bottle=[{bottle_xy[0]:.3f}, {bottle_xy[1]:.3f}], "
                  f"tilt={bottle_tilt:.3f}")

        if bottle_tilt < 0.7:
            print(f"  BOTTLE TIPPED at step {step}!")
            break

    bottle_end = data.xpos[bottle_body_id][:2]
    displacement = bottle_end - bottle_start[:2]

    print(f"\nRESULT:")
    print(f"  Bottle end: [{bottle_end[0]:.3f}, {bottle_end[1]:.3f}]")
    print(f"  Displacement: [{displacement[0] * 100:.1f}, {displacement[1] * 100:.1f}] cm")
    print(f"  Expected direction: [{push_direction[0]:.2f}, {push_direction[1]:.2f}]")

    # Check if displacement matches push direction
    if np.linalg.norm(displacement) > 0.01:
        actual_dir = displacement / np.linalg.norm(displacement)
        alignment = np.dot(actual_dir, push_direction)
        print(f"  Direction alignment: {alignment:.2f} (1.0 = perfect)")
        if alignment > 0.7:
            print(f"  ✓ SUCCESS - Bottle moved in correct direction!")
        elif alignment < -0.3:
            print(f"  ✗ FAILED - Bottle moved in OPPOSITE direction!")
        else:
            print(f"  ? PARTIAL - Bottle moved sideways")
    else:
        print(f"  ? Bottle barely moved")

    return displacement


# Test different push directions
print("\n" + "=" * 70)
print("ROBOT PUSH DIRECTION TEST")
print("Testing if robot can push bottle in all directions")
print("=" * 70)

# Define test directions
test_directions = [
    (np.array([0.0, -1.0]), "DOWN (-Y)"),  # Standard trajectory direction
    (np.array([0.0, 1.0]), "UP (+Y)"),  # Opposite
    (np.array([1.0, 0.0]), "RIGHT (+X)"),  # Away from robot
    (np.array([-1.0, 0.0]), "LEFT (-X)"),  # Toward robot
    (np.array([0.707, -0.707]), "DOWN-RIGHT"),  # Diagonal
    (np.array([-0.707, -0.707]), "DOWN-LEFT"),  # Diagonal toward robot
]

results = {}
for direction, name in test_directions:
    try:
        displacement = run_push_test(direction, name)
        results[name] = displacement
    except Exception as e:
        print(f"Error in {name}: {e}")
        results[name] = None

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for name, disp in results.items():
    if disp is not None:
        print(f"{name:15s}: displacement = [{disp[0] * 100:+6.1f}, {disp[1] * 100:+6.1f}] cm")
    else:
        print(f"{name:15s}: FAILED")

print("\nIf LEFT (-X) shows positive X displacement or DOWN shows positive Y,")
print("the robot geometry/kinematics may be preventing those motions.")