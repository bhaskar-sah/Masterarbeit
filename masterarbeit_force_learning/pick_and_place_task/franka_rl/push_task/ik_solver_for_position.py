import mujoco
import numpy as np
import os

# --- 1. LOAD MODEL ---
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "panda_robot_push.xml")

if not os.path.exists(xml_path):
    print(f"Error: Could not find {xml_path}")
    exit(1)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# --- 2. CONFIGURATION ---
target_pos = np.array([0.4, 0.38, 0.965])

# Target: DOWNWARD FACING
# [0, 1, 0, 0] = 180 deg around X (Flips Z-up to Z-down)
target_quat = np.array([0, 0.707107, 0.707107, 0])

# Reset simulation
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# --- 3. DEFINE INDICES ---
site_id = model.site("hand_center").id
joint_names = [f"joint{i}" for i in range(1, 8)]

qpos_ids = [model.joint(name).qposadr[0] for name in joint_names]
dof_ids = [model.joint(name).dofadr[0] for name in joint_names]
jnt_ids = [model.joint(name).id for name in joint_names]

# --- 3b. SET INITIAL GUESS (CRITICAL FIX) ---
# We force the robot into a "Ready" pose (Elbow bent up) before solving.
# This prevents the "twisted/snake" solutions.
# Values are approx: [0, -45, 0, -135, 0, 90, 45] degrees
start_pose = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

data.qpos[qpos_ids] = start_pose
mujoco.mj_forward(model, data)
print("Initialized robot to 'Ready' pose to guide IK.")

# --- 4. IK OPTIMIZATION LOOP ---
step_size = 0.5
tol = 1e-4
max_iter = 1000

print(f"Running IK... Target: {target_pos}")

jac = np.zeros((6, model.nv))
err_quat = np.zeros(3)

for i in range(max_iter):
    # 1. Calculate Errors
    curr_pos = data.site_xpos[site_id]
    curr_mat = data.site_xmat[site_id]

    err_pos = curr_pos - target_pos

    curr_quat = np.zeros(4)
    mujoco.mju_mat2Quat(curr_quat, curr_mat)
    mujoco.mju_negQuat(curr_quat, curr_quat)
    mujoco.mju_subQuat(err_quat, target_quat, curr_quat)

    # 2. COMBINE ERROR
    # We increase rotational weight (1.0) to ensure it points strictly down
    error = np.hstack((err_pos, err_quat * 1.0))

    if np.linalg.norm(error) < tol:
        print(f"Converged in {i} steps!")
        break

    # 3. JACOBIAN
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    jac_robot = jac[:, dof_ids]

    # 4. SOLVE
    # We dampen slightly more (1e-2) for stability
    dq_robot = -np.linalg.pinv(jac_robot, rcond=1e-2) @ error

    # 5. UPDATE
    data.qpos[qpos_ids] += dq_robot * step_size

    # 6. LIMITS
    for j, q_idx in enumerate(qpos_ids):
        limit_idx = jnt_ids[j]
        min_lim = model.jnt_range[limit_idx, 0]
        max_lim = model.jnt_range[limit_idx, 1]
        data.qpos[q_idx] = np.clip(data.qpos[q_idx], min_lim, max_lim)

    mujoco.mj_forward(model, data)

# --- 5. OUTPUT ---
bottle_qpos = data.qpos[0:7]
bottle_str = " ".join([f"{x:.4f}" for x in bottle_qpos])

robot_vals = data.qpos[qpos_ids]
robot_str = " ".join([f"{x:.4f}" for x in robot_vals])

fingers_str = "0.0 0.0"

print("\n" + "=" * 60)
print("UPDATED XML KEYFRAME:")
print("=" * 60)
print(f'qpos="{bottle_str} {robot_str} {fingers_str}"')
print(f'ctrl="{robot_str} 0"')
print("=" * 60)

# --- VERIFICATION STEP ---
final_pos = data.site_xpos[site_id]
final_error = np.linalg.norm(final_pos - target_pos)

print("\n" + "="*30)
print("FINAL VERIFICATION")
print("="*30)
print(f"Target Position: {target_pos}")
print(f"Actual Position: {final_pos}")
print(f"Distance Error:  {final_error:.6f} m")  # Should be close to 0.000000

if final_error < 0.005: # Less than 5mm error
    print("SUCCESS: Robot is at the target!")
else:
    print("WARNING: Robot is slightly off target.")
print("="*30)