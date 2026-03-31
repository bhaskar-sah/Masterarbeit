import numpy as np
import mujoco


def is_touching(model, data, bottle_body_id, robot_contact_bodies):
    """
    Check whether the robot is in contact with the bottle.
    """
    for i in range(data.ncon):
        contact = data.contact[i]

        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]

        robot_touch = (body1 in robot_contact_bodies) or (body2 in robot_contact_bodies)
        bottle_touch = (body1 == bottle_body_id) or (body2 == bottle_body_id)

        if robot_touch and bottle_touch:
            return True

    return False


def get_contact_force(model, data, bottle_body_id, robot_contact_bodies):
    """
    Compute total contact force between robot and bottle in world frame.
    Returns a 3D force vector.
    """
    total_force = np.zeros(3, dtype=np.float32)

    for i in range(data.ncon):
        contact = data.contact[i]

        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]

        robot_touch = (body1 in robot_contact_bodies) or (body2 in robot_contact_bodies)
        bottle_touch = (body1 == bottle_body_id) or (body2 == bottle_body_id)

        if robot_touch and bottle_touch:
            c_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, c_force)

            frame = contact.frame.reshape(3, 3)
            force_world = frame.T @ c_force[:3]

            total_force += force_world.astype(np.float32)

    return total_force