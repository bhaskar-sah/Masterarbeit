# contact.py
"""
contact.py
Contact Manager for Panda Push Environment.

Handles contact detection and force computation between robot and bottle.
"""

import numpy as np
import mujoco


class ContactManager:
    """
    Handles contact detection and force computation between the robot and the bottle.
    """

    def __init__(self, model, data, robot_contact_bodies, bottle_body_id):
        """
        Initialize contact manager.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            robot_contact_bodies: Set of body IDs for robot contact parts
            bottle_body_id: Body ID of the bottle
        """
        self.model = model
        self.data = data
        self.robot_contact_bodies = robot_contact_bodies
        self.bottle_body_id = bottle_body_id

    def get_contact_force(self):
        """
        Get contact force between robot and bottle.

        Returns: Force vector [fx, fy, fz] in world frame
        """
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

    def is_touching(self):
        """
        Check if robot is touching bottle.

        Returns: True if contact exists, False otherwise
        """
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]

            robot_touch = body1 in self.robot_contact_bodies or body2 in self.robot_contact_bodies
            bottle_touch = body1 == self.bottle_body_id or body2 == self.bottle_body_id

            if robot_touch and bottle_touch:
                return True

        return False