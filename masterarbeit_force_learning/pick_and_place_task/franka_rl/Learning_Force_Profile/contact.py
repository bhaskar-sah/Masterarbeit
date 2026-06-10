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

    # def get_contact_force(self):
    #     """
    #     Get contact force between robot and bottle.

    #     Returns: Force vector [fx, fy, fz] in world frame
    #     """
    #     total_force = np.zeros(3, dtype=np.float32)

    #     for i in range(self.data.ncon):
    #         contact = self.data.contact[i]
    #         body1 = self.model.geom_bodyid[contact.geom1]
    #         body2 = self.model.geom_bodyid[contact.geom2]

    #         robot_touch = body1 in self.robot_contact_bodies or body2 in self.robot_contact_bodies
    #         bottle_touch = body1 == self.bottle_body_id or body2 == self.bottle_body_id

    #         if robot_touch and bottle_touch:
    #             c_force = np.zeros(6)
    #             mujoco.mj_contactForce(self.model, self.data, i, c_force)
    #             frame = contact.frame.reshape(3, 3)
    #             force_world = frame.T @ c_force[:3]
    #             total_force += force_world.astype(np.float32)

    #     return total_force

    def get_contact_force(self):
        """
        Get the total contact force the robot applies on the bottle,
        expressed in the world frame.
        
        MuJoCo convention: c_force is the force body2 exerts on body1,
        in the contact frame. We detect which body is which and flip sign
        as needed so the returned force is always: robot -> bottle (world frame).
        """
        total_force = np.zeros(3, dtype=np.float32)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            
            robot_is_body1 = body1 in self.robot_contact_bodies
            robot_is_body2 = body2 in self.robot_contact_bodies
            bottle_is_body1 = body1 == self.bottle_body_id
            bottle_is_body2 = body2 == self.bottle_body_id
            
            # Only consider robot-bottle contacts
            if not ((robot_is_body1 and bottle_is_body2) or
                    (robot_is_body2 and bottle_is_body1)):
                continue
            
            # Read contact force (in contact frame)
            c_force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, c_force)
            
            # Convert to world frame
            # contact.frame is row-major: rows are world-axis vectors expressed in contact frame
            # So frame.T gives us a matrix whose columns are world-axis vectors
            # frame.T @ contact_vector gives the vector in world frame
            frame = contact.frame.reshape(3, 3)
            force_world = frame.T @ c_force[:3]

            # # === DIAGNOSTIC ===
            # # The contact normal in world frame is frame[0, :] (first row)
            # # The normal component of contact force is c_force[0]
            # # So the normal force vector in world frame should be:
            # #     normal_force_world = c_force[0] * frame[0, :]
            # # Or equivalently: (frame.T @ c_force[:3])[0] times the normal axis
            # # 
            # # Let's compute both and print them to see which matches geometry
            
            # normal_in_world = frame[0, :]   # contact normal expressed in world
            # normal_force_magnitude = c_force[0]
            # normal_force_world_method1 = normal_force_magnitude * normal_in_world
            
            # force_world_methodT = frame.T @ c_force[:3]
            # force_world_methodNoT = frame @ c_force[:3]
            
            # print(f"\n--- Contact {i} ---")
            # print(f"Body1 = {body1}, Body2 = {body2}")
            # print(f"Bottle is body1: {bottle_is_body1}, Bottle is body2: {bottle_is_body2}")
            # print(f"Contact normal (world): {normal_in_world}")
            # print(f"c_force[:3] in contact frame: {c_force[:3]}")
            # print(f"Normal force only (method1): {normal_force_world_method1}")
            # print(f"All force, frame.T method: {force_world_methodT}")
            # print(f"All force, frame method:   {force_world_methodNoT}")
            # # === END DIAGNOSTIC ===
            
            # MuJoCo convention: c_force is force from body2 on body1
            # We want: force from robot on bottle
            if robot_is_body1 and bottle_is_body2:
                # body2 (bottle) pushes body1 (robot), so the returned force is bottle->robot
                # We want robot->bottle
                force_robot_on_bottle = force_world
            else:  # robot_is_body2 and bottle_is_body1
                # body2 (robot) pushes body1 (bottle), so returned force IS robot->bottle
                force_robot_on_bottle = -force_world
            
            total_force += force_robot_on_bottle.astype(np.float32)
        
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