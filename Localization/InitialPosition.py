import math
import numpy as np


class InitialPosition:

    def __init__(self):
        # Known tag coordinates (in lander's local frame F_L)
        self.fiducials_tag_coordinates = {
            243: (0.691, -1.033, 0.894), 71: (1.033, -0.691, 0.894),
            462: (0.691, -1.033, 0.412), 37: (1.033, -0.691, 0.412),
            0: (1.033, 0.691, 0.894), 3: (0.691, 1.033, 0.894),
            2: (1.033, 0.691, 0.412), 1: (0.691, 1.033, 0.412),
            10: (-0.691, 1.033, 0.894), 11: (-1.033, 0.691, 0.894),
            8: (-0.691, 1.033, 0.412), 9: (-1.033, 0.691, 0.412),
            464: (-1.033, -0.691, 0.894), 459: (-0.691, -1.033, 0.894),
            258: (-1.033, -0.691, 0.412), 5: (-0.691, -1.033, 0.412),
            69: (0.0, 0.662, 0.325)
        }

    # Helper functions to create rotation matrices
    def rotation_matrix_x(self, angle):
        return np.array([
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)]
        ])

    def rotation_matrix_y(self, angle):
        return np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)]
        ])

    def rotation_matrix_z(self, angle):
        return np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])

    # Construct 4x4 transformation matrix from rotation and translation
    def create_transform(self, rotation, translation):
        # Extract angles (roll-X, pitch-Y, yaw-Z)
        roll = rotation.roll
        pitch = rotation.pitch
        yaw = rotation.yaw

        # Compute individual rotation matrices
        Rx = self.rotation_matrix_x(roll)
        Ry = self.rotation_matrix_y(pitch)
        Rz = self.rotation_matrix_z(yaw)

        # Combine rotations: R = Rz * Ry * Rx (applied in order X -> Y -> Z)
        R = Rz @ Ry @ Rx

        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [translation.x, translation.y, translation.z]
        return T


    def get_initial_lander_world_position(self):
        # Get initial transforms
        rover_transform = get_initial_position()
        lander_relative_transform = get_initial_lander_position()

        # Rover's world transform matrix (rover -> world)
        M_rover = self.create_transform(
            rover_transform.rotation,
            rover_transform.location
        )

        # Lander's relative transform matrix (lander -> rover)
        M_lander_relative = self.create_transform(
            lander_relative_transform.rotation,
            lander_relative_transform.location
        )

        # Compute lander's world transform (lander -> world)
        M_lander_world = M_rover @ M_lander_relative

        # Extract lander's world coordinates from the final matrix
        lander_world_x = M_lander_world[0, 3]
        lander_world_y = M_lander_world[1, 3]
        lander_world_z = M_lander_world[2, 3]

        # Translate fiducial coordinates to world coordinates
        for tag_id, (fx, fy, fz) in self.fiducials_tag_coordinates.items():
            self.fiducials_tag_coordinates[tag_id] = (lander_world_x + fx, lander_world_y + fy, lander_world_z + fz)

        return lander_world_x, lander_world_y, lander_world_z


    def get_fiducial_world_coordinates(self):
        # return fiducial_world_coords
        return self.fiducials_tag_coordinates

