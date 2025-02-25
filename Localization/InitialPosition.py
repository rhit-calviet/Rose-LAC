import math
import numpy as np


class InitialPosition:
    
    TAG_SIZE = 0.339

    def __init__(self, rover_transform, lander_relative_transform):
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

        self.rover_transform = rover_transform
        self.lander_relative_transform = lander_relative_transform

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
        # Rover's world transform matrix (rover -> world)
        M_rover = self.create_transform(
            self.rover_transform.rotation,
            self.rover_transform.location
        )

        # Lander's relative transform matrix (lander -> rover)
        M_lander_relative = self.create_transform(
            self.lander_relative_transform.rotation,
            self.lander_relative_transform.location
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
    
    def get_fiducial_world_coordinates_with_corners(self):
        """Returns a dictionary with tag IDs mapped to both center and corners in world coordinates."""
        world_coords = {}
        half_size = self.TAG_SIZE / 2

        # Define the rotation angles for each fiducial group
        group_rotations = {
            "A": np.deg2rad(135),
            "B": np.deg2rad(45),
            "C": np.deg2rad(315),
            "D": np.deg2rad(225)
        }

        # Assign tags to their respective groups
        group_mapping = {
            "A": [243, 71, 462, 37],
            "B": [3, 1, 2, 0],
            "C": [10, 11, 9, 8],
            "D": [464, 459, 258, 5]
        }

        # Map each tag ID to its corresponding rotation angle
        tag_to_rotation = {}
        for group, tags in group_mapping.items():
            for tag_id in tags:
                tag_to_rotation[tag_id] = group_rotations[group]

        for tag_id, (cx, cy, cz) in self.fiducials_tag_coordinates.items():
            # Center in world coordinates
            center_world = (cx, cy, cz)

            # Determine rotation angle for this tag
            rotation_angle = tag_to_rotation.get(tag_id, 0)  # Default to 0 if not found

            # Rotation matrix for the XY plane
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])

            # Corners in the local tag frame (relative to the center)
            local_corners = np.array([
                [-half_size, -half_size],  # Top-left
                [half_size, -half_size],   # Top-right
                [-half_size, half_size],   # Bottom-left
                [half_size, half_size]     # Bottom-right
            ])

            # Rotate corners in 2D space
            rotated_corners = np.dot(local_corners, rotation_matrix.T)

            # Translate to world coordinates (keeping the original Z coordinate)
            world_corners = [(cx + x, cy + y, cz) for x, y in rotated_corners]

            # Add both center and corners to the dictionary
            world_coords[tag_id] = {
                "center": center_world,
                "corners": world_corners
            }

        return world_coords

