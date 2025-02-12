class InitialPosition:

    def __init__(self):
        pass

    # Known tag coordinates (in lander's local frame F_L)
    TAG_COORDINATES_LANDER = {
        243: (0.691, -1.033, 0.894), 71: (1.033, -0.691, 0.894),
        462: (0.691, -1.033, 0.412), 37: (1.033, -0.691, 0.412),
        0: (1.033, 0.691, 0.894), 3: (0.691, 1.033, 0.894),
        2: (1.033, 0.691, 0.412), 1: (0.691, 1.033, 0.412),
        10: (-0.691, 1.033, 0.894), 11: (-1.033, 0.691, 0.894),
        8: (-0.691, 1.033, 0.412), 9: (-1.033, 0.691, 0.412),
        464: (-1.033, -0.691, 0.894), 459: (-0.691, -1.033, 0.894),
        258: (-1.033, -0.691, 0.412), 5: (-0.691, -1.033, 0.412)
    }


    def get_initial_lander_world_position(self):
    # Get the rover's transform in the world frame
        rover_transform = get_initial_position()

        # Get the lander's transform relative to the rover
        lander_relative = get_initial_lander_position()

        # Extract the lander's local coordinates
        lx = lander_relative.location.x
        ly = lander_relative.location.y
        lz = lander_relative.location.z

        # Apply the rover's transform to get the lander's world coordinates
        lander_world = rover_transform.transform(lx, ly, lz)

        # The result is the lander's position in the world coordinate system
        return lander_world.x, lander_world.y, lander_world.z


    def get_fiducial_world_coordinates(self):
        # Get the lander's world position
        lander_x, lander_y, lander_z = self.get_initial_lander_world_position()

        # Dictionary to store the updated fiducial positions
        fiducial_world_coords = {}

        for tag_id, (fx, fy, fz) in self.TAG_COORDINATES_LANDER.items():
            # Translate fiducial coordinates to world coordinates
            fiducial_world_coords[tag_id] = (lander_x + fx, lander_y + fy, lander_z + fz)

        return fiducial_world_coords

