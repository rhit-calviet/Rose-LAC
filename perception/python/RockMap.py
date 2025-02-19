class RockMap:
    def __init__(self, agent, depth_map, rock_detector):
        """
        Initializes the RockMap.
        :param agent: The AutonomousAgent instance to access the geometric map.
        :param depth_map: An instance of DepthMap to determine depth of rocks.
        :param rock_detector: An instance of the rock detection model.
        """
        self.geometric_map = agent.get_geometric_map()
        self.depth_map = depth_map
        self.rock_detector = rock_detector
        self.cell_size = 0.15 
        self.rock_positions = []  # List to store rock positions as vectors
        # Currently stored in a 3D vector of (x-coord, y-coord, depth)

    def map_rocks(self, left_image, right_image):
        """
        Maps detected rocks onto the geometric map and stores their relative positions.
        :param left_image: Left stereo image for rock detection and depth mapping.
        :param right_image: Right stereo image for depth mapping.
        """
        # Detect rocks in the left image
        rock_detections = self.rock_detector.get_centroids_with_variance(left_image)

        # Compute the disparity map (which contains depth in meters)
        depth_map = self.depth_map.compute(left_image, right_image)

        for (x, y), variance in rock_detections:
            if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                depth = depth_map[y, x]  # Extract depth directly

                if depth == 0:  # Ignore invalid points
                    continue

                # Convert pixel coordinates to real-world coordinates
                x_real = x * self.cell_size
                y_real = y * self.cell_size

                # Store rock as a vector
                self.rock_positions.append((x_real, y_real, depth))

                # Store in the geometric map, only if we need it
                x_idx, y_idx = self.geometric_map.get_cell_indexes(x_real, y_real)
                self.geometric_map.set_cell_rock(x_idx, y_idx, True)
                self.geometric_map.set_cell_distance(x_idx, y_idx, depth)

        return self.rock_positions
