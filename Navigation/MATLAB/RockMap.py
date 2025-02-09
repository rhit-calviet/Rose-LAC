import numpy as np
import cv2 as cv

class RockMap:
    def __init__(self, agent, depth_map, rock_detector, focal_length, baseline):
        """
        Initializes the RockMap.
        :param agent: The AutonomousAgent instance to access the geometric map.
        :param depth_map: An instance of DepthMap to determine depth of rocks.
        :param rock_detector: An instance of the rock detection model.
        :param focal_length: The focal length of the camera (in pixels).
        :param baseline: The baseline (distance between stereo cameras) in meters.
        """
        self.geometric_map = agent.get_geometric_map()
        self.depth_map = depth_map
        self.rock_detector = rock_detector
        self.cell_size = 0.15  # 15 cm per cell
        self.focal_length = focal_length
        self.baseline = baseline
    
    def map_rocks(self, left_image, right_image):
        """
        Maps detected rocks onto the geometric map.
        :param left_image: Left stereo image for rock detection and depth mapping.
        :param right_image: Right stereo image for depth mapping.
        """
        # Run rock detection on the left image
        rock_detections = self.rock_detector.get_centroids_with_variance(left_image)
        
        # Compute disparity map using left and right images
        disparity_map = self.depth_map.compute(left_image, right_image)
        
        for (x, y), variance in rock_detections:
            # Get disparity and compute depth (distance)
            if 0 <= y < disparity_map.shape[0] and 0 <= x < disparity_map.shape[1]:
                disparity = disparity_map[y, x]
                
                # If disparity is zero (i.e., no matching points), skip this point
                if disparity == 0:
                    continue
                
                # Calculate depth (distance from the camera to the rock)
                depth = (self.focal_length * self.baseline) / disparity
                
                # Convert pixel coordinates (x, y) to real-world coordinates (x_real, y_real)
                x_real = (x * self.cell_size * depth) / self.focal_length
                y_real = (y * self.cell_size * depth) / self.focal_length
                
                # Convert real-world coordinates to map grid coordinates
                x_idx, y_idx = self.geometric_map.get_cell_indexes(x_real, y_real)
                
                # Set rock presence in the appropriate map cell
                self.geometric_map.set_cell_rock(x_idx, y_idx, True)
                
                # Set estimated distance to the rock in the map (can be used as height or distance)
                self.geometric_map.set_cell_distance(x_idx, y_idx, depth)
            
    def visualize_map(self):
        """
        Prints a simple text-based representation of the rock map.
        """
        map_array = self.geometric_map.get_map_array()
        for row in map_array:
            print("".join("R" if cell else "." for cell in row))
