import cv2
import numpy as np
import sys
sys.path.insert(1, 'Localization')
from InitialPosition import InitialPosition

class LocalCoordinates:
    def __init__(self, image):
        self.image = image
        self.h, self.w = image.shape[:2]
        init_pos = InitialPosition()
        self.FIDUCIAL_TAG_COORDINATES = init_pos.get_fiducial_world_coordinates_with_corners()
        

    def detect_fiducial(self, image):
        """Detect fiducial using OpenCV's AprilTag detector."""
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect AprilTags
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) < 1:
            return None  # Require exactly 4 tags
        
        # Extract tag info (corners and IDs)
        tag_info = list(zip(corners, ids.flatten()))
        
        return tag_info

    def calculate_tag_centers(self, tag_info):
        """Calculate the center pixel coordinates of each detected tag."""
        centers = []
        for corners, tag_id in tag_info:
            # corners is an array of shape (1, 4, 2); we need to reshape it to (4, 2)
            corners = corners.reshape(4, 2)
            # Calculate the center as the average of the corner points
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            centers.append((tag_id, (center_x, center_y)))
        return centers


    def get_fiducials(self, image):
        tag_info = self.detect_fiducial(image)

        if tag_info is None:
            return None, None
        
        centers = self.calculate_tag_centers(tag_info)
        return centers
        

    def get_coordinates(self):
        # Get detected fiducials
        points = self.get_fiducials(self.image)

        if points is None:
            return None  # No fiducials detected

        # Get corresponding 3D-2D point pairs
        object_points = []
        image_points = []
        for tag_id, (x_pixel, y_pixel) in points:
            if tag_id in self.FIDUCIAL_TAG_COORDINATES:
                tag_data = self.FIDUCIAL_TAG_COORDINATES[tag_id]
        
                # Add center point
                center_global = tag_data["center"]
                object_points.append(center_global)
                image_points.append([x_pixel, y_pixel])
        
                # Add corner points
                corners_global = tag_data["corners"]
                for corner_global in corners_global:
                    object_points.append(corner_global)
                    image_points.append([x_pixel, y_pixel]) 

        # Calculate camera matrix (assuming 70° horizontal FOV)
        fov_x = 1.22  # radians (70 degrees)
        focal_length_x = self.w / (2 * np.tan(fov_x / 2))
        focal_length_y = focal_length_x  # Square pixels
        camera_matrix = np.array([
            [focal_length_x, 0, self.w/2],
            [0, focal_length_y, self.h/2],
            [0, 0, 1]
        ], dtype=np.float32)

        # Solve PnP (Perspective-n-Point)
        _, rvec, tvec = cv2.solvePnP(
            np.array(object_points, dtype=np.float32),
            np.array(image_points, dtype=np.float32),
            camera_matrix,
            None  # No distortion (documentation states perfect pinhole)
        )

        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)

        # Calculate camera position in global coordinates
        #camera_position_global = -np.matrix(R).T @ np.matrix(tvec)

        # Transform fiducial points to camera's local coordinate system
        camera_local_vectors = []
        for global_point in object_points:
            # Transform global point to camera's local frame
            global_point_homogeneous = np.array([global_point[0], global_point[1], global_point[2], 1])
            transform_matrix = np.vstack([np.hstack([R, tvec]), [0, 0, 0, 1]])
            camera_local_point = transform_matrix.T @ global_point_homogeneous
            camera_local_vectors.append(camera_local_point[:3].flatten())  # Flatten to 1D array


        coordinates = np.array(object_points, dtype=np.float32)  # Convert to ndarray
        for i, vector in enumerate(camera_local_vectors):
            camera_local_vectors[i] = (vector[2], -vector[0], -vector[1])
        vectors = np.array(camera_local_vectors, dtype=np.float32)  # Convert to ndarray
       
        
        # Estimate variance as the mean squared error of projection errors
        projected_points, _ = cv2.projectPoints(np.array(object_points, dtype=np.float32),
                                                rvec, tvec, camera_matrix, None)
        projected_points = projected_points.reshape(-1, 2)
        image_points_np = np.array(image_points, dtype=np.float32)

        residuals = np.linalg.norm(image_points_np - projected_points, axis=1)
        variance = float(np.var(residuals))  # Convert variance to float

        #print (list(zip(coordinates, vectors, [variance] * len(coordinates))))
        return list(zip(coordinates, vectors, [variance] * len(coordinates)))  # Return as a list of tuples


if __name__ == '__main__':
    image = cv2.imread('C:/Users/beaslebf/Projects/Rose-LAC/perception/python/test_images/fiducials_test.jpeg')
    local_coordinates = LocalCoordinates(image)
    print(local_coordinates.get_coordinates())