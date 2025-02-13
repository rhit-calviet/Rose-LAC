import cv2
import numpy as np
import sys
from detect_fiducials import *
sys.path.insert(1, 'Localization')
from InitialPosition import InitialPosition

class LocalCoordinates:
    def __init__(self, image):
        self.image = image
        self.h, self.w = image.shape[:2]
        init_pos = InitialPosition()
        self.FIDUCIAL_TAG_COORDINATES = init_pos.get_fiducial_world_coordinates()
        

    def get_coordinates(self):
        # Get detected fiducials
        fiducial_group, centers = fiducials(self.image)

        if(centers is None):
            return None

        # print('Fiducial Group: ' + fiducial_group)
        # for tag_id, (center_x, center_y) in centers:
        #     print(f"Tag ID {tag_id}: Center at ({center_x}, {center_y})")

        # Get corresponding 3D-2D point pairs
        object_points = []
        image_points = []
        for tag_id, (x_pixel, y_pixel) in centers:
            if tag_id in self.FIDUCIAL_TAG_COORDINATES:
                # Use fiducial coordinates directly (assume they are in global frame)
                global_point = self.FIDUCIAL_TAG_COORDINATES[tag_id]
                object_points.append(global_point)
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
        camera_position_global = -np.matrix(R).T @ np.matrix(tvec)

        # Transform fiducial points to camera's local coordinate system
        camera_local_vectors = []
        for global_point in object_points:
            # Transform global point to camera's local frame
            global_point_homogeneous = np.array([global_point[0], global_point[1], global_point[2], 1])
            transform_matrix = np.vstack([np.hstack([R, tvec]), [0, 0, 0, 1]])
            camera_local_point = transform_matrix.T @ global_point_homogeneous
            camera_local_vectors.append(camera_local_point[:3].flatten())  # Flatten to 1D array

        # Output vectors in camera's local coordinate system
        # print("\nVectors from camera to fiducials (camera local frame):")
        # for i, vector in enumerate(camera_local_vectors):
        #     print(f"Fiducial {i+1}: ({vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f})")


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