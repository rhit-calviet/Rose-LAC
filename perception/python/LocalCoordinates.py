import cv2
import numpy as np
import sys
import os
sys.path.insert(1, 'Localization')

from source_code.Localization.InitialPosition import InitialPosition
import pupil_apriltags as apriltag

print(os.getcwd())

class LocalCoordinates:
    def __init__(self, image, rover, lander):
        self.image = image
        self.h, self.w = image.shape[:2]
        init_pos = InitialPosition(rover, lander)
        self.FIDUCIAL_TAG_COORDINATES = init_pos.get_fiducial_world_coordinates_with_corners()
        
        self.FIDUCIAL_TAG_IDS = { # Known AprilTag IDs for each fiducial
            "A": [243, 71, 462, 37],   # IDs for Fiducial A
            "B": [0, 3, 2, 1],         # IDs for Fiducial B
            "C": [10, 11, 8, 9],       # IDs for Fiducial C
            "D": [464, 459, 258, 5],   # IDs for Fiducial D
            "Charger": [69]}          # ID for Charger Fiducial
        
    def preprocess_image(self, image):
        """Enhance image quality to improve AprilTag detection."""
        gray = cv2.GaussianBlur(image, (3, 3), 0.5)  # Reduce noise
        sharp = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 2), -0.5, 0)  # Sharpening
        return sharp
    
    def detect_apriltags(self, image):
        """Detect AprilTags in an image and return their pixel coordinates."""
        if image is None:
            raise ValueError("Error loading image. Check the path.")

        image = image.astype(np.uint8)
        image = self.preprocess_image(image)
        # print(f"DEBUG: bark")
        detector = apriltag.Detector()
        # print(f"DEBUG: furry")
        image = cv2.resize(image, (720, 480))
        results = detector.detect(image)
        # print(f"DEBUG: roar")
    
        centers = []
        tag_ids = []
        # print(f"DEBUG: results before for loop-{results}")
        # if results is None:
        #      return [], []
        # if not isinstance(results, list):
        #     print("Is not detection")
        #     return [], []
        
        # print(f"DEBUG: results after if-{len(results)}")
        
        for tag in results:
            # print(f"DEBUG: results during for-{tag}")

            center_x, center_y = int(tag.center[0]), int(tag.center[1])
            tag_ids.append(tag.tag_id)  # Get the tag ID
            centers.append((center_x, center_y))

        # print(f"DEBUG: results after for-{len(results)}")
   
        return centers, tag_ids

    def detect_fiducial(self, image):

        # Convert to grayscale and preprocess
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # print(f'DETECT: meow')

        # Detect AprilTags
        centers, ids = self.detect_apriltags(gray)

        # print(f'DEBUG: after return')

        print(f"DEBUG: detected corners-{centers}, detected IDs-{ids}")

        if ids is None or len(ids) < 1:
            return None  # Require at least 4 tags
        
        # Extract tag info (corners and IDs)
        tag_info = list(zip(ids, centers))
        
        return tag_info

    def get_fiducials(self, image):
        tag_info = self.detect_fiducial(image)

        if tag_info is None:
            return None
        
        return tag_info
        

    def get_coordinates(self):
        # Get detected fiducials

        points = self.get_fiducials(self.image)

        # print(f'DEBUG: trash0')

        if points is None:
            return None  # No fiducials detected

        # Get corresponding 3D-2D point pairs
        object_points = []
        image_points = []
        # print(f"DEBUG: centers-{centers}")
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

        # print(f'DEBUG: trash1')


        # Calculate camera matrix (assuming 70° horizontal FOV)
        fov_x = 1.22  # radians (70 degrees)
        focal_length_x = self.w / (2 * np.tan(fov_x / 2))
        focal_length_y = focal_length_x  # Square pixels
        camera_matrix = np.array([
            [focal_length_x, 0, self.w/2],
            [0, focal_length_y, self.h/2],
            [0, 0, 1]
        ], dtype=np.float32)

        # print(f'DEBUG: trash2')

        # Solve PnP (Perspective-n-Point)
        _, rvec, tvec = cv2.solvePnP(
            np.array(object_points, dtype=np.float32),
            np.array(image_points, dtype=np.float32),
            camera_matrix,
            None  # No distortion (documentation states perfect pinhole)
        )

        # print(f'DEBUG: trash3')

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

        # print(f'DEBUG: trash4')

        coordinates = np.array(object_points, dtype=np.float32)  # Convert to ndarray
        for i, vector in enumerate(camera_local_vectors):
            camera_local_vectors[i] = (vector[2], -vector[0], -vector[1])
        vectors = np.array(camera_local_vectors, dtype=np.float32)  # Convert to ndarray
       
        # print(f'DEBUG: trash5')

        # Estimate variance as the mean squared error of projection errors
        projected_points, _ = cv2.projectPoints(np.array(object_points, dtype=np.float32),
                                                rvec, tvec, camera_matrix, None)
        projected_points = projected_points.reshape(-1, 2)
        image_points_np = np.array(image_points, dtype=np.float32)

        # print(f'DEBUG: trash6')

        residuals = np.linalg.norm(image_points_np - projected_points, axis=1)
        variance = float(np.var(residuals))  # Convert variance to float

        # print(f'DEBUG: trash7')

        #print (list(zip(coordinates, vectors, [variance] * len(coordinates))))
        return list(zip(coordinates, vectors, [variance] * len(coordinates)))  # Return as a list of tuples


if __name__ == '__main__':
    img_path = "/root/Rose-LAC/LunarAutonomyChallenge/debug_output/Right_35.05.jpeg"
    # image = cv2.imread('C:/Users/beaslebf/Projects/Rose-LAC/perception/python/test_images/fiducials_test.jpeg')
    image = cv2.imread(img_path)
    local_coordinates = LocalCoordinates(image)