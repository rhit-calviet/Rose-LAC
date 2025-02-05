import cv2
import numpy as np
from fiducials import *

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

path = "C:/Users/calviet/git/Rose-LAC/perception/python/baba.jpg"

# Get image dimensions for camera matrix
image = cv2.imread(path)
if image is None:
    raise FileNotFoundError(f"Could not load image at {path}")
h, w = image.shape[:2]

# Get detected fiducials
fiducial_group, centers = fiducials(path)

print('Fiducial Group: ' + fiducial_group)
for tag_id, (center_x, center_y) in centers:
    print(f"Tag ID {tag_id}: Center at ({center_x}, {center_y})")

# Get corresponding 3D-2D point pairs
object_points = []
image_points = []
for tag_id, (x_pixel, y_pixel) in centers:
    if tag_id in TAG_COORDINATES_LANDER:
        # Use fiducial coordinates directly (assume they are in global frame)
        global_point = TAG_COORDINATES_LANDER[tag_id]
        object_points.append(global_point)
        image_points.append([x_pixel, y_pixel])

# Calculate camera matrix (assuming 70° horizontal FOV)
fov_x = 1.22  # radians (70 degrees)
focal_length_x = w / (2 * np.tan(fov_x / 2))
focal_length_y = focal_length_x  # Square pixels
camera_matrix = np.array([
    [focal_length_x, 0, w/2],
    [0, focal_length_y, h/2],
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
print("\nVectors from camera to fiducials (camera local frame):")
for i, vector in enumerate(camera_local_vectors):
    print(f"Fiducial {i+1}: ({vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f})")