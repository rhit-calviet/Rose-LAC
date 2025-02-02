import cv2
import numpy as np

# Known AprilTag IDs for each fiducial (update with your actual IDs)
FIDUCIAL_TAG_IDS = {
    "A": [243, 71, 462, 37],   # Example IDs for Fiducial A
    "B": [0, 3, 2, 1],   # Example IDs for Fiducial B
    "C": [10, 11, 8, 9],
    "D": [464, 459, 258, 5],
}


def order_tags(tag_info):
    """Order detected tags into [top-left, top-right, bottom-left, bottom-right]."""
    centers = np.array([np.mean(corners, axis=1).flatten() for (corners, _) in tag_info])

    # Ensure centers is a proper 2D array of shape (n, 2)
    centers = np.array(centers, dtype=np.float32)

    # Sort by y-coordinate to split into top and bottom rows
    sorted_indices = np.argsort(centers[:, 1])  # Sort by Y values
    top_indices = sorted_indices[:2].tolist()
    bottom_indices = sorted_indices[2:].tolist()

    # Ensure sorting keys return scalars
    top_indices = sorted(top_indices, key=lambda i: centers[i, 0].item())   # Convert to scalar
    bottom_indices = sorted(bottom_indices, key=lambda i: centers[i, 0].item())

    # Convert NumPy indices to Python integers
    ordered_indices = [int(idx) for idx in top_indices + bottom_indices]
    
    return ordered_indices




def detect_fiducial(image):
    """Detect fiducial using OpenCV's AprilTag detector."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Detect AprilTags
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) != 4:
        return None  # Require exactly 4 tags
    
    # Extract tag info (corners and IDs)
    tag_info = list(zip(corners, ids.flatten()))
    
    # Order tags spatially
    ordered_indices = order_tags(tag_info)
    ordered_ids = [tag_info[i][1] for i in ordered_indices]
    
    # Match against known configurations
    for fiducial, expected_ids in FIDUCIAL_TAG_IDS.items():
        if ordered_ids == expected_ids:
            return fiducial
    return None

# Example usage:
if __name__ == "__main__":
    image = cv2.imread("C:/Users/beaslebf/Projects/Rose-LAC/perception/python/baba.jpg")
    if image is None:
        print("Error: Could not load image.")
    else:
        fiducial = detect_fiducial(image)
        print(f"Detected Fiducial: {fiducial}")