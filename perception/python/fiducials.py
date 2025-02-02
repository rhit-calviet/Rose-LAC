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
    centers = [np.mean(corners, axis=0) for (corners, _) in tag_info]
    centers = np.array(centers)
    
    # Sort by y-coordinate to split into top and bottom rows
    sorted_indices = np.argsort(centers[:, 1])
    top_indices = sorted_indices[:2]
    bottom_indices = sorted_indices[2:]
    
    # Sort top and bottom rows by x-coordinate
    top_indices = top_indices[np.argsort(centers[top_indices, 0])]
    print(top_indices)
    bottom_indices = bottom_indices[np.argsort(centers[bottom_indices, 0])]
    
    # Convert NumPy indices to Python integers
    ordered_indices = (
        [int(idx) for idx in top_indices] + 
        [int(idx) for idx in bottom_indices]
    )
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
    image = cv2.imread("C:/Users/calviet/git/Rose-LAC/perception/python/baba.jpg")
    if image is None:
        print("Error: Could not load image.")
    else:
        fiducial = detect_fiducial(image)
        print(f"Detected Fiducial: {fiducial}")