import cv2
import numpy as np
import os

def load_markers(marker_folder):
    """Loads marker images from subfolders (A, B, C, D), each containing 4 images."""
    markers = {}
    for category in ['A', 'B', 'C', 'D', 'Charger']:
        category_path = os.path.join(marker_folder, category)
        if os.path.isdir(category_path):
            markers[category] = [cv2.imread(os.path.join(category_path, img), 0) 
                                 for img in sorted(os.listdir(category_path)) 
                                 if img.endswith(('.png', '.jpg', '.jpeg'))]
    return markers

def find_marker(image):
    """Detects a marker in the given image and extracts it."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:  # Assuming marker is a quadrilateral
            pts = np.array([point[0] for point in approx], dtype="float32")
            return four_point_transform(image, pts)
    
    return None

def four_point_transform(image, pts):
    """Applies a perspective transform to extract the marker."""
    rect = np.array(sorted(pts, key=lambda x: (x[1], x[0])))
    (tl, tr, br, bl) = rect
    
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (int(width), int(height)))

def split_marker(marker):
    """Splits the extracted marker into 4 sub-images assuming a 2x2 grid."""
    h, w = marker.shape[:2]
    return [
        marker[0:h//2, 0:w//2],
        marker[0:h//2, w//2:w],
        marker[h//2:h, 0:w//2],
        marker[h//2:h, w//2:w]
    ]

def match_marker(extracted_parts, markers):
    """Matches extracted marker parts with stored reference markers."""
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match = None
    best_score = 0
    
    for category, ref_images in markers.items():
        match_score = 0
        for i, ref_img in enumerate(ref_images):
            keypoints1, descriptors1 = orb.detectAndCompute(extracted_parts[i], None)
            keypoints2, descriptors2 = orb.detectAndCompute(ref_img, None)
            if descriptors1 is None or descriptors2 is None:
                continue
            matches = bf.match(descriptors1, descriptors2)
            match_score += len(matches)
        
        if match_score > best_score:
            best_score = match_score
            best_match = category
    
    return best_match

if __name__ == "__main__":
    marker_folder = "docs"  # Parent folder containing A, B, C, D subfolders
    markers = load_markers(marker_folder)
    
    test_image_path = "test_image.jpg"  # Path to input image containing one marker
    test_image = cv2.imread(test_image_path)
    
    if test_image is None:
        print("Error: Could not load test image.")
    else:
        extracted_marker = find_marker(test_image)
        if extracted_marker is not None:
            extracted_parts = split_marker(extracted_marker)
            matched_marker = match_marker(extracted_parts, markers)
            print(f"Detected Marker: {matched_marker}")
        else:
            print("No marker detected.")
