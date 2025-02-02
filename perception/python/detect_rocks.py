import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Code to run the model saved as a .pt file
# Need to have the ultralytics library installed. The model is a yolov8n rock detection model, with frozen layers trained on moon rock images

# Load trained YOLO model
model = YOLO('last.pt')  # Whatever path is needed to the model

def confidence_to_variance(conf, max_variance=30):
    """
    Converts confidence into variance (higher confidence → lower variance).
    
    Parameters:
        conf (float): Confidence score (between 0 and 1).
        max_variance (int): Maximum allowed variance in pixels.
    
    Returns:
        variance (float): Computed variance in pixels.
    """
    return max_variance * (1 - conf)

def get_centroids_with_variance(image_path):
    """
    Runs YOLO on an image and returns an array of centroids with variance.
    
    Parameters:
        image_path (str): Path to the input image.

    Returns:
        results_list (list of tuples): List of [(x, y), variance] for detected objects.
    """
    results = model(image_path)
    results_list = []  # Store results

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            conf = box.conf[0].item()  # Confidence score
            variance = confidence_to_variance(conf)  # Compute variance
            
            results_list.append([(centroid_x, centroid_y), variance])

    return results_list


# Sample Run:
# image_path = "/images/robot/Left_33.95.jpeg"

# Get centroids
# centroids_variance = get_centroids_with_variance(image_path)
# print("Centroids and Variance:", centroids_variance)