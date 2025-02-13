import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

class RockDetector:
    def __init__(self, model_path='last.pt', conf_threshold=0.40):
        """
        Initializes the RockDetector with a YOLO model.

        Parameters:
            model_path (str): Path to the trained YOLO model file.
            conf_threshold (float): Minimum confidence threshold to keep detections.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def confidence_to_variance(self, conf):
        """
        Converts confidence into variance (higher confidence → lower variance).
        
        Parameters:
            conf (float): Confidence score (between 0 and 1).
        
        Returns:
            variance (float): Computed variance in pixels.
        """
        return (1 - conf) * 100  # Scale variance dynamically

    def get_centroids_with_variance(self, image_path):
        """
        Runs YOLO on an image and returns an array of centroids with variance.
        
        Parameters:
            image_path (str): Path to the input image.

        Returns:
            results_list (list of tuples): List of [(x, y), variance] for detected objects.
        """
        results = self.model(image_path)
        results_list = []

        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()  # Confidence score
                if conf >= self.conf_threshold:  # Filter by confidence threshold
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2
                    variance = self.confidence_to_variance(conf)  # Compute variance
                    
                    results_list.append([(centroid_x, centroid_y), variance])

        return results_list

    def show_image_with_boxes(self, image_path):
        """
        Runs YOLO on an image and displays it with bounding boxes, centroids, and variance.

        Parameters:
            image_path (str): Path to the input image.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        results = self.model(image_path)

        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                if conf >= self.conf_threshold:  # Only keep high-confidence detections
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2
                    variance = self.confidence_to_variance(conf)  # Compute variance
                    cls = int(box.cls[0])
                    label = f"{self.model.names[cls]}: {conf:.2f}, Var: ±{variance:.1f}px"

                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Draw centroid
                    cv2.circle(image, (centroid_x, centroid_y), 5, (0, 255, 0), -1)  # Green dot
                    cv2.putText(image, f"({centroid_x}, {centroid_y})", (centroid_x + 10, centroid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.show()

# EXAMPLE USAGE
# detector = RockDetector(model_path='/content/last.pt', conf_threshold=0.20)

# image_path = "/content/Left_33.95.jpeg"

# centroids_variance = detector.get_centroids_with_variance(image_path)
# print("Centroids and Variance:", centroids_variance)

# detector.show_image_with_boxes(image_path)
