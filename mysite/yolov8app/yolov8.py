from ultralytics import YOLO
import numpy as np

class YOLOv8Model:
    def __init__(self, confidence=0.5, iou=0.25):
        self.model = YOLO('yolov8n.pt')
        self.confidence = confidence
        self.iou = iou

    def predict(self, image):
        # Perform inference with confidence and IoU thresholds
        results = self.model(image, conf=self.confidence, iou=self.iou)
        # Extract and format results
        detections = []
        for result in results:
            names = result.names  # Get the names dictionary
            for box in result.boxes:
                class_id = box.cls[0].item()
                class_name = names[int(class_id)]  # Convert class ID to name
                detections.append({
                    'xmin': box.xyxy[0][0].item(),
                    'ymin': box.xyxy[0][1].item(),
                    'xmax': box.xyxy[0][2].item(),
                    'ymax': box.xyxy[0][3].item(),
                    'confidence': box.conf[0].item(),
                    'class': class_name
                })
        return detections
