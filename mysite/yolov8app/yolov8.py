from ultralytics import YOLO
import numpy as np

class YOLOv8Model:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with any other YOLOv8 model variant

    def predict(self, image):
        # Perform inference
        results = self.model(image)
        
        # Extract and format results
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'xmin': box.xyxy[0][0].item(),
                    'ymin': box.xyxy[0][1].item(),
                    'xmax': box.xyxy[0][2].item(),
                    'ymax': box.xyxy[0][3].item(),
                    'confidence': box.conf[0].item(),
                    'class': box.cls[0].item()
                })
        return detections
