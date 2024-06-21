from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2
from .yolov8 import YOLOv8Model
import json

# Initialize the model
yolo_model = YOLOv8Model()

@csrf_exempt
def predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        result = yolo_model.predict(image)
        return JsonResponse(result, safe=False)
    return JsonResponse({'error': 'Invalid request'}, status=400)
