from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2
from .yolov8 import YOLOv8Model

# YOLOv8Model 인스턴스를 초기화하여 전역 변수로 정의
yolo_model = YOLOv8Model()

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        # 요청에서 이미지를 읽기
        image_file = request.FILES['image']
        image = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # 요청에서 frame_id, confidence와 iou 값을 읽기
        frame_id = int(request.POST.get('frame_id', 0))
        confidence = float(request.POST.get('confidence', 0.5))
        iou = float(request.POST.get('iou', 0.25))
        
        # 전역 YOLOv8Model 인스턴스의 confidence와 iou 값을 업데이트
        yolo_model.confidence = confidence
        yolo_model.iou = iou
        
        # 추론 수행
        detections = yolo_model.predict(image)
        
        # JSON 응답 생성
        response_data = {'frame_id': frame_id, 'detections': detections}
        return JsonResponse(response_data, safe=False)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)
