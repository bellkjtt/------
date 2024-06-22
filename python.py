import cv2
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

# Server URL
server_url = 'http://127.0.0.1:8000/yolov8/predict/'

# Parameters
data = {'size': 320, 'confidence': 0.35, 'iou': 0.7}

def send_request(frame_id, frame, data):
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(server_url, files={'image': img_encoded.tobytes()}, data={**data, 'frame_id': frame_id})
    return frame_id, response.json() if response.ok else []

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture('videoplayback.mp4')

frame_count = 0
results = []
frame_queue = deque()  # 큐를 사용하여 프레임을 저장

with ThreadPoolExecutor(max_workers=5) as executor:
    while True:
        # 프레임 캡처
        ret, frame = cap.read()
        
        # 프레임이 캡처되지 않으면 종료
        if not ret:
            break
        
        frame_count += 1
        
        # 매 프레임마다 큐에 저장
        frame_queue.append((frame_count, frame.copy()))
        
        # 매 프레임마다 서버에 전송
        if frame_count % 5 == 0:
            # 비동기로 요청을 보내고 future 객체를 저장
            future = executor.submit(send_request, frame_count, frame, data)
            results.append(future)
        
        # 완료된 future 객체에서 결과를 가져와서 처리
        completed_futures = [f for f in results if f.done()]
        for future in completed_futures:
            frame_id, response_data = future.result()
            bounding_boxes = response_data['detections']
            results.remove(future)
            
            # 큐에서 해당 프레임 ID와 일치하는 프레임 찾기
            while frame_queue and frame_queue[0][0] < frame_id:
                frame_queue.popleft()  # 오래된 프레임 삭제

            print(frame_queue,'큐')
            if frame_queue and frame_queue[0][0] == frame_id:
                _, current_frame = frame_queue.popleft()
                if len(bounding_boxes) > 0:
                    for detection in bounding_boxes:
                        xmin = int(detection['xmin'])
                        ymin = int(detection['ymin'])
                        xmax = int(detection['xmax'])
                        ymax = int(detection['ymax'])
                        confidence = detection['confidence']
                        class_id = detection['class']
                        
                        # 바운딩 박스 그리기
                        cv2.rectangle(current_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        # 텍스트 그리기
                        label = f"Class: {class_id}, Conf: {confidence:.2f}"
                        cv2.putText(current_frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 프레임을 화면에 표시
            cv2.imshow('YOLOv8 Detection', current_frame)
                    
                    # 일정 시간 대기
            if cv2.waitKey(1) & 0xFF == 27:
                break

# 비디오 캡처 객체 해제
cap.release()
cv2.destroyAllWindows()
