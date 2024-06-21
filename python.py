import cv2
import requests
# Server URL
server_url = 'http://127.0.0.1:8000/yolov8/predict/'

# Parameters
data = {'size': 320, 'confidence': 0.3, 'iou': 0.25}

# 비디오 캡처 객체 생성
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videoplayback.mp4')


while True:
    # 프레임 캡처
    ret, frame = cap.read()
    
    # 프레임이 캡처되지 않으면 종료
    if not ret:
        break
    
    # 이미지를 JPEG 형식으로 인코딩
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    # 서버로 이미지 전송
    response = requests.post(server_url, files={'image': img_encoded.tobytes()},data=data)
    
    # 서버의 응답 출력
    if response.ok:
        result = response.json()
        
        # 바운딩 박스를 그리기
        for detection in result:
            print(detection)
            xmin = int(detection['xmin'])
            ymin = int(detection['ymin'])
            xmax = int(detection['xmax'])
            ymax = int(detection['ymax'])
            confidence = detection['confidence']
            class_id = detection['class']
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # 텍스트 그리기
            label = f"Class: {class_id}, Conf: {confidence:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    else:
        print("Failed to get response from server")

    # 프레임을 화면에 표시
    cv2.imshow('YOLOv8 Detection', frame)
    
    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 비디오 캡처 객체 해제
cap.release()
cv2.destroyAllWindows()
