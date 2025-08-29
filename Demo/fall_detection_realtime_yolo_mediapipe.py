
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# โหลดโมเดล YOLOv8 ขนาดเล็ก
model = YOLO("yolov8n.pt")  # ต้องมีไฟล์ yolov8n.pt ในโฟลเดอร์เดียวกัน

# ตั้งค่า MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

# เปิดกล้อง
cap = cv2.VideoCapture("H:\FallG\Fall_ex.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0])  # ตรวจจับเฉพาะ "person"
    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # แปลงเป็น RGB สำหรับ MediaPipe
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_crop)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]

                # Logic ตรวจล้ม (แนวนอน)
                if abs(hip.y - shoulder.y) < 0.07 and abs(hip.y - knee.y) < 0.07:
                    cv2.putText(annotated_frame, "FALL DETECTED", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                else:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Fall Detection (Multi-person, Real-time)", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
