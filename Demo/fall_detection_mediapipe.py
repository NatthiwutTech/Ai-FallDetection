
import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("H:\FallG\Fall_ex.mp4") 
fall_detected = False
fall_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # ดึงค่าจุดสำคัญ: สะโพก, ไหล่, เข่า
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]

        if abs(hip.y - shoulder.y) < 0.05 and abs(hip.y - knee.y) < 0.05:
            if not fall_detected:
                fall_time = time.time()
                fall_detected = True
            elif time.time() - fall_time > 0.1: #timedelay = 0.1 sec
                print("🛑 พบการล้ม!")
        else:
            fall_detected = False

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
