import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# โหลด MoveNet MultiPose model
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']

def detect_pose(image):
    # Resize และ normalize input
    input_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    
    # ทำนาย
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()  # shape: (1, 6, 56)
    
    # ตัด batch dimension
    keypoints = keypoints[0]  # shape: (6, 56)
    
    # กรองเฉพาะคนที่มี confidence score สูงพอ
    valid_detections = []
    for detection in keypoints:
        # 56 values = 17 keypoints * 3 (y, x, confidence) + 5 additional values
        # The first 51 values are keypoints (17 * 3)
        keypoint_data = detection[:51].reshape(17, 3)  # 17 keypoints, each with y, x, confidence
        
        # เช็คว่ามีจุดสำคัญที่มี confidence สูงพอหรือไม่
        valid_keypoints = keypoint_data[keypoint_data[:, 2] > 0.1]  # confidence > 0.1
        if len(valid_keypoints) > 5:  # ต้องมีอย่างน้อย 5 จุดที่มองเห็นได้
            valid_detections.append(keypoint_data)
    
    return valid_detections

def draw_keypoints(frame, keypoints, threshold=0.3):
    height, width, _ = frame.shape
    
    # กำหนดสีสำหรับแต่ละคน
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for person_idx, person in enumerate(keypoints):
        color = colors[person_idx % len(colors)]
        
        for kp_idx, kp in enumerate(person):
            y, x, score = kp
            if score > threshold:
                cx, cy = int(x * width), int(y * height)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                
                # วาดหมายเลขจุดสำคัญ (optional)
                if score > 0.5:  # เฉพาะจุดที่มั่นใจมาก
                    cv2.putText(frame, str(kp_idx), (cx-10, cy-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

def detect_fall(keypoints, threshold=0.3):
    """ตรวจจับการล้ม"""
    global previous_keypoints, frame_count
    
    falls_detected = []
    
    for person_idx, person in enumerate(keypoints):
        # ดัชนีจุดสำคัญ (COCO format)
        nose = person[0] if len(person) > 0 else None
        left_shoulder = person[5] if len(person) > 5 else None
        right_shoulder = person[6] if len(person) > 6 else None
        left_hip = person[11] if len(person) > 11 else None
        right_hip = person[12] if len(person) > 12 else None
        left_knee = person[13] if len(person) > 13 else None
        right_knee = person[14] if len(person) > 14 else None
        
        fall_indicators = 0
        fall_reason = ""
        
        # เช็คจุดสำคัญที่จำเป็น
        key_points = [nose, left_shoulder, right_shoulder, left_hip, right_hip]
        valid_key_points = [kp for kp in key_points if kp is not None and kp[2] > threshold]
        
        if len(valid_key_points) < 3:
            continue  # ไม่มีจุดสำคัญเพียงพอ
        
        # คำนวณตำแหน่งเฉลี่ยของหัวและลำตัว
        head_y = nose[0] if nose is not None and nose[2] > threshold else None
        
        shoulder_points = []
        if left_shoulder is not None and left_shoulder[2] > threshold:
            shoulder_points.append(left_shoulder)
        if right_shoulder is not None and right_shoulder[2] > threshold:
            shoulder_points.append(right_shoulder)
        
        hip_points = []
        if left_hip is not None and left_hip[2] > threshold:
            hip_points.append(left_hip)
        if right_hip is not None and right_hip[2] > threshold:
            hip_points.append(right_hip)
        
        knee_points = []
        if left_knee is not None and left_knee[2] > threshold:
            knee_points.append(left_knee)
        if right_knee is not None and right_knee[2] > threshold:
            knee_points.append(right_knee)
        
        # คำนวณศูนย์กลางลำตัว (center of mass)
        body_center_y = 0
        body_points = shoulder_points + hip_points
        if body_points:
            body_center_y = sum([p[0] for p in body_points]) / len(body_points)
        
        # ตรวจสอบการเคลื่อนไหวอย่างรวดเร็ว (Rapid Movement Detection)
        rapid_fall = False
        if (frame_count >= len(previous_keypoints) and 
            len(previous_keypoints) > 0 and 
            person_idx < len(previous_keypoints[-1])):
            
            # เปรียบเทียบกับเฟรมก่อนหน้า 3-5 เฟรม
            frames_to_check = min(5, len(previous_keypoints))
            
            for i in range(1, frames_to_check + 1):
                if person_idx < len(previous_keypoints[-i]):
                    prev_person = previous_keypoints[-i][person_idx]
                    
                    # หาจุดสำคัญที่เทียบได้
                    prev_nose = prev_person[0] if len(prev_person) > 0 else None
                    prev_shoulders = []
                    prev_hips = []
                    
                    if len(prev_person) > 5 and prev_person[5][2] > threshold:
                        prev_shoulders.append(prev_person[5])
                    if len(prev_person) > 6 and prev_person[6][2] > threshold:
                        prev_shoulders.append(prev_person[6])
                    if len(prev_person) > 11 and prev_person[11][2] > threshold:
                        prev_hips.append(prev_person[11])
                    if len(prev_person) > 12 and prev_person[12][2] > threshold:
                        prev_hips.append(prev_person[12])
                    
                    # คำนวณศูนย์กลางลำตัวเก่า
                    prev_body_points = prev_shoulders + prev_hips
                    if prev_body_points:
                        prev_body_center_y = sum([p[0] for p in prev_body_points]) / len(prev_body_points)
                        
                        # เช็คการเคลื่อนไหวลงล่างอย่างรวดเร็ว
                        vertical_movement = body_center_y - prev_body_center_y
                        
                        # ถ้าเคลื่อนที่ลงมาเกิน 15% ของความสูงภาพใน 3-5 เฟรม = การตกอย่างรวดเร็ว
                        if vertical_movement > 0.15 and i <= 3:
                            rapid_fall = True
                            fall_indicators += 2  # ให้น้ำหนักมากเพราะเป็นสัญญาณที่แข็งแกร่ง
                            fall_reason += "การเคลื่อนไหวลงอย่างรวดเร็ว "
                            break
                        elif vertical_movement > 0.1 and i <= 5:
                            rapid_fall = True
                            fall_indicators += 1
                            fall_reason += "การเคลื่อนไหวลงเร็ว "
                            break
        
        # ตรวจสอบการล้มแบบดั้งเดิม
        is_fallen = False
        
        # 1. เช็คว่าหัวต่ำกว่าไหล่หรือสะโพกมากหรือไม่
        if head_y is not None and len(shoulder_points) > 0:
            avg_shoulder_y = sum([sp[0] for sp in shoulder_points]) / len(shoulder_points)
            if head_y > avg_shoulder_y + 0.12:  # ลดความเข้มงวดเล็กน้อย
                fall_indicators += 1
                fall_reason += "หัวต่ำกว่าไหล่ "
        
        if head_y is not None and len(hip_points) > 0:
            avg_hip_y = sum([hp[0] for hp in hip_points]) / len(hip_points)
            if head_y > avg_hip_y + 0.08:  # ลดความเข้มงวดเล็กน้อย
                fall_indicators += 1
                fall_reason += "หัวต่ำกว่าสะโพก "
        
        # 2. เช็คว่าลำตัวแนวนอนหรือไม่
        if len(shoulder_points) >= 2 and len(hip_points) >= 2:
            shoulder_height_diff = abs(shoulder_points[0][0] - shoulder_points[1][0])
            hip_height_diff = abs(hip_points[0][0] - hip_points[1][0])
            
            if shoulder_height_diff < 0.08 and hip_height_diff < 0.08:
                if body_center_y > 0.65:  # อยู่ในส่วนล่างของเฟรม
                    fall_indicators += 1
                    fall_reason += "ลำตัวแนวนอน "
        
        # 3. เช็คว่าเข่าสูงกว่าสะโพกหรือไม่
        if len(knee_points) > 0 and len(hip_points) > 0:
            avg_knee_y = sum([kp[0] for kp in knee_points]) / len(knee_points)
            avg_hip_y = sum([hp[0] for hp in hip_points]) / len(hip_points)
            
            if avg_knee_y < avg_hip_y - 0.03:
                fall_indicators += 1
                fall_reason += "เข่าสูงกว่าสะโพก "
        
        # 4. เช็คสัดส่วนความสูงของคน
        if head_y is not None and len(hip_points) > 0:
            avg_hip_y = sum([hp[0] for hp in hip_points]) / len(hip_points)
            body_height = abs(head_y - avg_hip_y)
            
            if body_height < 0.15:
                fall_indicators += 1
                fall_reason += "ลำตัวสั้นผิดปกติ "
        
        # ตัดสินใจว่าล้มหรือไม่ (ให้น้ำหนักการเคลื่อนไหวรวดเร็วมากขึ้น)
        if rapid_fall and fall_indicators >= 2:
            is_fallen = True
        elif not rapid_fall and fall_indicators >= 3:  # ต้องการหลักฐานมากกว่าถ้าไม่มีการเคลื่อนไหวรวดเร็ว
            is_fallen = True
        
        falls_detected.append({
            'person_id': person_idx,
            'is_fallen': is_fallen,
            'confidence': fall_indicators,
            'reason': fall_reason.strip(),
            'rapid_fall': rapid_fall
        })
    
    return falls_detected

def draw_skeleton(frame, keypoints, threshold=0.3):
    """วาดเส้นเชื่อมจุดสำคัญ"""
    height, width, _ = frame.shape
    
    # กำหนดการเชื่อมต่อของจุดสำคัญ (COCO format)
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # หัว
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # แขน
        (5, 11), (6, 12), (11, 12),  # ลำตัว
        (11, 13), (13, 15), (12, 14), (14, 16)  # ขา
    ]
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for person_idx, person in enumerate(keypoints):
        color = colors[person_idx % len(colors)]
        
        for connection in connections:
            kp1, kp2 = connection
            if kp1 < len(person) and kp2 < len(person):
                y1, x1, score1 = person[kp1]
                y2, x2, score2 = person[kp2]
                
                if score1 > threshold and score2 > threshold:
                    pt1 = (int(x1 * width), int(y1 * height))
                    pt2 = (int(x2 * width), int(y2 * height))
                    cv2.line(frame, pt1, pt2, color, 2)

def draw_fall_alert(frame, falls_detected):
    """วาดการแจ้งเตือนการล้ม"""
    for fall_info in falls_detected:
        if fall_info['is_fallen']:
            # วาดกรอบเตือนสีแดง
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 8)
            
            # ข้อความเตือนใหญ่และชัดเจน
            alert_text = "FALL DETECTED!"
            cv2.putText(frame, alert_text, (width//2 - 200, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

# เปิดกล้อง
cap = cv2.VideoCapture("H:\FallG\Fall_ex.mp4")

# ตั้งค่าความละเอียด (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("กด ESC เพื่อออกจากโปรแกรม")
print("กด 's' เพื่อแสดง/ซ่อนเส้นเชื่อมจุดสำคัญ")
print("กด 'f' เพื่อเปิด/ปิดการตรวจจับการล้ม")

show_skeleton = False
fall_detection_enabled = True
fall_alert_sound = False

# ตัวแปรสำหรับเก็บสถานะการล้ม
fall_history = []
alert_cooldown = 0
previous_keypoints = []  # เก็บ keypoints ของเฟรมก่อนหน้า
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านเฟรมจากกล้องได้")
        break
    
    frame_count += 1
    
    # แปลงสีจาก BGR เป็น RGB สำหรับ TensorFlow
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor_frame = tf.convert_to_tensor(rgb_frame)
    
    # ตรวจจับท่าทาง
    keypoints = detect_pose(tensor_frame)
    
    # เก็บ keypoints สำหรับการวิเคราะห์การเคลื่อนไหว
    if len(keypoints) > 0:
        previous_keypoints.append(keypoints)
        # เก็บเฉพาะ 10 เฟรมล่าสุด
        if len(previous_keypoints) > 10:
            previous_keypoints.pop(0)
    
    # วาดผล
    falls_detected = []
    if len(keypoints) > 0:
        draw_keypoints(frame, keypoints)
        if show_skeleton:
            draw_skeleton(frame, keypoints)
        
        # ตรวจจับการล้ม
        if fall_detection_enabled:
            falls_detected = detect_fall(keypoints)
            
            # เช็คว่ามีคนล้มหรือไม่
            any_fall = any(fall['is_fallen'] for fall in falls_detected)
            
            if any_fall:
                fall_history.append(1)
                if alert_cooldown <= 0:
                    print("⚠️ FALL DETECTED! ⚠️")
                    for fall in falls_detected:
                        if fall['is_fallen']:
                            fall_type = "การล้มอย่างรวดเร็ว" if fall['rapid_fall'] else "การล้มปกติ"
                            print(f"Person {fall['person_id'] + 1}: {fall_type}")
                    alert_cooldown = 30  # แจ้งเตือนทุก 30 เฟรม
            else:
                fall_history.append(0)
                
            # เก็บประวัติ 10 เฟรมล่าสุด
            if len(fall_history) > 10:
                fall_history.pop(0)
            
            # ลด cooldown
            if alert_cooldown > 0:
                alert_cooldown -= 1
        
        # วาดการแจ้งเตือน
        if falls_detected:
            draw_fall_alert(frame, falls_detected)
    
    cv2.imshow('MoveNet Fall Detection System', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('s'):  # 's' key
        show_skeleton = not show_skeleton
        print(f"Skeleton display: {'ON' if show_skeleton else 'OFF'}")
    elif key == ord('f'):  # 'f' key
        fall_detection_enabled = not fall_detection_enabled
        print(f"Fall detection: {'ON' if fall_detection_enabled else 'OFF'}")

cap.release()
cv2.destroyAllWindows()