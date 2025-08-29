import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from collections import deque, defaultdict
import math
import time

# Configuration - ปรับปรุงเพื่อความแม่นยำและความเร็ว
CONFIG = {
    'confidence_threshold': 0.25,      # ลดจาก 0.3 เพื่อความแม่นยำ
    'keypoint_threshold': 0.08,        # ลดจาก 0.1 เพื่อตรวจจับจุดได้มากขึ้น
    'min_keypoints': 4,                # ลดจาก 5 เพื่อไม่พลาดคนที่บังบางส่วน
    'frame_buffer_size': 8,            # ลดจาก 10 เพื่อประสิทธิภาพ
    'fall_buffer_size': 4,             # ลดจาก 5 เพื่อตอบสนองเร็วขึ้น
    'alert_cooldown': 25,              # ลดจาก 30 เพื่อตอบสนองเร็วขึ้น
    'person_tracking_threshold': 0.12, # ลดจาก 0.15 เพื่อการติดตามที่แม่นยำ
    'resize_width': 640,
    'resize_height': 480,
    'input_size': 256
}

# โหลด MoveNet MultiPose model
print("Loading MoveNet model...")
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']
print("Model loaded successfully!")

class PersonTracker:
    """ระบบติดตามบุคคลแต่ละคน"""
    def __init__(self):
        self.persons = {}
        self.next_id = 0
        self.max_distance = CONFIG['person_tracking_threshold']
        
    def update(self, keypoints_list):
        """อัปเดตการติดตามบุคคล"""
        if not keypoints_list:
            return []
            
        # คำนวณ center point สำหรับแต่ละคน
        centers = []
        for keypoints in keypoints_list:
            valid_points = keypoints[keypoints[:, 2] > CONFIG['keypoint_threshold']]
            if len(valid_points) > 0:
                center = np.mean(valid_points[:, :2], axis=0)
                centers.append(center)
            else:
                centers.append(None)
        
        # จับคู่กับบุคคลที่มีอยู่
        matched_persons = []
        used_ids = set()
        
        for i, center in enumerate(centers):
            if center is None:
                continue
                
            best_id = None
            best_distance = float('inf')
            
            # หาคนที่ใกล้ที่สุด
            for person_id, person_data in self.persons.items():
                if person_id in used_ids:
                    continue
                    
                last_center = person_data['last_center']
                distance = np.linalg.norm(center - last_center)
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_id = person_id
            
            # สร้าง ID ใหม่หรือใช้ ID เดิม
            if best_id is None:
                person_id = self.next_id
                self.next_id += 1
                self.persons[person_id] = {
                    'keypoints_history': deque(maxlen=CONFIG['frame_buffer_size']),
                    'fall_history': deque(maxlen=CONFIG['fall_buffer_size']),
                    'last_center': center,
                    'alert_cooldown': 0,
                    'last_seen': 0
                }
            else:
                person_id = best_id
                used_ids.add(person_id)
                self.persons[person_id]['last_center'] = center
                self.persons[person_id]['last_seen'] = 0
            
            # เพิ่ม keypoints ลงในประวัติ
            self.persons[person_id]['keypoints_history'].append(keypoints_list[i])
            matched_persons.append((person_id, keypoints_list[i]))
        
        # อัปเดตคนที่หายไป และลบออกหากหายไปนานเกินไป
        to_remove = []
        for person_id in list(self.persons.keys()):
            if person_id not in used_ids:
                self.persons[person_id]['last_seen'] += 1
                if self.persons[person_id]['last_seen'] > 30:
                    to_remove.append(person_id)
        
        for person_id in to_remove:
            if person_id in self.persons:
                del self.persons[person_id]
        
        return matched_persons

class FallDetector:
    """ระบบตรวจจับการล้มที่ปรับปรุงแล้ว"""
    
    def __init__(self):
        # ดัชนีจุดสำคัญ (COCO format)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # กลุ่มจุดสำคัญ
        self.body_parts = {
            'head': [0, 1, 2, 3, 4],
            'torso': [5, 6, 11, 12],
            'arms': [7, 8, 9, 10],
            'legs': [13, 14, 15, 16]
        }
    
    def get_body_angle(self, keypoints):
        """คำนวณมุมของลำตัว"""
        shoulder_points = []
        hip_points = []
        
        # ไหล่
        if keypoints[5][2] > CONFIG['keypoint_threshold']:  # left shoulder
            shoulder_points.append(keypoints[5])
        if keypoints[6][2] > CONFIG['keypoint_threshold']:  # right shoulder
            shoulder_points.append(keypoints[6])
        
        # สะโพก
        if keypoints[11][2] > CONFIG['keypoint_threshold']:  # left hip
            hip_points.append(keypoints[11])
        if keypoints[12][2] > CONFIG['keypoint_threshold']:  # right hip
            hip_points.append(keypoints[12])
        
        if len(shoulder_points) >= 1 and len(hip_points) >= 1:
            shoulder_center = np.mean([sp[:2] for sp in shoulder_points], axis=0)
            hip_center = np.mean([hp[:2] for hp in hip_points], axis=0)
            
            # คำนวณมุมจากแนวตั้ง
            dy = hip_center[0] - shoulder_center[0]  # y เป็น vertical
            dx = hip_center[1] - shoulder_center[1]  # x เป็น horizontal
            
            angle = math.atan2(abs(dx), abs(dy)) * 180 / math.pi
            return angle
        
        return None
    
    def get_body_center(self, keypoints):
        """หาจุดกึ่งกลางของร่างกาย"""
        torso_points = []
        for idx in self.body_parts['torso']:
            if idx < len(keypoints) and keypoints[idx][2] > CONFIG['keypoint_threshold']:
                torso_points.append(keypoints[idx][:2])
        
        if len(torso_points) >= 2:
            return np.mean(torso_points, axis=0)
        return None
    
    def calculate_body_ratio(self, keypoints):
        """คำนวณสัดส่วนของร่างกาย (ความสูง/ความกว้าง)"""
        # หาขอบเขตของร่างกาย
        valid_points = keypoints[keypoints[:, 2] > CONFIG['keypoint_threshold']]
        
        if len(valid_points) < 3:
            return None
        
        min_y, max_y = np.min(valid_points[:, 0]), np.max(valid_points[:, 0])
        min_x, max_x = np.min(valid_points[:, 1]), np.max(valid_points[:, 1])
        
        height = max_y - min_y
        width = max_x - min_x
        
        if width > 0:
            return height / width
        return None
    
    def detect_rapid_movement(self, person_keypoints_history):
        """ตรวจจับการเคลื่อนไหวอย่างรวดเร็ว"""
        if len(person_keypoints_history) < 3:
            return False, 0
        
        # เปรียบเทียบตำแหน่งศูนย์กลางร่างกายใน 3 เฟรมล่าสุด
        centers = []
        for keypoints in list(person_keypoints_history)[-3:]:
            center = self.get_body_center(keypoints)
            if center is not None:
                centers.append(center)
        
        if len(centers) < 3:
            return False, 0
        
        # คำนวณความเร็วในการเคลื่อนไหวแนวตั้ง
        vertical_speeds = []
        for i in range(1, len(centers)):
            dy = centers[i][0] - centers[i-1][0]  # การเคลื่อนไหวแนวตั้ง
            vertical_speeds.append(dy)
        
        # ตรวจสอบการเคลื่อนไหวลงล่างอย่างรวดเร็ว
        avg_speed = np.mean(vertical_speeds)
        max_speed = np.max(vertical_speeds)
        
        # เกณฑ์การตกอย่างรวดเร็ว
        rapid_fall = avg_speed > 0.08 or max_speed > 0.12
        confidence = min(avg_speed * 10, max_speed * 8)
        
        return rapid_fall, confidence
    
    def analyze_pose_changes(self, person_keypoints_history):
        """วิเคราะห์การเปลี่ยนแปลงท่าทาง"""
        if len(person_keypoints_history) < 2:
            return 0
        
        current_keypoints = person_keypoints_history[-1]
        prev_keypoints = person_keypoints_history[-2]
        
        # เปรียบเทียบมุมลำตัว
        current_angle = self.get_body_angle(current_keypoints)
        prev_angle = self.get_body_angle(prev_keypoints)
        
        if current_angle is not None and prev_angle is not None:
            angle_change = abs(current_angle - prev_angle)
            if angle_change > 30:  # เปลี่ยนมุมมากกว่า 30 องศา
                return 2
            elif angle_change > 15:
                return 1
        
        return 0
    
    def detect_fall(self, person_data):
        """ตรวจจับการล้มสำหรับบุคคลหนึ่ง"""
        keypoints_history = person_data['keypoints_history']
        
        if len(keypoints_history) == 0:
            return False, 0, ""
        
        current_keypoints = keypoints_history[-1]
        fall_score = 0
        reasons = []
        
        # 1. ตรวจสอบการเคลื่อนไหวอย่างรวดเร็ว
        rapid_fall, rapid_score = self.detect_rapid_movement(keypoints_history)
        if rapid_fall:
            fall_score += rapid_score * 2
            reasons.append(f"เคลื่อนไหวรวดเร็ว ({rapid_score:.2f})")
        
        # 2. วิเคราะห์มุมลำตัว
        body_angle = self.get_body_angle(current_keypoints)
        if body_angle is not None:
            if body_angle > 60:  # ลำตัวเอียงมากกว่า 60 องศา
                angle_score = (body_angle - 60) / 30  # normalize
                fall_score += angle_score
                reasons.append(f"ลำตัวเอียง ({body_angle:.1f}°)")
        
        # 3. วิเคราะห์สัดส่วนร่างกาย
        body_ratio = self.calculate_body_ratio(current_keypoints)
        if body_ratio is not None:
            if body_ratio < 1.2:  # สัดส่วนความสูง/ความกว้าง ต่ำ = นอน
                ratio_score = (1.2 - body_ratio) * 2
                fall_score += ratio_score
                reasons.append(f"สัดส่วนร่างกาย ({body_ratio:.2f})")
        
        # 4. ตรวจสอบตำแหน่งหัวเทียบกับลำตัว
        head_indices = [0, 1, 2, 3, 4]  # จุดหัว
        torso_indices = [5, 6, 11, 12]  # จุดลำตัว
        
        head_points = [current_keypoints[i] for i in head_indices 
                      if i < len(current_keypoints) and current_keypoints[i][2] > CONFIG['keypoint_threshold']]
        torso_points = [current_keypoints[i] for i in torso_indices 
                       if i < len(current_keypoints) and current_keypoints[i][2] > CONFIG['keypoint_threshold']]
        
        if len(head_points) >= 1 and len(torso_points) >= 2:
            avg_head_y = np.mean([hp[0] for hp in head_points])
            avg_torso_y = np.mean([tp[0] for tp in torso_points])
            
            if avg_head_y > avg_torso_y + 0.1:  # หัวต่ำกว่าลำตัวมาก
                head_score = (avg_head_y - avg_torso_y - 0.1) * 5
                fall_score += head_score
                reasons.append(f"หัวต่ำกว่าลำตัว ({head_score:.2f})")
        
        # 5. วิเคราะห์การเปลี่ยนแปลงท่าทาง
        pose_change_score = self.analyze_pose_changes(keypoints_history)
        if pose_change_score > 0:
            fall_score += pose_change_score
            reasons.append(f"เปลี่ยนท่าทางรวดเร็ว ({pose_change_score})")
        
        # 6. ตรวจสอบตำแหน่งในเฟรม (คนล้มมักจะอยู่ส่วนล่างของเฟรม)
        body_center = self.get_body_center(current_keypoints)
        if body_center is not None and body_center[0] > 0.7:  # อยู่ส่วนล่างของเฟรม
            fall_score += 0.5
            reasons.append("อยู่ส่วนล่างเฟรม")
        
        # ตัดสินใจขั้นสุดท้าย
        is_fallen = fall_score >= 2.0
        confidence = min(fall_score / 4.0, 1.0) * 100  # แปลงเป็นเปอร์เซ็นต์
        
        return is_fallen, confidence, "; ".join(reasons)

def detect_pose_optimized(image):
    """ตรวจจับท่าทางที่ปรับปรุงประสิทธิภาพ"""
    # ลดขนาดภาพก่อนประมวลผล
    height, width = image.shape[:2]
    if width > CONFIG['resize_width']:
        scale = CONFIG['resize_width'] / width
        new_width = CONFIG['resize_width']
        new_height = int(height * scale)
        image = tf.image.resize(image, [new_height, new_width])
    
    # Resize และ normalize input
    input_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 
                                         CONFIG['input_size'], CONFIG['input_size'])
    input_image = tf.cast(input_image, dtype=tf.int32)
    
    # ทำนาย
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()
    
    # ตัด batch dimension
    keypoints = keypoints[0]
    
    # กรองเฉพาะคนที่มี confidence score สูงพอ
    valid_detections = []
    for detection in keypoints:
        keypoint_data = detection[:51].reshape(17, 3)
        valid_keypoints = keypoint_data[keypoint_data[:, 2] > CONFIG['keypoint_threshold']]
        
        if len(valid_keypoints) >= CONFIG['min_keypoints']:
            valid_detections.append(keypoint_data)
    
    return valid_detections

def draw_optimized_pose_visualization(frame, tracked_persons, fall_results):
    """วาดkeypoints และ skeleton ที่ปรับปรุงประสิทธิภาพและความแม่นยำ"""
    height, width, _ = frame.shape
    
    # สีสำหรับแต่ละคน (ใช้สีที่แตกต่างกันชัดเจน)
    person_colors = [
        (0, 255, 0),    # เขียว
        (255, 165, 0),  # ส้ม
        (255, 0, 255),  # ม่วง
        (0, 255, 255),  # ฟ้า
        (255, 255, 0),  # เหลือง
        (128, 0, 128)   # ม่วงเข้ม
    ]
    
    # การเชื่อมต่อจุดสำคัญที่ปรับปรุงแล้ว (เรียงตามความสำคัญ)
    connections = [
        # ลำตัวหลัก (สำคัญที่สุด)
        (5, 6),   # ไหล่ซ้าย-ขวา
        (11, 12), # สะโพกซ้าย-ขวา
        (5, 11),  # ไหล่ซ้าย-สะโพกซ้าย
        (6, 12),  # ไหล่ขวา-สะโพกขวา
        
        # แขน
        (5, 7), (7, 9),   # แขนซ้าย
        (6, 8), (8, 10),  # แขนขวา
        
        # ขา
        (11, 13), (13, 15),  # ขาซ้าย
        (12, 14), (14, 16),  # ขาขวา
        
        # หัว (วาดทีหลัง)
        (0, 1), (0, 2),      # จมูก-ตา
        (1, 3), (2, 4),      # ตา-หู
    ]
    
    # สร้าง dict สำหรับ fall results เพื่อให้ค้นหาเร็วขึ้น
    fall_dict = {result['person_id']: result for result in fall_results}
    
    # วาดแต่ละคน
    for person_id, keypoints in tracked_persons:
        # เลือกสี
        base_color = person_colors[person_id % len(person_colors)]
        
        # ตรวจสอบสถานะการล้ม
        fall_result = fall_dict.get(person_id, {'is_fallen': False, 'confidence': 0})
        
        # เปลี่ยนสีเป็นแดงถ้าตรวจพบการล้ม
        if fall_result['is_fallen']:
            skeleton_color = (0, 0, 255)  # สีแดงสำหรับ skeleton
            keypoint_color = (0, 0, 255)  # สีแดงสำหรับ keypoints
        else:
            skeleton_color = base_color
            keypoint_color = base_color
        
        # คำนวณตำแหน่งจริงบนภาพล่วงหน้า (เพิ่มประสิทธิภาพ)
        keypoint_positions = []
        keypoint_scores = []
        
        for kp in keypoints:
            y, x, score = kp
            cx, cy = int(x * width), int(y * height)
            keypoint_positions.append((cx, cy))
            keypoint_scores.append(score)
        
        # วาดเส้นเชื่อม (skeleton) ก่อน
        for connection in connections:
            kp1_idx, kp2_idx = connection
            
            if (kp1_idx < len(keypoint_scores) and kp2_idx < len(keypoint_scores) and
                keypoint_scores[kp1_idx] > CONFIG['confidence_threshold'] and 
                keypoint_scores[kp2_idx] > CONFIG['confidence_threshold']):
                
                pt1 = keypoint_positions[kp1_idx]
                pt2 = keypoint_positions[kp2_idx]
                
                # ปรับความหนาของเส้นตามความสำคัญ
                if connection in [(5, 6), (11, 12), (5, 11), (6, 12)]:  # ลำตัวหลัก
                    thickness = 3
                else:
                    thickness = 2
                
                cv2.line(frame, pt1, pt2, skeleton_color, thickness)
        
        # วาด keypoints ทับบนเส้นเชื่อม
        for i, (pos, score) in enumerate(zip(keypoint_positions, keypoint_scores)):
            if score > CONFIG['confidence_threshold']:
                cx, cy = pos
                
                # ปรับขนาดจุดตามความสำคัญ
                if i in [5, 6, 11, 12]:  # จุดลำตัวสำคัญ
                    radius = 5
                elif i in [0]:  # จมูก
                    radius = 4
                else:
                    radius = 3
                
                # วาดจุดพร้อมเส้นขอบ
                cv2.circle(frame, (cx, cy), radius, keypoint_color, -1)
                cv2.circle(frame, (cx, cy), radius + 1, (255, 255, 255), 1)

def draw_fall_alert_only(frame, fall_results):
    """วาดเฉพาะการแจ้งเตือนการล้ม"""
    fallen_persons = [result for result in fall_results if result['is_fallen']]
    
    if not fallen_persons:
        return
    
    height, width = frame.shape[:2]
    
    # วาดกรอบเตือนสีแดงกระพริบ
    current_time = time.time()
    if int(current_time * 3) % 2:  # กระพริบ 3 ครั้งต่อวินาที
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 8)
    
    # ข้อความเตือนหลัก
    alert_text = "⚠️ FALL DETECTED! ⚠️"
    text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = (width - text_size[0]) // 2
    text_y = 50
    
    # พื้นหลังข้อความ
    cv2.rectangle(frame, (text_x - 10, text_y - 35), 
                 (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
    cv2.putText(frame, alert_text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

def main():
    """ฟังก์ชันหลัก - ลดการทำงานที่ไม่จำเป็น"""
    # เปิดไฟล์วิดีโอ
    cap = cv2.VideoCapture("H:\FallG\Fall_ex.mp4")
    
    if not cap.isOpened():
        print("ไม่สามารถเปิดไฟล์วิดีโอได้")
        return
    
    # ตั้งค่าความละเอียด
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['resize_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['resize_height'])
    
    # สร้างอ็อบเจ็กต์
    person_tracker = PersonTracker()
    fall_detector = FallDetector()
    
    print("=== Optimized Fall Detection System ===")
    print("กด ESC เพื่อออกจากโปรแกรม")
    print("=== กำลังประมวลผล... ===")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("จบไฟล์วิดีโอหรือไม่สามารถอ่านเฟรมได้")
            break
        
        # ปรับขนาดเฟรมสำหรับประมวลผล
        if frame.shape[1] > CONFIG['resize_width']:
            scale = CONFIG['resize_width'] / frame.shape[1]
            new_width = CONFIG['resize_width']
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # แปลงสีและตรวจจับท่าทาง
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = tf.convert_to_tensor(rgb_frame)
        keypoints_list = detect_pose_optimized(tensor_frame)
        
        # ติดตามบุคคล
        tracked_persons = person_tracker.update(keypoints_list)
        
        # ตรวจจับการล้ม
        fall_results = []
        for person_id, keypoints in tracked_persons:
            # ตรวจสอบว่า person_id ยังมีอยู่ใน tracker หรือไม่
            if person_id not in person_tracker.persons:
                continue
            
            person_data = person_tracker.persons[person_id]
            
            # ลด cooldown
            if person_data['alert_cooldown'] > 0:
                person_data['alert_cooldown'] -= 1
            
            is_fallen, confidence, reason = fall_detector.detect_fall(person_data)
            
            # เพิ่มผลลัพธ์ลงในประวัติ
            person_data['fall_history'].append(is_fallen)
            
            # ตรวจสอบการล้มต่อเนื่อง
            recent_falls = list(person_data['fall_history'])[-3:]  # 3 เฟรมล่าสุด
            consistent_fall = sum(recent_falls) >= 2  # ล้มอย่างน้อย 2 ใน 3 เฟรม
            
            # แจ้งเตือนเฉพาะเมื่อยืนยันการล้ม
            if consistent_fall and person_data['alert_cooldown'] <= 0:
                print(f"🚨 FALL ALERT - Person {person_id}: {confidence:.1f}%")
                person_data['alert_cooldown'] = CONFIG['alert_cooldown']
            
            fall_results.append({
                'person_id': person_id,
                'is_fallen': consistent_fall,
                'confidence': confidence,
                'reason': reason
            })
        
        # วาด keypoints และ skeleton
        draw_optimized_pose_visualization(frame, tracked_persons, fall_results)
        
        # วาดการแจ้งเตือนการล้ม
        draw_fall_alert_only(frame, fall_results)
        
        # แสดงเฟรม
        cv2.imshow('Fall Detection System', frame)
        
        # จัดการการกดปุ่ม
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    # ปิดทรัพยากร
    cap.release()
    cv2.destroyAllWindows()
    print("ปิดระบบเรียบร้อยแล้ว")

if __name__ == "__main__":
    main()  