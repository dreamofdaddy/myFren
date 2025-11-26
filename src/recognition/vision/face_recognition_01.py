import cv2
import face_recognition
import os
import csv
from datetime import datetime
from ultralytics import YOLO
import uuid
import numpy as np

# ------------------------------
# 1) YOLO 모델 로드
# ------------------------------
yolo_model = YOLO("./models/yolov8n-face.pt")

# ------------------------------
# 2) RTSP 입력 주소
# ------------------------------
RTSP_URL = "rtsp://172.22.207.33:8554/live/webcam"
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("RTSP 스트림 열기 실패")
    exit()

# ------------------------------
# 3) 얼굴 DB 로드
# ------------------------------
known_face_encodings = []
known_face_names = []

for person in os.listdir("faces"):
    person_dir = os.path.join("faces", person)
    if not os.path.isdir(person_dir):
        continue
    
    for file in os.listdir(person_dir):
        img = face_recognition.load_image_file(os.path.join(person_dir, file))
        enc = face_recognition.face_encodings(img)

        if len(enc) > 0:
            known_face_encodings.append(enc[0])
            known_face_names.append(person)

# ------------------------------
# 4) Unknown 저장 폴더
# ------------------------------
os.makedirs("unknown_faces", exist_ok=True)

# Unknown 인코딩 저장 리스트
unknown_encodings_db = []   # [{'id': person_id, 'encoding': vector}, ...]

# ------------------------------
# 5) 로그 파일 준비
# ------------------------------
LOG_FILE = "entry_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "enter_time", "leave_time"])

# ------------------------------
# 6) 트래킹 상태 메모리
# ------------------------------
active_people = {}  
LEAVE_TIMEOUT = 5

# ------------------------------
# 7) 실시간 처리 루프
# ------------------------------
print("▶ 실시간 얼굴 입장/퇴장 기록 시작 (Ctrl+C로 종료)")

def is_new_unknown(face_encoding, threshold=0.45):
    """새로운 Unknown인지 기존 Unknown인지 비교"""
    if len(unknown_encodings_db) == 0:
        return True, None
    
    for item in unknown_encodings_db:
        known_enc = item['encoding']
        dist = np.linalg.norm(known_enc - face_encoding)
        if dist < threshold:
            return False, item['id']

    return True, None


try:
    frame_count = -1
    frame_skip = 60
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 360))
        now = datetime.now()

        # YOLO 얼굴 검출
        results = yolo_model(frame, verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy()

        face_locations = []
        for (x1, y1, x2, y2, *_ ) in detections:
            face_locations.append((int(y1), int(x2), int(y2), int(x1)))

        encodings = face_recognition.face_encodings(frame, face_locations)
        seen_ids_this_frame = set()

        for (top, right, bottom, left), face_encoding in zip(face_locations, encodings):

            # Known faces 검사
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.45
            )

            name = "Unknown"
            person_id = None

            if True in matches:
                idx = matches.index(True)
                name = known_face_names[idx]
                person_id = name

            else:
                # Unknown 처리
                is_new, existing_id = is_new_unknown(face_encoding)

                if is_new:
                    # 새로운 Unknown → ID 생성 후 저장
                    person_id = "Unknown_" + str(uuid.uuid4())[:8]
                    
                    unknown_encodings_db.append({
                        "id": person_id,
                        "encoding": face_encoding
                    })

                    # 이미지 저장
                    face_img = frame[top:bottom, left:right]
                    save_path = f"unknown_faces/{person_id}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(save_path, face_img)
                    print(f"[UNKNOWN SAVED] {person_id} → {save_path}")

                else:
                    # 기존 Unknown
                    person_id = existing_id

            seen_ids_this_frame.add(person_id)

            # 🔵 신규 입장
            if person_id not in active_people:
                active_people[person_id] = {
                    "name": name,
                    "enter_time": now,
                    "last_seen": now
                }
                print(f"[ENTER] {name} ({person_id}) 입장 → {now}")

            else:
                active_people[person_id]["last_seen"] = now

        # 🔴 퇴장 판단
        to_remove = []
        for pid, info in active_people.items():
            if (now - info["last_seen"]).total_seconds() > LEAVE_TIMEOUT:

                with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        info["name"],
                        info["enter_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        info["last_seen"].strftime("%Y-%m-%d %H:%M:%S")
                    ])
                print(f"[LEAVE] {info['name']} ({pid}) 퇴장 → {info['last_seen']}")

                to_remove.append(pid)

        for pid in to_remove:
            del active_people[pid]

except KeyboardInterrupt:
    print("\n종료됨")
finally:
    cap.release()
