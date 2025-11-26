#pip install ultralytics
#pip install insightface
#pip install onnxruntime-gpu  # GPU 사용 시
#pip install onnxruntime      # CPU만 있을 때
import cv2
import os
import csv
import uuid
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort

# ================================================================
# 1) GPU(CUDA) 자동 감지 후 ArcFace 모델 로드
# ================================================================
providers = []

if "CUDAExecutionProvider" in ort.get_available_providers():
    print(">> CUDA 사용 가능 → GPU 모드로 ArcFace 실행합니다.")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
else:
    print(">> GPU 없음 → CPU 모드로 ArcFace 실행합니다.")
    providers = ['CPUExecutionProvider']

app = FaceAnalysis(name='buffalo_l', providers=providers)
app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1)

# ================================================================
# 2) YOLO 로드 (CPU/GPU 자동)
# ================================================================
yolo_model = YOLO("./models/yolov8n-face.pt")

# ================================================================
# 3) RTSP 스트림
# ================================================================
RTSP_URL = "rtsp://172.22.207.33:8554/live/webcam"
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("RTSP 스트림 열기 실패")
    exit()

# ================================================================
# 4) Known Faces 임베딩 로드
# ================================================================
known_face_embs = []
known_face_names = []

print(">> Known faces 로딩 중...")
for person in os.listdir("faces"):
    person_dir = os.path.join("faces", person)
    if not os.path.isdir(person_dir):
        continue

    for file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) > 0:
            known_face_embs.append(faces[0].embedding)
            known_face_names.append(person)

print(f">> Loaded {len(known_face_names)} known embeddings")

# ================================================================
# 5) Unknown 저장
# ================================================================
os.makedirs("unknown_faces", exist_ok=True)
unknown_db = []   # { "id": str, "emb": vector }

# ================================================================
# 6) 로그 파일 준비
# ================================================================
LOG_FILE = "entry_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "enter_time", "leave_time"])

# ================================================================
# 7) 입/퇴장 트래킹 메모리
# ================================================================
active_people = {}
LEAVE_TIMEOUT = 5

# ================================================================
# 8) Unknown 신규 여부 판별
# ================================================================
def is_new_unknown(embed, threshold=1.0):
    if len(unknown_db) == 0:
        return True, None

    for item in unknown_db:
        dist = np.linalg.norm(embed - item["emb"])
        if dist < threshold:
            return False, item["id"]

    return True, None

# ================================================================
# 9) 실시간 Main Loop
# ================================================================
print("▶ YOLO + ArcFace 기반 인식 시작 (Ctrl+C 종료)")

try:
    frame_count = -1
    frame_skip = 20

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

        # ---------------------
        # YOLO 얼굴 검출
        # ---------------------
        results = yolo_model(frame, verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy()

        seen_ids = set()

        for (x1, y1, x2, y2, *_ ) in detections:
            top, left = int(y1), int(x1)
            bottom, right = int(y2), int(x2)

            face_img = frame[top:bottom, left:right]

            faces = app.get(face_img)
            if len(faces) == 0:
                continue

            emb = faces[0].embedding

            # ---------------------
            # Known matching
            # ---------------------
            name = "Unknown"
            person_id = None

            if len(known_face_embs) > 0:
                dists = np.linalg.norm(known_face_embs - emb, axis=1)
                min_idx = np.argmin(dists)

                if dists[min_idx] < 1.0:   # ArcFace threshold 조정 가능
                    name = known_face_names[min_idx]
                    person_id = name

            # ---------------------
            # Unknown 처리
            # ---------------------
            if name == "Unknown":
                is_new, old_id = is_new_unknown(emb, threshold=1.0)

                if is_new:
                    person_id = "Unknown_" + str(uuid.uuid4())[:8]

                    unknown_db.append({"id": person_id, "emb": emb})

                    save_path = f"unknown_faces/{person_id}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(save_path, face_img)
                    print(f"[UNKNOWN SAVED] {person_id} -> {save_path}")

                else:
                    person_id = old_id

            seen_ids.add(person_id)

            # ---------------------
            # 입장 처리
            # ---------------------
            if person_id not in active_people:
                active_people[person_id] = {
                    "name": name,
                    "enter_time": now,
                    "last_seen": now
                }
                print(f"[ENTER] {name} ({person_id}) → {now}")

            else:
                active_people[person_id]["last_seen"] = now

        # ---------------------
        # 퇴장 처리
        # ---------------------
        remove_list = []

        for pid, info in active_people.items():
            if (now - info["last_seen"]).total_seconds() > LEAVE_TIMEOUT:
                with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        info["name"],
                        info["enter_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        info["last_seen"].strftime("%Y-%m-%d %H:%M:%S")
                    ])

                print(f"[LEAVE] {info['name']} ({pid}) → {info['last_seen']}")
                remove_list.append(pid)

        for pid in remove_list:
            del active_people[pid]

except KeyboardInterrupt:
    print("\n종료됨")

finally:
    cap.release()
