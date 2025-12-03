"""
Vision Object Detection + Face Recognition + Pose Estimation
- WC(Web Cam)
- OD(Object Detection)
- FR(Face Recognition)
- PE(Pose Estimation)
"""
#pip install ultralytics deepface opencv-python numpy pandas tensorflow
#pip install tf-keras

import os
import json
import time
import argparse
import numpy as np
from datetime import datetime

import cv2
from ultralytics import YOLO # type: ignore
from deepface import DeepFace

class Vision():
    def __init__(self, args):
        self.args = args
        self.initialize()
    
    # COCO skeleton 연결 구조
    SKELETON = [
        (0,1),(0,2),(1,3),(2,4),
        (5,6),(5,7),(7,9),
        (6,8),(8,10),
        (5,11),(6,12),
        (11,12),(11,13),(13,15),
        (12,14),(14,16)
    ]

    # Bounding box + label 표시 함수
    def draw_bbox(self, frame, box, name, conf, action):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        label = f"{name} {conf:.2f} | {action}"

        # Label background
        (tw, th), _ = cv2.getTextSize(label,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,1)
        cv2.rectangle(frame,
                    (x1, y1-th-6),
                    (x1+tw, y1),
                    (0,255,0),-1)

        # Label text
        cv2.putText(frame,
                    label,
                    (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0,0,0),1)

    # Pose 스켈레톤 표시 함수
    def draw_skeleton(self, frame, box, keypoints):

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        h, w = y2-y1, x2-x1

        # 키포인트 → 원본 프레임 좌표로 변환
        pts = []
        for (x,y,_) in keypoints:
            pts.append((int(x+x1), int(y+y1)))

        # 점 찍기
        for (x,y) in pts:
            cv2.circle(frame, (x,y), 4, (0,0,255), -1)

        # 라인 연결
        for p1,p2 in self.SKELETON:
            if p1 < len(pts) and p2 < len(pts):
                cv2.line(frame,
                        pts[p1], pts[p2],
                        (255,0,0), 2)



    # 아규먼트 변수르 받은 인자들 정리하고 클래스 변수 설정
    def initialize(self):
        self.WC_number = self.args.WC_number

        self.OD_model = self.args.OD_model
        
        self.PE_model = self.args.PE_model

        self.FR_model = self.args.FR_model
        self.FR_DB_path = self.args.FR_DB_path
        self.FR_metrics = self.args.FR_metrics

        # 각종 모델들 빌드
        self.build_machine()

        # 사람들 입/출입 관리하는 파일
        self.active_people = dict()

    # 여기서 yolo부터 카메라까지 각종 Macine들 정의
    def build_machine(self):
        self.WC = cv2.VideoCapture(self.args.WC_number) # Web Cam
        if not self.WC.isOpened():
            raise RuntimeError("❌ Camera open failed")
        
        self.OD = YOLO(os.path.join("./models", self.OD_model))                    # Object Detection
        self.OD_label = self.OD.names

        self.PE = YOLO(os.path.join("./models", self.PE_model))                     # Pose estimation

        print(f'\nWeb Cam: {self.WC}')
        print(f'{self.OD.info()}')
        print(f'{self.PE.info()}')
        print(f'{self.OD_label}')
    
    # 키포인트 기반으로 포즈 추정
    def action_from_pose(self, keypoints):
        """
        keypoints: (17, 3) COCO format
        간단한 휴리스틱 기반 동작 인식
        """

        # 좌표
        nose = keypoints[0]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        left_shoulder, right_shoulder = keypoints[5], keypoints[6]
        left_hip, right_hip = keypoints[11], keypoints[12]

        # 손이 머리보다 위인지
        hand_raise = (
            (left_wrist[1] < nose[1]) or
            (right_wrist[1] < nose[1])
            )

        # 발이 매우 낮으면 앉는 동작
        sitting = (
            abs(left_ankle[1] - right_ankle[1]) < 20
            )

        # 넘어지는 동작
        shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
        ankle_avg_y = (left_ankle[1] + right_ankle[1]) / 2
        fall = shoulder_avg_y > ankle_avg_y - 30 and abs(left_shoulder[0]-right_hip[0]) > 50

        # 포즈 추정 결과 출력
        if hand_raise:
            return "Hand Raise"
        if sitting:
            return "Sitting"
        if fall:
            return "fall"

        return "Standing"

    # 실질적으로 본 코드를 실행하는 부분
    def run(self):
        print(f"\nVISION RUN: ...")

        # 웹캠 스트리밍
        while True:
            now = datetime.now()
            print(f'\nVision {now.strftime("%Y-%m-%d %H:%M:%S")} <<<<<<<<<<<<<<<<<<<<<<<<<<< ')

            # 프레임 단위 메커니즘 적용
            ret, frame = self.WC.read()
            if not ret: break # 만약 캠이 시동이 안돼서 프레임이 안 받아지면 종료

            det = self.OD(frame, verbose=False)
            
            if det: # 캠에 걸리는 물체가 있으면 아래 얼굴 인식 및 포즈 추정 진행
                det = det[0]

                det_boxes = det.boxes

                # 바운딩 박스 별로 프로세스 진행
                for box in det_boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    label = self.OD_label[cls]

                    # 사람일 경우 포즈 추정 진행
                    if label == 'person':

                        # 사람에 해당하는 부분만 잘라내기
                        crop_person = frame[y1:y2, x1:x2]
                        
                        # 키포인트 추정
                        pose_person = self.PE(crop_person, verbose=False)[0]
                        if not pose_person: break

                        # 키포인트 기반 포즈 추정
                        if pose_person.keypoints is None:
                            action = None
                        else:
                            keypoints = pose_person.keypoints.data.numpy().reshape(-1,3)
                            action = self.action_from_pose(keypoints)

                            # Skeleton draw
                            self.draw_skeleton(frame, box, keypoints)
                    else:
                        action = None

                    # 사람 혹은 사람의 얼굴이 검출될 경우
                    if label == "face" or label == "person":
                        
                        # 사람 혹은 얼굴 부분 잘라내기
                        crop_face = frame[y1:y2, x1:x2]
                        if crop_face.size == 0:
                            continue

                        # DB 기반 얼굴 판별
                        recog_face = DeepFace.find(
                            img_path = crop_face,
                            db_path =  self.FR_DB_path,
                            distance_metric = self.FR_metrics,
                            model_name = self.FR_model,
                            enforce_detection=False,
                            silent=True
                            )
                        
                        # 얼굴 인식이 된 경우
                        if len(recog_face) > 0 and len(recog_face[0]) > 0:
                            recog_info = recog_face[0]
                            top_matched = recog_info.iloc[0] # type: ignore

                            person_name = os.path.basename(os.path.dirname(top_matched['identity']))
                            conf = top_matched['confidence']

                            # 신원판별: 특정 임계값 이상의 확신도가 있는 경우
                            if conf > self.args.FR_threshold:
                                
                                # 🔵 신규 입장
                                if person_name not in self.active_people:
                                    self.active_people[person_name] = {
                                        "name": person_name,
                                        "enter_time": now,
                                        "last_seen": now
                                        }
                                    print(f'[ENTER] ({person_name}, {conf}), ({action}, {box.xyxy[0]})')
                                
                                else: # 신규입장은 아니고 이전 시점에 있던 사람이 그대로 나온 경우
                                    self.active_people[person_name]["last_seen"] = now
                                    print(f'[STAY] ({person_name}, {conf}), ({action}, {box.xyxy[0]})')

                                self.draw_bbox(frame, box, person_name, conf, action)
                            
                            else: # 얼굴 인식은 됐으나 확신도가 임계값을 못 넘는 경우 Unknown 처리
                                print(f'[Unknown] ({action}, {box.xyxy[0]})')
                                self.draw_bbox(frame, box, "Unknown", 0, action)
                        
                        else: # 얼굴 인식이 제대로 안됐으니 Unknown 처리
                            print(f'[Unknown] ({action}, {box.xyxy[0]})')
                            self.draw_bbox(frame, box, "Unknown", 0, action)
                
            # 🔴 퇴장 판단
            to_remove = []
            for name, info in self.active_people.items():
                
                # 현재 시간과 마지막으로 등장한 시점 간에 차이가 Timeout 시간을 넘는 경우 Leave 처리
                if (now - info["last_seen"]).total_seconds() > self.args.Leave_timeout:
                    to_remove.append(name)
                    print(f"[LEAVE] {name} {info['enter_time'].strftime('%Y-%m-%d %H:%M:%S')} 입장 --> {now.strftime('%Y-%m-%d %H:%M:%S')} 퇴장)")

            for name in to_remove: del self.active_people[name]

            # 몇 초 마다 검출을 진행할 지
            time.sleep(self.args.Sleep_time)

            cv2.imshow("Vision View", frame)
            if cv2.waitKey(1) & 0xFF == 27:   # ESC 종료
                break
                
        self.WC.release()
        cv2.destroyAllWindows()

            
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--STR_DUMP", default='', type=str)
    parser.add_argument("--INT_DUMP", default=0, type=int)

    parser.add_argument("--WC_number", default=0, type=int, help='웹캡번호, 0이 보통 일반적으로 노트북 웹캠')
    parser.add_argument("--OD_model", default='yolov8n.pt', type=str, help='객체탐지 모델 이름')
    parser.add_argument("--PE_model", default='yolov8n-pose.pt', type=str, help='포즈추정 모델 이름')
    parser.add_argument("--FR_model", default='VGG-Face', type=str, help='얼굴인식 모델 이름')
    parser.add_argument("--FR_DB_path", default='FR_DB/faces', type=str, help='얼굴인식 데이터베이스 경로')
    parser.add_argument("--FR_metrics", default='cosine', type=str, help='얼굴인식 유사도 비교 평가지표 이름')
    parser.add_argument("--FR_threshold", default=55.0, type=float, help='얼굴인식 신뢰도 임계값')

    parser.add_argument("--Sleep_time", default=3, type=int, help='몇 초 마다 한 번씩 프로세스 돌릴지')
    parser.add_argument("--Leave_timeout", default=5, type=int, help='사람 입/출입 관리할 때 사용하는 Timeout 시간(초)')
    
    args = parser.parse_args()
    return args
    

if __name__ == "__main__": 

    # ✅ Argument 가져오기.
    args = parse_args()

    print(f"\n>>>>> Vision Object Detection + Face Recognition + Pose Estimation ")
    print(f"\nWC_number: {args.WC_number}")
    print(f"OD_model: {args.OD_model}")
    print(f"PE_model: {args.PE_model}")
    print(f"FR_DB_path: {args.FR_DB_path}")

    exe = Vision(args)
    exe.run()
    