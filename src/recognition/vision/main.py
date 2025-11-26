"""
Vision Object Detection + Face Recognition + Pose Estimation
- WC(Web Cam)
- OD(Object Detection)
- FR(Face Recognition)
- PE(Pose Estimation)
"""

import os
import json
import time
import argparse
import numpy as np
from datetime import datetime

import cv2
from ultralytics import YOLO
from deepface import DeepFace

class Vision():
    def __init__(self, args):
        self.args = args
        self.initialize()
    
    def initialize(self):
        self.WC_number = self.args.WC_number

        self.OD_model = self.args.OD_model
        
        self.PE_model = self.args.PE_model

        self.FR_model = self.args.FR_model
        self.FR_DB_path = self.args.FR_DB_path
        self.FR_metrics = self.args.FR_metrics

        self.build_machine()

        self.active_people = dict()

    def build_machine(self):
        RTSP_URL = "rtsp://172.22.207.33:8554/live/webcam"
        self.WC = cv2.VideoCapture(RTSP_URL) # Web Cam
        
        self.OD = YOLO("./models/yolov8n.pt")                    # Object Detection
        self.OD_label = self.OD.names

        self.PE = YOLO("./models/yolov8n-pose.pt")               # Pose estimation

        print(f'\nWeb Cam: {self.WC}')
        print(f'{self.OD.info()}')
        print(f'{self.PE.info()}')
        print(f'{self.OD_label}')
    
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

        if hand_raise:
            return "Hand Raise"
        if sitting:
            return "Sitting"
        if fall:
            return "fall"

        return "Standing"

    def run(self):
        print(f"\nVISION RUN: ...")

        # Steaming WebCam
        while True:
            now = datetime.now()
            print(f'\nVision {now.strftime("%Y-%m-%d %H:%M:%S")} <<<<<<<<<<<<<<<<<<<<<<<<<<< ')

            # Discrete Streaming
            ret, frame = self.WC.read()
            if not ret: break

            det = self.OD(frame, verbose=False)
            if not det: break
            det = det[0]

            det_boxes = det.boxes

            # Solve by Bounding Boxes
            for box in det_boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = self.OD_label[cls]

                # if label is person then do some PE
                if label == 'person':

                    # Pose Estimation
                    crop_person = frame[y1:y2, x1:x2]
                    
                    pose_person = self.PE(crop_person, verbose=False)
                    if not pose_person: break
                    pose_person = pose_person[0]

                    keypoints = pose_person.keypoints.data.numpy().reshape(-1,3)
                    try:
                        action = self.action_from_pose(keypoints)
                    except:
                        action = None
                else:
                    action = None

                # if label is person then do some FR    
                if label == "face" or label == "person":
                    
                    # Face Recognition
                    crop_face = frame[y1:y2, x1:x2]
                    recog_face = DeepFace.find(
                        img_path = crop_face,
                        db_path =  self.FR_DB_path,
                        distance_metric = self.FR_metrics,
                        model_name = self.FR_model,
                        enforce_detection=False,
                        silent=True
                        )
                    
                    if len(recog_face) > 0 and len(recog_face[0]) > 0:
                        recog_info = recog_face[0]
                        top_matched = recog_info.iloc[0]

                        person_name = top_matched['identity'].split('/')[-2]
                        conf = top_matched['confidence']

                        # 신원판별
                        if conf > self.args.FR_threshold:
                            
                            # 🔵 신규 입장
                            if person_name not in self.active_people:
                                self.active_people[person_name] = {
                                    "name": person_name,
                                    "enter_time": now,
                                    "last_seen": now
                                    }
                                print(f'[ENTER] ({person_name}, {conf}), ({action}, {box.xyxy[0]})')
                            else:
                                self.active_people[person_name]["last_seen"] = now
                                print(f'[STAY] ({person_name}, {conf}), ({action}, {box.xyxy[0]})')
                        else:
                            print(f'[Unknown] ({action}, {box.xyxy[0]})')
            
            # 🔴 퇴장 판단
            to_remove = []
            for name, info in self.active_people.items():
                if (now - info["last_seen"]).total_seconds() > self.args.Leave_timeout:
                    to_remove.append(name)
                    print(f"[LEAVE] {person_name} {info['enter_time'].strftime('%Y-%m-%d %H:%M:%S')} 입장 --> {now.strftime('%Y-%m-%d %H:%M:%S')} 퇴장)")
            for name in to_remove: del self.active_people[name]

            time.sleep(self.args.Sleep_time)
            
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--STR_DUMP", default='', type=str)
    parser.add_argument("--INT_DUMP", default=0, type=int)

    parser.add_argument("--WC_number", default=1, type=int)
    parser.add_argument("--OD_model", default='yolov8n.pt', type=str)
    parser.add_argument("--PE_model", default='yolov8n-pose.pt', type=str)
    parser.add_argument("--FR_model", default='VGG-Face', type=str)
    parser.add_argument("--FR_DB_path", default='FR_DB/faces', type=str)
    parser.add_argument("--FR_metrics", default='cosine', type=str)
    parser.add_argument("--FR_threshold", default=55.0, type=float)

    parser.add_argument("--Sleep_time", default=3, type=int)
    parser.add_argument("--Leave_timeout", default=5, type=int)
    
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
