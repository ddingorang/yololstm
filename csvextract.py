import cv2
import mediapipe as mp
import numpy as np
import csv

# MediaPipe Pose 모델 로드
# 구형 버전 코드!!! PoseLandmarker 모듈을 사용하는 코드로 바꿔야 함
# 구형 버전이라 한 사람만 landmark tracking이 가능
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.4)

def extract(vidname) :
    # 비디오 캡처
    cap = cv2.VideoCapture(vidname)  # 비디오 파일 경로 또는 카메라 장치 번호
    imglist = [] # 1. 영상을 쪼갠 frame을 저장하는 배열
    csv_element = []
    csv_data = [] # 2. 추출한 landmark 좌표 정보를 저장하는 배열
    height = 0 # 영상의 높이, 폭
    width = 0

    # 1. 영상을 frame 단위로 쪼개 저장
    if cap.isOpened():
        cnt = 0
        while True :
            ret, frame = cap.read()

            if ret :
                frame = cv2.resize(frame, (1920, 1080))
                height, width, _ = frame.shape
                # cv2.imshow('detection', frame)
                # cv2.waitKey(1)
                imglist.append(frame)
                #cv2.imwrite("frame/image%d.jpg"%cnt, frame)
                cnt = cnt + 1
            else : break

    imglist = np.asarray(imglist)
    print(imglist.shape)

    # 2. frame 단위로 pose landmark 추출하여, 배열에(csv_data) 저장

    for img in imglist :
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            # 감지된 포즈 랜드마크의 좌표에 접근
            landmarks = results.pose_landmarks.landmark
            wantedlm = [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28, 0]  # 원하는 landmark만... 순서대로
            for element in wantedlm :
                cx, cy = int(landmarks[element].x * width), int(landmarks[element].y * height)
                # 순서대로 배열에 넣고
                csv_element.append(cx)
                csv_element.append(cy)
            csv_data.append(csv_element)
            csv_element = []

    print(len(csv_data))
    with open("walk/walkcsv.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in csv_data :
            writer.writerow(row)

for i in range(101, 152):
    print("{}번째".format(i))
    filename = 'walk/walk ({}).mp4'.format(i)
    extract(filename)