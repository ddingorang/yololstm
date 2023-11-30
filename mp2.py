import cv2
import mediapipe as mp

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode

class onlyLstm(nn.Module) :
    def __init__(self, input_shape):
        super(onlyLstm, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        # self.dropout1 = nn.Dropout(0.1)
        self.lstm4 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        # self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        # self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        # self.dropout2 = nn.Dropout(0.1)
        # self.lstm7 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        # x = self.dropout1(x)
        x, _ = self.lstm4(x)
        # x, _ = self.lstm5(x)
        # x, _ = self.lstm6(x)
        # x = self.dropout2(x)
        # x, _ = self.lstm7(x)
        x = self.fc(x[:, -1, :])
        return x


time_steps = 12
N_FEATURES = 26

# gpu 사용 가능한 환경이면 gpu로, 아니면 cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False
model = torch.load("model.pt", map_location=device) # lstm.py에서 학습했던 모델 불러오기
model.eval() # 평가 모드!!! 중요

# MediaPipe Pose 모델 로드
# 구형 버전 코드!!! PoseLandmarker 모듈을 사용하는 코드로 바꿔야 함
# 구형 버전이라 한 사람만 landmark tracking이 가능
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.4)

# 비디오 캡처
cap = cv2.VideoCapture('assault3.mp4')  # 비디오 파일 경로 또는 카메라 장치 번호
imglist = [] # 1. 영상을 쪼갠 frame을 저장하는 배열
csv_data = [] # 2. 추출한 landmark 좌표 정보를 저장하는 배열
height = 0 # 영상의 높이, 폭
width = 0
framecnt = 0
status = 'None' # 이상행동 상태 라벨링
out_imglist = [] # landmark 표시, 이상행동 추론 결과를 화면에 추가한 영상의 배열

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
first = True
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
            csv_data.append(cx)
            csv_data.append(cy)
            # 감지된 landmark는 영상에 표시
            cv2.putText(img, str(element), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        # 매 timestep 동안 움직임의 좌표가 배열에 축적될 때 마다 추론 실행
        if len(csv_data) % (0.5 * time_steps * N_FEATURES) == 0 :
            if first is True :
                first = False
                continue
            reshaped_segments = np.asarray(csv_data, dtype=np.float32).reshape(- 1, time_steps, N_FEATURES)
            y = np.asarray(time_steps * [0]).reshape(-1, time_steps)

            X_inf = torch.Tensor(reshaped_segments)
            y_inf = torch.Tensor(y)
            # X_inf.to(device)
            # y_inf.to(device)
            inf_dataset = TensorDataset(X_inf, y_inf)
            inf_loader = DataLoader(inf_dataset)

            for image, label in inf_loader:
                with torch.no_grad():
                    result = model(image.to(device)) # 추론
                    print(result)
                    _, out_index = torch.max(result, 1)
                    if out_index.item() == 0 : status = 'theft' # 절도
                    elif out_index.item() == 1 : status = 'assault' # 폭행
                    elif out_index.item() == 2 : status = 'damage' # 기물파손
                    elif out_index.item() == 3: status = 'walk'
            # 한 timestep 분량의 추론이 끝났으면 초기화, 다음 timestep 분량의 좌표를 다시 저장하여 추론 반복
            del csv_data[0:int(0.5*time_steps*N_FEATURES)]


        cv2.putText(img, status, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)

    out_imglist.append(img)
    cv2.imwrite("frame/imagedetected%d.jpg" % framecnt, img) # 추론 결과를 이미지 저장
    framecnt = framecnt + 1
    cv2.imshow('Pose Landmarks Detection', img)
    cv2.waitKey(11)

# cv2.imshow('Pose Landmarks Detection', imglist)
# if cv2.waitKey(1) & 0xFF == 27:  # 'ESC' 키로 종료
#     break

print(len(csv_data))
print(len(out_imglist))
out_imglist = np.asarray(out_imglist)

# 추론 결과를 영상(mp4)으로 저장
out = cv2.VideoWriter('./out.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 3, (1920, 1080), True)
for outimg in out_imglist :
    out.write(outimg)
out.release()


#------------------------------------------소스 코드-------------------------------------------

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # BGR을 RGB로 변환
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # 포즈 랜드마크 감지
#     results = pose.process(frame_rgb)
#
#     if results.pose_landmarks:
#         # 감지된 포즈 랜드마크의 좌표에 접근
#         landmarks = results.pose_landmarks.landmark
#         #print(landmarks)
#         wantedlm = [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28, 0]  # 원하는 landmark만... 순서대로
#         for landmark in landmarks :
#             for element in wantedlm :
#                 height, width, _ = frame.shape
#                 cx, cy = int(landmarks[element].x * width), int(landmarks[element].y * height)
#                 csv_data.append(cx)
#                 csv_data.append(cy)
#                 cv2.putText(frame, str(element), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
#                 cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
#                 print(len(csv_data))

    #print(reshaped_segments.shape)



        # 랜드마크 좌표 출력
        # for i, landmark in enumerate(landmarks):
        #     #print(landmarks)
        #     wantedlm = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] # 원하는 landmark만...
        #     if i not in wantedlm : continue # 원하는 landmark 아니면 좌표를 찍지 않음
        #     height, width, _ = frame.shape
        #     cx, cy = int(landmark.x * width), int(landmark.y * height)
        #     csv_data.append(cx)
        #     csv_data.append(cy)
        #     cv2.putText(frame, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        #     cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        #
        # # landmarks 리스트는 모든 랜드마크의 정보를 포함하며, 각 랜드마크는 인덱스로 식별됩니다.
        # for landmark, landmark_id in zip(landmarks, range(len(landmarks))):
        #     x = int(landmark.x * frame.shape[1])  # x 좌표
        #     y = int(landmark.y * frame.shape[0])  # y 좌표
        #     csv_data.append([x, y])
            #print([x, y])

            #z = landmark.z  # z 좌표 (3D 포즈 추적에서 사용)

            # 각 랜드마크의 좌표를 사용하여 원하는 작업 수행

    # cv2.imshow('Pose Landmarks Detection', frame)
    # if cv2.waitKey(1) & 0xFF == 27:  # 'ESC' 키로 종료
    #     break

# print(len(csv_data))

