from __future__ import with_statement
from contextlib import closing

import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import torch
import torch.nn as nn

import sqlite3
import requests
from flask import Flask, request, session, g, redirect, url_for, \
     abort, render_template, flash
from flask import Response
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from statistics import mode
from mediapipe.tasks.python import vision

# 앞에서 훈련시킨 모델 정보 다시 작성해야...모델 로드가 가능
class onlyLstm(nn.Module) :
    def __init__(self, input_shape):
        super(onlyLstm, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_shape, hidden_size=32, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True)
        # self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        # self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        # self.dropout2 = nn.Dropout(0.1)
        # self.lstm7 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        # x, _ = self.lstm5(x)
        # x, _ = self.lstm6(x)
        # x = self.dropout2(x)
        # x, _ = self.lstm7(x)
        x = self.relu(x)
        x = self.fc(x[:, -1, :])
        return x

time_steps = 20
N_FEATURES = 26
status0 = 'None' # 이상행동 상태 라벨링
status1 = 'None'
framecnt = 0
livestrm = False

# 신체 좌표 저장하는 배열
csvdata0 = np.array([]) # 첫번째 사람
csvdata1 = np.array([]) # 두번째 사람

prevresult0 = 0
prevresult1 = 0

statuslist0 = []
statuslist1 = []

bagscore = 0
bagscorelist = []
assltscore=0
assltscorelist = []

sc = MinMaxScaler()
frame_number = 0
csv_data = []
frame_array = np.array([])

out_imglist = [] # landmark 표시, 이상행동 추론 결과를 화면에 추가한 영상의 배열

# YOLO 모델 로드
yolomodel = YOLO('yolov8n.yaml')  # build a new model from YAML
yolomodel = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

model_path = 'pose_landmarker_full.task' # pretrained된 pose landmarker모델 경로
video_path = 'testvideo/assault1.mp4' # 추론할 영상 경로

# 최근 몇개 status 합산하여 제일 다수인 행동 -> 결정

# 현재와 이전의 확률값 비교하여 가장 크게 증가한 라벨

#output_csv = 'path_to_where_you_want_to_store_your_csv_file' # 추출한 좌표 저장할 csv 파일

# gpu 사용 가능한 환경이면 gpu로, 아니면 cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False
model = onlyLstm(N_FEATURES).cuda()
model.load_state_dict(torch.load("model.pt", map_location=device))
#model = torch.load("model.pt", map_location=device) # lstm.py에서 학습했던 모델 불러오기
model.eval() # 평가 모드!!! 중요


# PoseLandmarker 옵션 설정
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
# 영상 모드로 landmarker 객체 생성
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path), # 모델 경로 위에서 지정함
    running_mode=VisionRunningMode.VIDEO, num_poses=2,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7, min_tracking_confidence=0.7) # 비디오


# 감지된 landmark 영상에 출력
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  # 프레임 별로 한장 한장 landmark 그림
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    for landmark in pose_landmarks :
        pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)])

    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())

    cv2.putText(annotated_image, str(idx), (int(1920 * pose_landmarks[23].x), int(1080*pose_landmarks[23].y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

  return annotated_image

def data_ready(rgb_image, detection_result, yoloresult) :

    global csvdata0
    global csvdata1

    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    wantedlm = [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28, 0]  # 원하는 landmark만... 순서대로

    for idx in range(len(pose_landmarks_list)):
        appendto1 = False
        pose_landmarks = pose_landmarks_list[idx]
        bodycenterx = (pose_landmarks[11].x + pose_landmarks[12].x + pose_landmarks[23].x + pose_landmarks[24].x) / 4
        bodycentery = (pose_landmarks[11].y + pose_landmarks[12].y + pose_landmarks[23].x + pose_landmarks[24].y) / 4

        # cx, cy = int(bodycenterx * 1920), int(bodycentery * 1080)
        # for result in yoloresult:
        #     for box in result.boxes.cpu().numpy():
        #         print(box.id[0])
        #         print(box.xyxy[0])
        #         if box.xyxy[:,0] <= cx <= box.xyxy[:,2] and box.xyxy[:,1] <= cy <= box.xyxy[:,3] :
        #             appendto1 = True
        #         else : break
        for lm in wantedlm:
            landmark = pose_landmarks[lm]

            cx, cy = float(landmark.x), float(landmark.y)

            # 순서대로 배열에 넣고
            if idx==0:
                #csvdata0.append(cx)
                #csvdata0.append(cy)
                csvdata0 = np.append(csvdata0, cx)
                csvdata0 = np.append(csvdata0, cy)
            elif idx==1:
                #csvdata1.append(cx)
                #csvdata1.append(cy)
                csvdata1 = np.append(csvdata1, cx)
                csvdata1 = np.append(csvdata1, cy)

    return len(csvdata0), len(csvdata1)

def infer(data, status, prevrslt) :

    reshaped_segments = np.asarray(data, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    y = np.asarray(time_steps * [0]).reshape(-1, time_steps)

    X_inf = torch.Tensor(reshaped_segments)
    y_inf = torch.Tensor(y)
    # X_inf.to(device)
    # y_inf.to(device)
    inf_dataset = TensorDataset(X_inf, y_inf)
    inf_loader = DataLoader(inf_dataset)

    for image, label in inf_loader:
        with torch.no_grad():
            result = model(image.to(device))  # 추론
            currprev = result - prevrslt
            # print("prev : ", prevrslt)
            # print("curr : ", result)
            # print("curr-prev : ", currprev)
            # if(currprev[:,3] < 0 and currprev[:,3] == currprev.min()) :
            #     #out_index = currprev[:,0:3].argmax()
            #     out_index = currprev[:,0:3].argmax()
            # else :
            #     _, out_index = torch.max(result, 1)
            _, out_index = torch.max(result, 1)

            if out_index.item() == 0:
                status = 'abnml'  # 절도
            elif out_index.item() == 1:
                status = 'abnml'  # 폭행
            elif out_index.item() == 2:
                status = 'abnml'  # 기물파손
            elif out_index.item() == 3:
                status = 'walk' # 매장 내 이동
    # 한 timestep 분량의 추론이 끝났으면 초기화, 다음 timestep 분량의 좌표를 다시 저장하여 추론 반복
    #del data[0:int(0.5 * time_steps * N_FEATURES)]
    data = np.delete(data, np.s_[:int(0.25 * time_steps)], axis=0)
    return data, status, result

def inference(csvdata, status, prevrslt) :
    csvdata = sc.fit_transform(np.asarray(csvdata, dtype=np.float32).reshape(-1, 26))
    csvdata, status, result = infer(csvdata, status, prevrslt)
    print(frame_number)
    return csvdata, status, result

def predict(mode):
    global frame_number, statuslist0, statuslist1
    global csvdata0
    global csvdata1
    global status0
    global status1
    global out_imglist
    global framecnt
    global prevresult0
    global prevresult1
    global bagscore, bagscorelist
    global assltscore, assltscorelist
    with PoseLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        # landmarker 객체 생성됨, 여기서부터 사용
        # Use OpenCV’s VideoCapture to load the input video.
        if livestrm == True:
            cap = cv2.VideoCapture(0)  # 영상 불러오고
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        else:
            cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            # if livestrm == True:
            #     cv2.waitKey(333)
            if not ret:
                break

            # Convert the frame to RGB
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # yolo로 객체 감지(사람, 백팩, 에코백에 대해서만(classes))
            yoloresults = yolomodel.track(frame, classes=[0, 24, 26], tracker="bytetrack.yaml", conf=0.5, persist=True,
                                          verbose=False)
            # Process the frame with MediaPipe Pose
            timestamps = frame_number * cap.get(cv2.CAP_PROP_FPS)  # 타임스탬프 : 프레임 장수 * fps, ms 단위임
            # print(cv2.CAP_PROP_FPS)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)  # 뽑아낸 프레임을 mp.Image 객체로 바꿔줘야함
            result = landmarker.detect_for_video(mp_image, int(timestamps))  # landmark 감지
            frame_number = frame_number + 1  # 프레임 장수 증가

            # Draw the pose landmarks on the frame
            if result.pose_landmarks:
                # mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result)  # 감지된 landmark 표시
                data_ready(mp_image.numpy_view(), result, yoloresults)  # 감지된 landmark 좌표들을 배열에 저장하여 추론할 준비

                if len(csvdata0) == time_steps * N_FEATURES:
                    csvdata0, status0, result0 = inference(csvdata0, status0, prevresult0)
                    prevresult0 = result0
                    statuslist0.append(status0)
                    bagscorelist.append(bagscore)
                    bagscore = 0
                    assltscorelist.append(assltscore)
                    assltscore = 0
                    # print(prevresult0)

                if len(csvdata1) == time_steps * N_FEATURES:
                    csvdata1, status1, result1 = inference(csvdata1, status1, prevresult1)
                    prevresult1 = result1
                    statuslist1.append(status1)

                if len(statuslist0) == 15 :
                    print(assltscorelist)
                    if statuslist0.count('walk') < 10 :

                        if max(assltscorelist) > 3 :
                            print(assltscorelist)
                            assltlvl = 1
                        else :
                            assltlvl = 0
                        try :
                            url = "http://localhost:8080/detect"
                            data = {"id" : 0, "message": "abnormal behavior detected", "time": framecnt, "assltlvl": assltlvl}
                            response = requests.get(url, json=data)
                            if response.status_code == 200:
                                print('Notification sent successfully', 200)
                            else:
                                print('Failed to send notification', 400)
                        except requests.exceptions.ConnectionError as errc:
                            print("Error Connecting:", errc)

                        cv2.putText(annotated_image, "alert sent", (1500, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                                    (0, 0, 255), 2)

                    del statuslist0[:3]
                    del bagscorelist[:3]
                    del assltscorelist[:3]

                    #print(mode(statuslist0))
                    # statuslist0.clear()

                if len(statuslist1) == 15 :
                    if statuslist1.count('walk') < 10 :

                        try :
                            url = "http://localhost:8080/detect"
                            data = {"id" : 1, "message": "abnormal behavior detected", "time" : framecnt}
                            response = requests.get(url, json=data)
                            if response.status_code == 200:
                                print('Notification sent successfully', 200)
                            else:
                                print('Failed to send notification', 400)
                        except requests.exceptions.ConnectionError as errc:
                            print("Error Connecting:", errc)

                        cv2.putText(annotated_image, "alert sent", (900, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                                    (0, 0, 255), 2)
                    del statuslist1[:3]
                    #print(mode(statuslist1))
                    # statuslist1.clear()

                # 추론 결과(이상행동 판단)에 대해 표시
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cv2.putText(annotated_image, "0 : {}".format(status0), (0, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255),
                            2)
                cv2.putText(annotated_image, "1 : {}".format(status1), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                            (0, 0, 255), 2)


                # yolo로 감지한 객체에 대해서 표시(직사각형 바운딩 박스로)
                for result in yoloresults:
                    #print("result : ", result.boxes.cpu().numpy())
                    if len(result.boxes.cpu().numpy()) == 2:
                        lim = result.boxes.cpu().numpy().xywh[0, 2] / 2 + result.boxes.cpu().numpy().xywh[1, 2] / 2
                        print(lim)
                        print(abs(result.boxes.cpu().numpy().xywh[0, 0] - result.boxes.cpu().numpy().xywh[1, 0]))
                        if abs(result.boxes.cpu().numpy().xywh[0, 0] - result.boxes.cpu().numpy().xywh[1, 0]) - abs(lim) < 100 :
                            assltscore += 1
                            print(assltscore)
                    for box in result.boxes.cpu().numpy():
                        # Add the landmark coordinates to the list and print them
                        #print("box : " ,box)
                        r = box.xyxy[0].astype(int)
                        cv2.rectangle(annotated_image, r[:2], r[2:], (0, 0, 255), 2)
                        if box.is_track == True:
                            id = box.id[0]
                            cls = box.cls[0]
                            conf = box.conf[0]
                            if cls == 24 or 26 : bagscore += 1
                            cv2.putText(annotated_image, "id : {}".format(int(id)), (r[0], r[1] - 10),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        1.5, (0, 0, 255), 2)
                            cv2.putText(annotated_image, "cls : {}".format(int(cls)), (r[0] + 200, r[1] - 10),
                                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
                            cv2.putText(annotated_image, "prob : {:.2f}".format(conf), (r[0] + 400, r[1] - 10),
                                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)

                # write_landmarks_to_csv(result, frame_number, csv_data)

                out_imglist.append(annotated_image)
                cv2.imwrite("frame/imagedetected%d.jpg" % framecnt, annotated_image)  # 추론 결과를 이미지 저장
                framecnt = framecnt + 1
                cv2.imshow('MediaPipe Pose', annotated_image)
                cv2.waitKey(1)

                if mode == 'web' :
                    ret, frame = cv2.imencode(".jpg", annotated_image)
                    frame = frame.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


            else:
                cv2.putText(frame, "0 : {}".format(status0), (0, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
                cv2.putText(frame, "1 : {}".format(status1), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
                out_imglist.append(frame)

                if mode == 'web' :
                    ret, frame = cv2.imencode(".jpg", frame)
                    frame = frame.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


            # out_imglist = np.asarray(out_imglist)
            # # 추론 결과를 영상(mp4)으로 저장
            # out = cv2.VideoWriter('./out.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 3, (1920, 1080), True)
            # for outimg in out_imglist:
            #     out.write(outimg)
            # out.release()
            # out_imglist = []

            if len(out_imglist) == 170:
                out_imglist = np.asarray(out_imglist)
                # 추론 결과를 영상(mp4)으로 저장
                out = cv2.VideoWriter('./out.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 3, (1920, 1080), True)
                for outimg in out_imglist:
                    out.write(outimg)
                out.release()
                out_imglist = []
                try :
                    url = "http://localhost:8080/detect"
                    data = "out.mp4"
                    response1 = requests.get(url, data=data)
                    print(response1.status_code)
                    if response1.status_code == 200:
                        print('video sent successfully', 200)
                    else:
                        print('Failed to send notification', 400)
                except requests.exceptions.ConnectionError as errc:
                    print("Error Connecting:", errc)


################################################

# configuration
DATABASE = '/tmp/flaskr.db'
DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'admin'
PASSWORD = 'default'

app = Flask(__name__)
app.config.from_object(__name__)

@app.before_request
def before_request():
    g.db = connect_db()

@app.teardown_request
def teardown_request(exception):
    g.db.close()
def connect_db():
    return sqlite3.connect(app.config['DATABASE'])

def init_db():
    with closing(connect_db()) as db:
        with app.open_resource('schema.sql') as f:
            db.cursor().executescript(f.read().decode('utf-8'))
        db.commit()
@app.route('/')
def index() :
    predict('web')
    return f'''
    <html>
    <body>
        <img src="/video_feed" style='width:50%; height:auto;' />
    </body>
    </html>
    '''

@app.route('/stop')
def stop() :
    return 'stopped'
@app.route('/video_feed')
def video_feed():
    return Response(predict('web'), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != app.config['USERNAME']:
            error = 'Invalid username'
        elif request.form['password'] != app.config['PASSWORD']:
            error = 'Invalid password'
        else:
            session['logged_in'] = True
            flash('You were logged in')
            return redirect(url_for('index'))
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(port=5000)


# out_imglist = np.asarray(out_imglist)
# # 추론 결과를 영상(mp4)으로 저장
# out = cv2.VideoWriter('./out.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 3, (1920, 1080), True)
# for outimg in out_imglist :
#     out.write(outimg)
# out.release()

  # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
  # You’ll need it to calculate the timestamp for each frame.

  # Loop through each frame in the video using VideoCapture#read()

  # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    #mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # result = landmarker.detect(mp_image)
    # annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result)
    # cv2.imshow('window', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

# frame_number = 0
# csv_data = []
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert the frame to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the frame with MediaPipe Pose
#     #result = pose.process(frame_rgb)
#     result = landmarker.detect(frame_rgb)
#
#     # Draw the pose landmarks on the frame
#     if result.pose_landmarks:
#         #mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         annotated_image = draw_landmarks_on_image(frame_rgb.numpy_view(), result)
#         # Add the landmark coordinates to the list and print them
#         write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data)
#
#     # Display the frame
#     cv2.imshow('MediaPipe Pose', frame)
#
#     # Exit if 'q' keypyt

  # # 매 timestep 동안 움직임의 좌표가 배열에 축적될 때 마다 추론 실행
            # if len(csvdata0) % (0.5 * time_steps * N_FEATURES) == 0:
            #     if first is True:
            #         first = False
            #         continue
            #     reshaped_segments = np.asarray(csvdata0, dtype=np.float32).reshape(- 1, time_steps, N_FEATURES)
            #     y = np.asarray(time_steps * [0]).reshape(-1, time_steps)
            #
            #     X_inf = torch.Tensor(reshaped_segments)
            #     y_inf = torch.Tensor(y)
            #     # X_inf.to(device)
            #     # y_inf.to(device)
            #     inf_dataset = TensorDataset(X_inf, y_inf)
            #     inf_loader = DataLoader(inf_dataset)
            #
            #     for image, label in inf_loader:
            #         with torch.no_grad():
            #             result = model(image.to(device))  # 추론
            #             print(result)
            #             _, out_index = torch.max(result, 1)
            #             if out_index.item() == 0:
            #                 status = 'theft'  # 절도
            #             elif out_index.item() == 1:
            #                 status = 'assault'  # 폭행
            #             elif out_index.item() == 2:
            #                 status = 'damage'  # 기물파손
            #             elif out_index.item() == 3:
            #                 status = 'walk'
            #     # 한 timestep 분량의 추론이 끝났으면 초기화, 다음 timestep 분량의 좌표를 다시 저장하여 추론 반복
            #     del csvdata1[0:int(0.5 * time_steps * N_FEATURES)]
