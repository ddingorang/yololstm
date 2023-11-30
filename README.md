# yololstm
**YOLO와 LSTM으로 이상행동 탐지 ㅎㅎㅎㅎㅎ**  
  
readme는 마크다운 문법으로 써야 되는게 약간 빡세다 :sob:  
가봅시다 레고

## 주요 파일  
- yolo.py
- lstm.py
- mp.py
- mp2.py
- 데이터(csv) 파일들  
  - data.csv
  - databackup.csv
  - datafull.csv
  - datasample.csv
  - datavalid.csv
  - datavalidbackup.csv

### yolo.py  
yolo 모델을 학습/추론하는 코드, 영상에서 객체 탐지에 사용됨  
건드릴 일은 딱히 없음

```python
if __name__ == '__main__':
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model.train(data='bag/data.yaml', epochs=300, patience=50, batch=32)
```
  
### lstm.py :star:
lstm 모델을 학습/추론,\
신체 landmark 좌표들이 들어있는 데이터 파일을 입력으로 하여 이상행동을 탐지/분류할 수 있는 모델 학습/추론  

> #### 1. 데이터 불러오기
>   pandas 데이터프레임으로 csv 파일에서 데이터 불러오기
>   ```python
>   # 데이터 불러오기
>def load_file(filepath):
>    df = pd.read_csv(filepath, header=0, index_col=False)
>    return df
>
>  data = load_file('datafull.csv') # 학습 데이터 : 55,152개의 행, 39개의 열 중 34개의 feature
>  test = load_file('datavalid.csv') # 테스트 데이터 : 13,006개의 행, 39개의 열 중 34개의 feature
>  # label : 0=절도, 1=폭행, 2=기물파손
>  ```
> 위에서 소개한 데이터 파일들에 대해 간략히 소개하면
> - data.csv : 44,000여개의 데이터가 있는 좌표 데이터 파일
> - databackup.csv : data.csv의 백업본
> - datafull.csv : aihub에서 다운받은 전체 영상에 대한 좌표 데이터 파일(약 27만여개 데이터)
> - datasample.csv : 한 영상에 대한 좌표 데이터 파일(샘플임)
> - datavalid.csv : 테스트 위한 좌표 데이터 파일
> - datavalidbackup.csv : datavalid.csv의 백업본 
> #### 2. minmax 스케일링
> ```python
> # minmax 스케일링 수행
>sc = MinMaxScaler()
>scaledtraindata = sc.fit_transform(data.loc[:,'SHOULDER_LEFT_X' : 'HEAD_Y']) # 신체 좌표 부분에 대해서만
>scaledtestdata = sc.fit_transform(test.loc[:, 'SHOULDER_LEFT_X' : 'HEAD_Y'])
># print(scaledtraindata.shape)
># print(scaledtestdata.shape)
> ```
> #### 3. 데이터 전처리(segment 메서드)
> ```python
> # 데이터 전처리
># 어깨(shoulder)부터 발목, 머리까지 13개의 부위, x, y 좌표까지 하여 26개의 feature
># 입력 데이터 : (배치 크기, timestep, feature 개수)의 3차원
># ---> 배치 크기 = 데이터 행 수 / timestep, ex) 55,152 / 72 = 766
># 출력 데이터 : (배치 크기, label 수)
> def segments(df, time_steps):
>    N_FEATURES = 26 # feature 개수 : 26개 (13x2)
>    segments = []
>    sgm = []
>    labels = []
>
>    for i in range(0, len(df) - time_steps, time_steps):
>        for lb in df.loc[:,'SHOULDER_LEFT_X' : 'HEAD_Y']:
>            sgm.append(df[lb].values[i:i + time_steps])
>
>        labels.append(mode(df['label'].values[i:i+time_steps])[0]) # 최빈값을 한 timestep의 label로 지정
>        segments.append([sgm])
>        sgm = []
>
>    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
>    labels = np.asarray(labels)
>    print(len(segments))
>    print(len(sgm))
>
>    return reshaped_segments, labels
>  # 입력 데이터(x)와 출력 데이터(y)가 반환됨
> ```
> 예를 하나 들어보면! \
> **ex) 55000개의 행, 신체 좌표 개수가 26이라고 하면**\
> **배열 sgm** : 26개가 들어갈 때마다 append됨(sgm.append(df[lb].values[i:i+time_steps][0]))\
> **배열 segments** : 26개 포장된 걸 append(segments.append([sgm])\
> **그러면 segments에는 26개씩 포장된 배열이 55,000개 있음 ---> 크기는 (55000, 26)이 된다**\
> 이걸 (배치 개수, timestep, 특성 개수)로 reshape하려고 한다(reshaped_segments)\
> 그러면 reshape된 데이터의 크기는? timestep이 55라고 하면, **(1000, 55, 26)이 된다**
> #### 4. time sequence(timestep) 지정
> ```python
> # time sequence 설정
> TIME_PERIOD = 16
> ```
> timestep을 지정하여, segment 메서드에서 움직임 데이터를 timestep 단위로 묶는 작업을 수행할 수 있음! \
> segment 메서드를 호출할 때 timestep이 인수로 들어감 
> ```python
> # 데이터 전처리
> x_train, y_train = segments(data, TIME_PERIOD)
> x_test, y_test = segments(test, TIME_PERIOD)
> ```
> #### 5. 데이터 전처리(cuda 텐서로 바꾸고, dataloader로 변환)
  
  
#### :star: lstm 모델 두 개
**1. cnn-lstm 모델** \
   괜찮은 모델인거 같아서 가져와 보았으나, 입력 크기 조절해야 하기도 해서 의도치 않게 유기되었음
   ```python
  #cnn-lstm model(pytorch)
class LstmModel(nn.Module):
    def __init__(self, input_shape):
        super(LstmModel, self).__init__()
        self.lstm1 = nn.LSTM(input_shape, batch_first=True, num_layers=2, hidden_size=26)
        #self.lstm2 = nn.LSTM(32, batch_first=True,  hidden_size=32)
        self.conv1 = nn.Conv1d(in_channels=765, out_channels=64, kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool1d(kernel_size=4, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=192, kernel_size=2, stride=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.batch_norm = nn.BatchNorm1d(192, eps=1e-06)
        self.fc = nn.Linear(192, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        #x, _ = self.lstm2(x)
        #x = self.reshape1(x)
        x = x.reshape(-1,765, 26)
        x = self.conv1(x)
        #x = self.reshape2(x)
        x = x.reshape(1360, 64)
        x = self.max_pool(x)
        x = self.conv2(x)
        #x = self.reshape3(x)
        x = x.reshape(339,192)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.batch_norm(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
  ```
**2. 순수한 lstm 모델** \
  그냥 lstm만 사용한 모델, hidden-state 크기나 layer 개수만 입력 데이터의 복잡도에 따라 조절하면 됨
  ```python
# 순수 lstm만 있는 모델
class onlyLstm(nn.Module) :
    def __init__(self, input_shape):
        super(onlyLstm, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_shape, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        # self.dropout1 = nn.Dropout(0.1)
        # #self.lstm4 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        # self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        # self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        # self.dropout2 = nn.Dropout(0.1)
        # self.lstm7 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        # x = self.dropout1(x)
        # #x, _ = self.lstm4(x)
        # x, _ = self.lstm5(x)
        # x, _ = self.lstm6(x)
        # x = self.dropout2(x)
        # x, _ = self.lstm7(x)
        x = self.fc(x[:, -1, :])
        return x
  ```
**현재는 이 모델을 사용하고 있음!!!!**
```python
# 모델 생성
input_shape = 26 
#model = LstmModel(input_shape).cuda()
model = onlyLstm(input_shape).cuda() # 순수 lstm 모델을 사용중!!! (10/29 기준)
```

> #### 6. 학습 준비
> - 모델을 gpu에 올리고,
>   ```python
>   # 모델을 gpu에 올림
>    model.to(device)
>   ```
> - 하이퍼파라미터 설정한다
>   ```python
>   # hyperparameter 설정
>    criterion = nn.CrossEntropyLoss() # 크로스 엔트로피 loss
>    optimizer = optim.Adam(model.parameters(), lr=0.001) # 옵티마이저는 Adam
>   ```
> #### 7. 학습/테스트
> - 훈련 코드
>   ```python
>   # 훈련 코드
>    epochs = 200
>    for epoch in range(epochs):
>      running_loss = 0.0
>      for i, data in enumerate(train_loader, 0):
>          inputs, labels = data
>          #print(inputs.shape, labels.shape)
>          optimizer.zero_grad()  # 그래디언트 초기화
>
>          outputs = model(inputs)
>          #print(outputs)
>          _, out = torch.max(outputs, 1)
>          #print(torch.max(outputs,1))
>          loss = criterion(outputs, labels)
>          loss.backward()
>          optimizer.step()
>
>          running_loss += loss.item()
>
>          acc_partial = (out == labels).float().sum()
>          acc_partial = acc_partial / len(labels)
>          #print("accuracy : " + str(acc_partial.item()))
>
>      print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
>
>   print('Finished Training')
>   torch.save(model, './model.pt') # 학습 완료한 모델은 저장한다
>   ```

### mp.py
임의의 영상에서 감지되는 사람의 landmark를 추출하고, 그것을 모델에 입력 데이터로 넣어서 이상행동을 추론 \
pose landmark detection 위해 **mediapipe의 PoseLandmaker 모듈**을 사용 \
이게 최신 버전이라, **두 명 이상의 landmark도 추출 가능하나(이건 뒤늦게 알았다)** 이걸 사용한 예제 코드라던지 레퍼런스 같은게 너무 부실해서 적용하기가 너무 어려워 때려쳤었음 \
그래서 후에 서술할 solutions.pose 모듈(구버전)을 적용했던 것인데, 이건 한 명만 detection이 가능하다는걸 코드를 다 짜고 나서야 알았다 \
**최종적으로는 PoseLandmarker 모듈을 적용하여 구현하는 것이 목표!! 앞으로 진행할 예정**, 일단 뒤의 mp2.py을 사용하자...

### mp2.py :star:
mediapipe의 solutions.pose 모듈을 사용하여 pose landmark detection을 진행한 코드 \
앞에서 서술하였듯이 한 명의 landmark만 추출이 가능하단 치명적인 단점이 있음, \
궁극적으로는 PoseLandmarker로 구현한 코드로의 전환이 필요하다
#### 1. 모델 불러오기, landmark 추출 모듈 불러오기
```python
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
```
#### 나머지는 추후에 차근차근...작성하기로 하고
***
### 일단 다 제쳐두고, 영상에 대해 추론하면 최종적으로 나오는 값이 뭐냐? 이게 궁금한거잖아
mp2.py의 115행부터 살펴보자
```python
            for image, label in inf_loader:
                with torch.no_grad():
                    result = model(image.to(device)) # 추론
                    print(result)
                    _, out_index = torch.max(result, 1)
                    if out_index.item() == 0 : status = 'theft' # 절도
                    elif out_index.item() == 1 : status = 'assault' # 폭행
                    elif out_index.item() == 2 : status = 'damage' # 기물파손
                    else : status = 'None'
            # 한 timestep 분량의 추론이 끝났으면 초기화, 다음 timestep 분량의 좌표를 다시 저장하여 추론 반복
            csv_data = []

        cv2.putText(img, status, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
```
이것이 임의 영상에 등장하는 사람의 좌표를 뽑아내어, 모델에 입력 데이터로 넣어 무슨 이상행동인지 추론하는 코드인데 \
inf_loader에 한 timestep 내의 프레임 별 사람의 좌표가 다 저장되어 있다, 이걸 입력 데이터로 넣어서 추론할 것임
```python
result = model(image.to(device)) # 추론
```
- result : 출력인데, 우리가 분류하고자 하는 이상행동은 3가지니까 출력은 3개다 결국은 숫자 3개가 최종적으로 나올 것인데,
```python
      _, out_index = torch.max(result, 1)
      if out_index.item() == 0 : status = 'theft' # 절도
      elif out_index.item() == 1 : status = 'assault' # 폭행
      elif out_index.item() == 2 : status = 'damage' # 기물파손
      else : status = 'None'
```
- torch.max(result, 1) : result인 3개의 숫자 중에서 가장 큰 값(**_**)과 그 값의 인덱스(**out_index**)가 출력된다. \
  **숫자가 가장 큰 라벨이 그 라벨일 확률이 제일 높은 것** \
  예를 한번 들어보겠습니다 \
  **ex) 첫 timestep(15, 즉 5초 간에 포착한 사람, 3fps 기준)에서 초딩 한 명이 가방에 과자를 하나 집어넣고 있는 장면을 추론하고자 한다** \
  추론 결과 result가 (**-1.255, 3.177, 0.992**)임 -> torch.max를 통과하면? -> (**3.177, 1**)이 됨 \
  index가 1인 이상행동은? 폭행 -> 추론이 잘못된 것임, 0이 나와야 하는데

#### 웹 서버에 전해져야 하는 감지 정보가 뭐냐? 한다면
변수 status일 것 같다, 'theft, assault, damage'

#### 잘 이해했다면, 눈치챘을 수도 있지만
yolo만큼 빠르게 탐지하는 것이 불가능, yolo는 한 순간 프레임만 딱 보고 무슨 물체인지 탐지가 가능하지만 \
lstm은 행동을 탐지하여 분류하려면 일단 일정 timestep 만큼의 좌표 움직임이 필요함 \
우리가 사람의 행동을 판단할 때도 연속적인 움직임을 봐야 이 사람이 멀 하고 있는지 알지, 어느 순간의 한 장면만 보고서는 알 수가 없다 \
**따라서 일정 timestep 길이만큼 좌표 값을 모으고, 모델로 무슨 행동인지 추론하고, 판단 결과를 출력할 때까지 delay가 있을 수 밖에 없다** \
**그래서 yolo처럼 실시간 영상에 대해 바로바로 탐지 내용을 보여줄 수는 없다**, 하지만 delay는 무슨 1분이 넘게 걸리고 그렇다는게 아니고 \
즉각적인 출력이 불가능하다는 것이다 그냥
