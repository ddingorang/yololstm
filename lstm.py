import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, Reshape, Activation
from keras.layers import Conv1D, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
#%matplotlib inline
#
# import warnings
# warnings.filterwarnings("ignore")
#
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(e)

# gpu 사용 가능한 환경이면 gpu로, 아니면 cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False

# 데이터 불러오기
def load_file(filepath):
    df = pd.read_csv(filepath, header=0, index_col=False)
    return df

data = load_file('data.csv') # 학습 데이터
valid = load_file('datavalid.csv') # 검증 데이터
test = load_file('dataedit.csv') # 테스트 데이터

# label : 0=절도, 1=폭행, 2=기물파손

# minmax 스케일링 수행
sc = MinMaxScaler()
data.loc[:,'SHOULDER_LEFT_X' : 'HEAD_Y'] = sc.fit_transform(data.loc[:,'SHOULDER_LEFT_X' : 'HEAD_Y']) # 신체 좌표 부분에 대해서만
#scaledtraindata = pd.DataFrame(sc.fit_transform(data.loc[:,'SHOULDER_LEFT_X' : 'HEAD_Y'])) # 신체 좌표 부분에 대해서만
valid.loc[:, 'SHOULDER_LEFT_X' : 'HEAD_Y'] = sc.fit_transform(valid.loc[:, 'SHOULDER_LEFT_X' : 'HEAD_Y'])
#scaledvaliddata = pd.DataFrame(sc.fit_transform(valid.loc[:, 'SHOULDER_LEFT_X' : 'label']))
test.loc[:, 'SHOULDER_LEFT_X' : 'HEAD_Y'] = sc.fit_transform(test.loc[:, 'SHOULDER_LEFT_X' : 'HEAD_Y'])
# print(scaledtraindata.shape)
# print(scaledtestdata.shape)

# 데이터 전처리
# 어깨(shoulder)부터 발목, 머리까지 13개의 부위, x, y 좌표까지 하여 26개의 feature
# 입력 데이터 : (배치 크기, timestep, feature 개수)의 3차원
# ---> 배치 크기 = 데이터 행 수 / timestep, ex) 55,152 / 72 = 766
# 출력 데이터 : (배치 크기, label 수)
def segments(df, time_steps, df2):
    N_FEATURES = 26 # feature 개수 : 26개 (13x2)
    segments = []
    sgm = []
    labels = []

    # for i in range(0, len(df) - time_steps, time_steps):
    #     for lb in df.loc[:,'SHOULDER_LEFT_X' : 'HEAD_Y']:
    #         sgm.append(df[lb].values[i:i + time_steps])
    #
    #     labels.append(mode(df['label'].values[i:i+time_steps])[0]) # 최빈값을 한 timestep의 label로 지정
    #     segments.append([sgm])
    #     sgm = []

    for i in range(0, len(df) - time_steps, int(0.25*time_steps)):
        for lb in df.loc[:,'SHOULDER_LEFT_X' : 'HEAD_Y']:
            sgm.append(df[lb].values[i:i + time_steps])

        labels.append(mode(df['label'].values[i:i+time_steps])[0]) # 최빈값을 한 timestep의 label로 지정
        segments.append([sgm])
        sgm = []

    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)
    print(len(segments))

    return reshaped_segments, labels

# time sequence 설정
TIME_PERIOD = 24
epochs = 200

# 데이터 전처리
x_train, y_train = segments(data, TIME_PERIOD, data)
x_valid, y_valid = segments(valid, TIME_PERIOD, valid)
x_test, y_test = segments(test, TIME_PERIOD, test)

print(x_train)

print('x_train shape:', x_train.shape)
print('Training samples:', x_train.shape[0])
print('y_train shape:', y_train.shape)

print('x_valid shape:', x_valid.shape)
print('Valid samples:', x_valid.shape[0])
print('y_valid shape:', y_valid.shape)

print('x_test shape:', x_test.shape)
print('Test samples:', x_test.shape[0])
print('y_test shape:', y_test.shape)

# Input and Output Dimensions
time_period, points = x_train.shape[1], x_train.shape[2] # timestep = 80, 특성 개수 = 34
# num_classes = label_encode.classes_.size
# print(list(label_encode.classes_))

# 데이터 모양 수정(reshape)
input_shape = time_period * points # 72x26 = 1,872
#x_train = x_train.reshape(x_train.shape[0], input_shape)
# (배치 크기, timestep, feature 개수) -->  (배치 개수, timestep * feature) 2차원으로 reshape하는 함수
# 일단은 사용하지 않음
print("Input Shape: ", input_shape)
print("Input Data Shape: ", x_train.shape)

# 데이터를 float형으로
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_valid = x_valid.astype('float32')
y_valid = y_valid.astype('float32')

#one-hot 인코딩 ---> 이것도 현재 필요하지 않음
#y_train = to_categorical(y_train)
# 0:절도, 1:폭행, 2:기물파손 one-hot encoding
# 절도=[1,0,0], 폭행=[0,1,0], 기물파손=[0,0,1]
#print("y_train shape: ", y_train_hot.shape) # (765, 3)

# 테스트, 검증, 훈련 데이터를 텐서로 바꾸고, gpu에 올림
X_train = torch.Tensor(x_train).cuda()
y_train = torch.LongTensor(y_train).cuda()#Long
X_train.to(device)
y_train.to(device)

X_valid = torch.Tensor(x_valid).cuda()
y_valid = torch.LongTensor(y_valid).cuda()
X_valid.to(device)
y_valid.to(device)

X_test = torch.Tensor(x_test).cuda()
y_test = torch.LongTensor(y_test).cuda()#Long
X_test.to(device)
y_test.to(device)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

valid_dataset = TensorDataset(X_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

#cnn-lstm model(tensorflow 버전)
# model = Sequential()
# model.add(LSTM(32, return_sequences=True, input_shape=(input_shape,1), activation='relu'))
# model.add(LSTM(32,return_sequences=True, activation='relu'))
# model.add(Reshape((1, 2448, 32))) # 2720
# model.add(Conv1D(filters=64,kernel_size=2, activation='relu', strides=2))
# model.add(Reshape((1224, 64))) #1360
# model.add(MaxPool1D(pool_size=4, padding='same'))
# model.add(Conv1D(filters=192, kernel_size=2, activation='relu', strides=1))
# model.add(Reshape((305, 192))) #339
# model.add(GlobalAveragePooling1D())
# model.add(BatchNormalization(epsilon=1e-06))
# model.add(Dense(3))
# model.add(Activation('softmax'))
#
# print(model.summary())
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(x_train,
#                     y_train_hot,
#                     batch_size= 192,
#                     epochs=100
#                    )

#cnn-lstm model(pytorch)
# class LstmModel(nn.Module):
#     def __init__(self, input_shape):
#         super(LstmModel, self).__init__()
#         self.lstm1 = nn.LSTM(input_shape, batch_first=True, num_layers=2, hidden_size=52)
#         #self.lstm2 = nn.LSTM(26, batch_first=True,  hidden_size=26)
#         self.conv1 = nn.Conv1d(in_channels=90, out_channels=32, kernel_size=2, stride=2)
#         self.max_pool = nn.MaxPool1d(kernel_size=4, padding=2)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=45, kernel_size=2, stride=1)
#         self.global_avg_pool = nn.AdaptiveAvgPool1d(45)
#         #self.batch_norm = nn.BatchNorm1d(192, eps=1e-06)
#         self.fc = nn.Linear(45, 4)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x, _ = self.lstm1(x)
#         #x, _ = self.lstm2(x)
#         #x = self.reshape1(x)
#         x = x.reshape(-1, 90, 26)
#
#         x = F.relu(self.conv1(x))
#         #x = self.reshape2(x)
#         x = x.reshape(32, 13)
#         x = self.max_pool(x)
#         x = F.relu(self.conv2(x))
#         #x = self.reshape3(x)
#         x = x.reshape(45, 3)
#         x = self.global_avg_pool(x)
#         #x = x.view(x.size(0), -1)  # Flatten
#         #x = self.batch_norm(x)
#         x = self.fc(x)
#         x = self.softmax(x)
#         return x

# 순수 lstm만 있는 모델
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

# 모델 생성
input_shape = 26
#model = LstmModel(input_shape).cuda()
model = onlyLstm(input_shape).cuda()

# 모델을 gpu에 올림
model.to(device)

# hyperparameter 설정
criterion = nn.CrossEntropyLoss() # 크로스 엔트로피 loss
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 옵티마이저는 Adam

def trainval(dataloader, mode):
    acc1 = 0
    iterloss=[]
    iteracc=[]
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        # print(inputs.shape, labels.shape)

        if mode == 'train' :
            model.train()
        else :
            model.eval()

        outputs = model(inputs)
        # print(outputs)
        _, out = torch.max(outputs, 1)
        # print(torch.max(outputs,1))
        loss = criterion(outputs, labels)
        iterloss.append(loss.item())
        #running_loss += loss.item()

        if mode == 'train' :
            optimizer.zero_grad()  # 그래디언트 초기화
            loss.backward()
            optimizer.step()

        acc_partial = (out == labels).float().sum()
        acc_partial = acc_partial / len(labels)
        iteracc.append(acc_partial.item())
    # print("accuracy : " + str(acc_partial.item()))

    return np.average(iterloss), np.average(iteracc)

    # print("accuracy : " + str(acc1 / 34))
    # print(f'Epoch {epoch + 1}, Loss: {t_loss}')
    acc1 = 0

# 훈련 코드
acc1=0
t_loss=[]
t_acc=[]
v_loss=[]
v_acc=[]
for epoch in range(epochs):

    tloss, tacc = trainval(train_loader, mode='train')
    print(f'Epoch {epoch + 1}, Loss: {tloss}')
    print("train accuracy : " + str(tacc))
    print("")
    t_loss.append(tloss)
    t_acc.append(tacc)
    with torch.no_grad() :
        vloss, vacc = trainval(test_loader, mode='val')
        print(f'Epoch {epoch + 1}, Loss: {vloss}')
        print("val accuracy : " + str(vacc))
        print("")
        v_loss.append(vloss)
        v_acc.append(vacc)

print('Finished Training')
torch.save(model.state_dict(), './model.pt')

plt.plot(t_loss)
plt.plot(v_loss)
plt.ylim(0.4,1.4)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['trainloss', 'valloss'])
plt.show()

#test 코드
# correct = 0
# total = 0
# acc2=0
# with torch.no_grad() :
#     for epoch in range(epochs):
#         running_loss = 0
#
#         for i, data in enumerate(test_loader, 0):
#             image, label = data
#             x = image.to(device)
#             y_= label.to(device)
#
#             model.eval()
#             outputs = model(x)
#             _, out_index = torch.max(outputs, 1)
#             loss = criterion(outputs, y_)
#             running_loss += loss.item()
#
#             total += label.size(0)
#             correct = (out_index == y_).float().sum()
#             accpartial = correct / len(label)
#             acc2 += accpartial.item()
#
#         v_loss = running_loss / len(test_loader)
#         vloss.append(v_loss)
#         v_acc = acc2 / 12
#         vacc.append(v_acc)
#         print('test accuracy : ', v_acc)
#         print(f'Epoch {epoch + 1}, Loss: {v_loss}')
#         acc2 = 0



tft = [0, 0, 0, 0]
aslt = [0, 0, 0, 0]
dmg = [0, 0, 0, 0]
nml = [0, 0, 0, 0]


correct = 0
total = 0
for image, label in test_loader:
    x = image.to(device)
    y_ = label.to(device)

    model.eval()
    outputs = model(x)
    _, out_index = torch.max(outputs, 1)

    print(out_index.size())
    print(y_.size())

    total += label.size(0)
    for i in range(y_.size(0)) :
        if y_[i] == 0 :
            if out_index[i] == 0 : tft[0] += 1
            elif out_index[i] == 1 : tft[1] += 1
            elif out_index[i] == 2 : tft[2] += 1
            elif out_index[i] == 3 : tft[3] += 1
        elif y_[i] == 1 :
            if out_index[i] == 0 : aslt[0] += 1
            elif out_index[i] == 1 : aslt[1] += 1
            elif out_index[i] == 2 : aslt[2] += 1
            elif out_index[i] == 3 : aslt[3] += 1
        elif y_[i] == 2 :
            if out_index[i] == 0 : dmg[0] += 1
            elif out_index[i] == 1 : dmg[1] += 1
            elif out_index[i] == 2 : dmg[2] += 1
            elif out_index[i] == 3 : dmg[3] += 1
        elif y_[i] == 3 :
            if out_index[i] == 0 : nml[0] += 1
            elif out_index[i] == 1 : nml[1] += 1
            elif out_index[i] == 2 : nml[2] += 1
            elif out_index[i] == 3 : nml[3] += 1

    correct += (out_index == y_).sum().float()

print(correct)
print(total)
print(tft, aslt, dmg, nml)

print('test accuracy : ', 100 * correct.item() / total)