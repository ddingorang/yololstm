from flask import Flask, Response
import mp
import torch
import torch.nn as nn

class onlyLstm(nn.Module) :
    def __init__(self, input_shape):
        super(onlyLstm, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_shape, hidden_size=32, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        #self.lstm4 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        # self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        # self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        # self.dropout2 = nn.Dropout(0.1)
        # self.lstm7 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 4)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        #x, _ = self.lstm4(x)
        # x, _ = self.lstm5(x)
        # x, _ = self.lstm6(x)
        # x = self.dropout2(x)
        # x, _ = self.lstm7(x)
        x = self.fc(x[:, -1, :])
        return x

app = Flask(__name__)

@app.route('/')
def index() :
    return f'''
    <html>
    <body>
        <img src='/video_feed' style='width:50%; height:auto;' />
        <h2>0 : {mp.status0}, 1 : {mp.status1}</h2>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(mp.predict(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=5000)

