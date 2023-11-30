from ultralytics import YOLO
import torch

from pandas import read_csv, unique
import numpy as np

# YOLO 모델 훈련하고 평가하는 코드임

if __name__ == '__main__':
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model.train(data='bag/data.yaml', epochs=300, patience=50, batch=32)


# if __name__ == '__main__':
#     # Load a model
#     model = YOLO("yolov8n.yaml")  # build a new model from scratch
#     model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
#
#     # Use the model
#     model.train(data="coco128.yaml", epochs=300, patience=30, batch=32)# train the model
#     metrics = model.val()  # evaluate model performance on the validation set
#     results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#     path = model.export(format="onnx")  # export the model to ONNX format

# df=read_csv('data.csv', header=None, names=['PELVIS_X', 'PELVIS_Y'])
#
# print(df)