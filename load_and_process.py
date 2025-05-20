import cv2
import numpy as np
import pandas as pd

dataset_path = "fer2013/fer2013.csv"
image_size = (48, 48)


def load_fer2013(num_classes=7):  # 默认加载所有7个类别，以便向后兼容或在其他地方使用
    data = pd.read_csv(dataset_path)
    # 筛选指定数量的类别
    # FER2013数据集的情感标签是 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    # 如果 num_classes 是 6, 我们通常会排除掉一个类别，例如 'Disgust' (标签1) 或者 'Neutral' (标签6)
    # 这里我们假设用户希望保留前 num_classes 个类别 (0 到 num_classes-1)
    # 或者，更常见的做法是，如果 num_classes=6，则排除标签为6的'Neutral'类别，保留0-5
    # 为了与 train_emotion_classifier.py 中的 num_classes=6 保持一致，我们假设排除标签为6的类别
    if num_classes < 7:
        data = data[data['emotion'] < num_classes]
    pixels = data["pixels"].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(" ")]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype("uint8"), image_size)
        # cv2.imshow('a', face)
        # cv2.waitKey(0)
        faces.append(face.astype("float32"))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    # emotions = pd.get_dummies(data['emotion']).as_matrix()
    emotions = pd.get_dummies(data["emotion"]).values
    return faces, emotions


def preprocess_input(x, v2=True):
    x = x.astype("float32")
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def preprocess_input_0(x):
    x = x.astype("float32")
    mean = np.mean(x, axis=0)
    x = x - mean
    # std = np.std(x)
    # x = (x - mean) / std
    return x
