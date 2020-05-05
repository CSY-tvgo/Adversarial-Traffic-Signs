# XXX: 旧代码，部分内容需要修改后才可使用

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import pandas as pd
import random

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model

# Helper libraries
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print(tf.__version__)

names = {'停车检查': 0, '停车让行': 1, '减速让行': 2, '右侧行驶': 3, '右转': 4,
         '左侧行驶': 5, '左转': 6, '左转和右转': 7, '机动车道': 8, '注意上坡': 9,
         '注意下坡': 10, '注意事故易发路段': 11, '注意人行横道': 12, '注意信号灯': 13, '注意危险': 14,
         '注意反向弯路': 15, '注意向右T型交叉': 16, '注意向右急转弯': 17, '注意向左T型交叉': 18, '注意向左急转弯': 19,
         '注意学校': 20, '注意左右绕行': 21, '注意慢行': 22, '注意施工': 23, '注意无人看守铁道路口': 24,
         '注意有人看守铁道路口': 25, '注意村镇': 26, '注意连续弯道': 27, '注意非机动车': 28, '环岛': 29,
         '直行': 30, '直行和右转': 31, '禁止右转': 32, '禁止左转': 33, '禁止左转和右转': 34,
         '禁止机动车': 35, '禁止直行': 36, '禁止直行和右转': 37, '禁止直行和左转': 38, '禁止调头': 39,
         '禁止超车': 40, '禁止车辆临时或长时停放': 41, '禁止通行': 42, '禁止驶入': 43, '禁止鸣笛': 44,
         '解除40km/h限速': 45, '解除50km/h限速': 46, '调头': 47, '限速15km/h': 48, '限速30km/h': 49,
         '限速40km/h': 50, '限速50km/h': 51, '限速5km/h': 52, '限速60km/h': 53, '限速70km/h': 54,
         '限速80km/h': 55, '非机动车道': 56, '鸣笛': 57}


def getname(index):
    return np.array(list(names))[index]


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


model = keras.models.load_model('../trainedModels/CTSRD_ResNet101_20200422.h5')


video_path = "./videos/video_raw.mp4"
vid = cv2.VideoCapture(video_path)
out_path = "./videos/video_result.avi"
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter(out_path, fourcc, 30, (936, 721), True)

while True:
    return_value, frame = vid.read()
    if not return_value:
        cv2.destroyAllWindows()
        out.release()
        break
    # x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = frame.copy()[266:787, 164:705]
    x = cv2.resize(x, (112, 112))
    x = np.expand_dims(x, axis=0) / 255
    pred = model.predict(x)
    result_index = np.argmax(pred)
    result_name = getname(result_index)
    top5_index = np.fliplr(pred.argsort())[0, 0:5]
    top5_score = pred[0][top5_index] * 100
    top5_name = getname([top5_index])

    print("Prediction: %d-%s" % (result_index, result_name))
    top5_str = "置信度 Top 5:\n"
    for i in range(5):
        top5_str = top5_str + \
            "%.02f%%\t: %d-%s\n" % (top5_score[i], top5_index[i], top5_name[i])

    sign = cv2.rectangle(frame.copy(), (164, 266), (705, 787),
                         color=(255, 0, 0), thickness=2)
    sign = cv2.copyMakeBorder(
        sign, 0, 0, 0, 400, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    sign = cv2ImgAddText(sign, "预测区域\nPrediction Area",
                         167, 220, textColor=(0, 0, 255))
    sign = cv2ImgAddText(sign, top5_str,
                         720, 220, textColor=(0, 0, 255))
    sign = sign[166:887, 64:1000]
    cv2.imshow('frame', sign)
    out.write(sign)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        out.release()
        break
