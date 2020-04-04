# coding:utf8
# Facial expression recognition
# This is for realtime camera video test
# 注意：根据硬件性能不同，需要调节实时视频的帧数
# =============================================================================
import cv2
import numpy as np
import cv2
import os
import dlib
import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from net import simpleconv3
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import PIL

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)  # 68个关键点的检测器
cascade_path = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)  # 人脸检测器


def cv_imread(file_path=""):
    img_mat = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return img_mat


# 表情识别，需要输入为opencv读取出的图片，输出为原图＋识别框+中文识别结果
def Face_expression_recognition(im):
    # 定义网络
    net = simpleconv3()

    # 加载训练好的网络权值
    modelpath = 'models/model_simpleconv3_9282_9430'
    net.load_state_dict(torch.load(modelpath, map_location=lambda storage, loc: storage))

    # 读取图片、检测人脸、给出关键点
    try:
        rects = cascade.detectMultiScale(im, 1.3, 5)
        x, y, w, h = rects[0]
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    except:
        print('No face detected')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    # 由嘴唇关键点的位置计算出一个正方形的框
    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0
    for i in range(48, 67):
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
    roiwidth = xmax - xmin
    roiheight = ymax - ymin
    roi = im[ymin:ymax, xmin:xmax, 0:3]
    if roiwidth > roiheight:
        dstlen = 1.5 * roiwidth
    else:
        dstlen = 1.5 * roiheight

    diff_xlen = dstlen - roiwidth
    diff_ylen = dstlen - roiheight
    newx = xmin
    newy = ymin
    imagerows, imagecols, channel = im.shape
    if newx >= diff_xlen / 2 and newx + roiwidth + diff_xlen / 2 < imagecols:
        newx = newx - diff_xlen / 2
    elif newx < diff_xlen / 2:
        newx = 0
    else:
        newx = imagecols - dstlen
    if newy >= diff_ylen / 2 and newy + roiheight + diff_ylen / 2 < imagerows:
        newy = newy - diff_ylen / 2
    elif newy < diff_ylen / 2:
        newy = 0
    else:
        newy = imagerows - dstlen

    # 裁剪出框中的图像，然后预处理
    testsize = 48
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    roi = im[int(newy):int(newy + dstlen), int(newx):int(newx + dstlen), 0:3]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roiresized = cv2.resize(roi, (testsize, testsize)).astype(np.float32) / 255.0
    imgblob = data_transforms(roiresized).unsqueeze(0)
    imgblob.requires_grad = False
    imgblob = Variable(imgblob)
    torch.no_grad()

    # 由模型给出预测结果
    predict = F.softmax(net(imgblob))
    index = np.argmax(predict.detach().numpy())

    # 在原图中标记处roi嘴唇区域并给出预测结果
    im_show = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('msyh.ttf', 40)
    color = (255, 255, 0)
    draw = ImageDraw.Draw(im_show)
    pos_x = int(newx)
    pos_y = int(newy)
    font = ImageFont.truetype('msyh.ttf', 20)
    position = (pos_x, pos_y - 30)
    if index == 0:
        draw.text(position, '面无表情', font=font, fill=color)
    elif index == 1:
        draw.text(position, '张嘴', font=font, fill=color)
    elif index == 2:
        draw.text(position, '撅嘴', font=font, fill=color)
    else:
        draw.text(position, '微笑', font=font, fill=color)
    im_show = np.asarray(im_show)
    cv2.rectangle(im_show, (int(newx), int(newy)), (int(newx + dstlen), int(newy + dstlen)), (255, 255, 0), 2)

    return im_show


# 读取摄像头的方法，参数1为窗口名，参数2为设备id，默认为0或-1
def openvideo(window_name, video_id):
    cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(video_id)
    while cap.isOpened():
        ok, frame = cap.read()  # ok为读取状态，bool。frame为一帧图片
        if not ok:
            break
        result = Face_expression_recognition(frame)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        cv2.imshow(window_name, result)
        C = cv2.waitKey(10)  # 10ms显示一帧,10帧视频流
        if C & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Cam Closed')


openvideo('My Cam', 0)

# 将opencv读取的图片转为PIL.Image格式，然后用Image在图片中标记中文，再转成opencv array格式输出

# image = cv_imread('./images/test1.jpg')
# image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# font = ImageFont.truetype('msyh.ttf', 40)
# color = (255, 255, 0)
# position = (100, 100)
# draw = ImageDraw.Draw(image_pil)
# draw.text(position,'汉字', font=font, fill=color)
# image_cv = np.asarray(image_pil)
# plt.figure()
# plt.imshow(image_cv)