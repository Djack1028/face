import cv2
import os
import numpy as np
 
# 检测人脸
def detect_face(img):
    #将测试图像转换为灰度图像，因为opencv人脸检测器需要灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    #加载OpenCV人脸检测分类器Haar
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
 
    #检测多尺度图像，返回值是一张脸部区域信息的列表（x,y,宽,高）
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
 
    # 如果未检测到面部，则返回原始图像
    if (len(faces) == 0):
        return None, None
 
    #目前假设只有一张脸，xy为左上角坐标，wh为矩形的宽高
    (x, y, w, h) = faces[0]
 
    #返回图像的正面部分
    return gray[y:y + w, x:x + h], faces[0]
 
#根据给定的（x，y）坐标和宽度高度在图像上绘制矩形
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 0), 2)
# 根据给定的（x，y）坐标标识出人名
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 2)
 
# 此函数识别传递的图像中的人物并在检测到的脸部周围绘制一个矩形及其名称
def predict(test_img):
    #生成图像的副本，这样就能保留原始图像
    img = test_img.copy()
    #检测人脸
    face, rect = detect_face(img)

    # 表情
    label_text = '表情'
 
    # 在检测到的脸部周围画一个矩形
    draw_rectangle(img, rect)
    # 标出预测的名字
    draw_text(img, label_text, rect[0], rect[1] - 5)

    print(face)
    #返回预测的图像
    return img
 
#加载测试图像
test_img1 = cv2.imread("test_data/test1.jpg")
 
#执行预测
predicted_img1 = predict(test_img1)
 
#显示两个图像
cv2.imshow('表情', predicted_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()