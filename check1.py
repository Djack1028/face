import cv2
import os
import numpy as np
 
# 检测人脸
def detect_face(img):
    #将测试图像转换为灰度图像，因为opencv人脸检测器需要灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    #加载OpenCV人脸检测分类器Haar   不能使用网上下载的文件，需要opencv自带的
    face_cascade = cv2.CascadeClassifier('C:\Program Files\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
 
    #检测多尺度图像，返回值是一张脸部区域信息的列表（x,y,宽,高）  scaleFactor表示每次图像尺寸减小的比例,minNeighbors表示每一个目标至少要被检测到3次才算是真的目标
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
 
    # 如果未检测到面部，则返回原始图像
    if (len(faces) == 0):
        return None, None
 
    # #目前假设只有一张脸，xy为左上角坐标，wh为矩形的宽高
    # (x, y, w, h) = faces[0]
 
    # #返回图像的正面部分
    # return gray[y:y + w, x:x + h], faces[0]

    return faces
 
#根据给定的（x，y）坐标和宽度高度在图像上绘制矩形
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 0), 2)

# 根据给定的（x，y）坐标在图片上写文字
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (128, 128, 0), 1)
 
# 此函数识别传递的图像中的人物并在检测到的脸部周围绘制一个矩形及其名称
def predict(test_img):
    #生成图像的副本，这样就能保留原始图像
    img = test_img.copy()
    #检测人脸
    # face, rect = detect_face(img)
    faces=detect_face(img)

    # 表情
    label_text = 'emotion'
 
    for rect in faces:
        # face=gray[y:y + w, x:x + h]

        # 在检测到的脸部周围画一个矩形
        draw_rectangle(img, rect)

        # 标出预测的名字
        draw_text(img, label_text, rect[0], rect[1] - 5)

        # print(face)

    #返回预测的图像
    return img
 
#加载测试图像
test_img1 = cv2.imread("test_data/test2.png")
 
#执行预测
predicted_img1 = predict(test_img1)
 
#显示两个图像
cv2.imshow('emotion', predicted_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()