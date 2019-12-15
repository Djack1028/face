import cv2
import sys
from PIL import Image


def CatchUsbVideo(window_name):
    cv2.namedWindow(window_name)
    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("22.flv")

    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("E:\\videotoimage\\haarcascade_frontalface_default.xml")
    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
    i = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break
            # 将当前帧转换成灰度图像
        i +=1
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=15, minSize=(32, 32), flags=4)
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x - 10, y - 10), (x + w, y + h), color, 2)
        cv2.imshow(window_name, frame)
        cv2.imwrite("E:\\videotoimage\\imagedata\\{}.jpg".format(i),frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):# 如果强制停止执行程序，结束视频放映
            break
if __name__ == '__main__':
    CatchUsbVideo("detect_image")
