#coding:utf-8
import sys
from mtcnn.MtcnnDetector import MtcnnDetector
from mtcnn.detector import Detector
from mtcnn.fcn_detector import FcnDetector
from mtcnn.mtcnn_model import P_Net, R_Net, O_Net
from mtcnn.loader import TestLoader
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
test_mode = "ONet"
#thresh = [0.9, 0.6, 0.7]

thresh = [0.45, 0.3, 0.35]

min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['model/MTCNN_model/PNet_landmark/PNet', 'model/MTCNN_model/RNet_landmark/RNet', 'model/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
# load pnet model
if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
#gt_imdb = []
#gt_imdb.append("35_Basketball_Basketball_35_515.jpg")
#imdb_ = dict()"
#imdb_['image'] = im_path
#imdb_['label'] = 5
#path = "data"
#for item in os.listdir(path):
   # if('jpg' not in item):
      #  continue
    #gt_imdb.append(os.path.join(path,item))

#print(gt_imdb)

def faceDect(imageDate):
    gt_imdb = []
    gt_image = []
    gt_box =[]
    gt_imdb.append(imageDate)
    test_data = TestLoader(gt_imdb)
    all_boxes,landmarks = mtcnn_detector.detect_face(test_data)
    count = 0
    #num = 0
    for imagepath in gt_imdb:
        #image = cv2.imread(imagepath)
        image = imagepath
        for bbox,landmark in zip(all_boxes[count],landmarks[count]):
            #cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
            #cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
            imageTmp = image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
            #imageName = "data/faceolny/%d.png" % (num)
            #cv2.imwrite(imageName,imageTmp)
            #num += 1
            #print(num)
            gt_image.append(imageTmp)
            gt_box.append(bbox)
            
        #for landmark in landmarks[count]:
            #for i in range(int(len(landmark)/2)):
                #print(landmark[2*i],landmark[2*i+1])
                #cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))
            
        count = count + 1
        #cv2.imwrite("result_landmark/%d.png" %(count),image)

        #cv2.imshow("lala",image)
        #cv2.waitKey(0)  
    return gt_image,gt_box

if __name__=="__main__":
    #face检测出的人类，box原图片人类像素的坐标
    image = cv2.imread("data\\face.jpg")
    face,box = faceDect(image)
    #face ,box = faceDect("data\\face.jpg")
    num = 0
    for imageFace in face:
        imageName = "data/faceolny/%d.png" % (num)
        cv2.imwrite(imageName,imageFace)
        num += 1
        print(num)