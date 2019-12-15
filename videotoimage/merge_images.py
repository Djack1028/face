import os
import cv2
 
# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
length = 20 #帧率
 
# Create an output movie file (make sure resolution/frame rate matches input video!)
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output_movie = cv2.VideoWriter('E:\\videotoimage\\mergevideo\\output2_9.mp4', fourcc, length, (640, 480))
 
frame_number = 0
file_path = 'E:\\videotoimage\\imagedata\\'
 
for i in range(95):
    i+=1
    # Grab a single frame of video
    frame = cv2.imread(file_path+str(i)+".jpg")
    print("E:\\videotoimage\\imagedata\\{}.jpg".format(i))
    cv2.namedWindow("11")
    cv2.imshow("11", frame)
    frame_number += 1
 
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)
    c = cv2.waitKey(10)
    if cv2.waitKey(10) & 0xFF == ord('q'):# 如果强制停止执行程序，结束视频放映
        break
# All done!
cv2.destroyAllWindows()