# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:19:06 2019

@author: Eric
"""

#change personName to the person's name
#change vidcap string to the video name (should be in same folder)
#change path to the location of the dataset.

import cv2

personName = "Ashwin"
path = "../opencv-face-recognition/dataset/"
vidcap = cv2.VideoCapture('Ashwin.MOV')
success,image = vidcap.read()
count = 0
while success:
    
    img2 = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
  
    cv2.imwrite("%s%s/frame%d.jpg" % (path,personName,count), img2)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
  
vidcap.release()