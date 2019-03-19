# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:02:33 2019

@author: Eric
"""
import os

#os.system("python recognize_videoDepth.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle")
os.system("python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle")