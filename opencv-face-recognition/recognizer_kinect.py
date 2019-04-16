'''
recognize along with depth
'''

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from pythonosc import udp_client
from websocket import create_connection
import json

from freenect import sync_get_depth as get_depth
from freenect import sync_get_video as get_video
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from  recognizer import Streamer

class Streamer_kinect(Streamer):
    def __init__(self,isKinnect=False,_detector="face_detection_model",\
          _emb_model = "face_detection_model/openface_nn4.small2.v1.t7", \
            _recognizer = "output/recognizer.pickle",
            _le = "output/le.pickle",_confidence=0.5):
            super().__init__(_detector,_emb_model,_recognizer,_le,_confidence)

    def main_loop_kinnect(self):
        print("[INFO] starting video stream...")
        # vs = VideoStream(src=0).start()
        # time.sleep(2.0)

        # start the FPS throughput estimator
        self.fps = FPS().start()
        cv2.namedWindow('Depth')
        cv2.namedWindow('Frame')
        # loop over frames from the video file stream
        while True:
            # grab the frame from the threaded video stream
            # print("here")
            frame = get_video()
            depth_frame = cv2.GaussianBlur(get_depth(), (5, 5), 0)
            cv2.imshow('Depth', depth_frame)
            # print(np.array(depth_frame).shape)

            # input()
            # resize the frame to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            frame = imutils.resize(frame, width=600)
            depth_frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            self.detector.setInput(imageBlob)
            detections = self.detector.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > self.confidence:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    midX = (startX + endX) // 2
                    midY = (startY + endY) // 2
                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    masked = np.ma.masked_equal(depth_frame[startY:endY, startX:endX], 0)
                    depth_face = masked.mean()
                    unmask = np.mean(depth_frame[startY:endY, startX:endX])
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                        (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    self.embedder.setInput(faceBlob)
                    vec = self.embedder.forward()

                    # perform classification to recognize the face
                    preds = self.recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = self.le.classes_[j]
                    # print(name)       
                    # draw the bounding box of the face along with the
                    # associated probability
                    faceWidth = abs(startX-endX)        
                    depth = self.scale*4*self.faceSizes[name]/faceWidth
                    #depth=1;
                    text = "{}: {:.2f}% d:{:.1f}".format(name, proba * 100,depth_face)
                    text = "{} rel :{:.1f} abs: {:.1f}".format(name,depth,unmask)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    
                    
                    # self.client.send_message("/faces", [name,int(midX),int(depth*10)] )
                    wsString = json.dumps([name,int(midX),int(depth*10)])
                    self.ws.send(wsString)

            # update the FPS counter
            self.fps.update()

            # show the output frame
            # print(startX-endX)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            if key == ord("n"):
                self.add_images(test_depth.get_video)

        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

        # do a bit of cleanup
        cv2.destroyAllWindows()
  

# s = Streamer()
# s.main_loop()