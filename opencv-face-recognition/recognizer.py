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


class Streamer():
    def __init__(self,_detector="face_detection_model",\
          _emb_model = "face_detection_model/openface_nn4.small2.v1.t7", \
            _recognizer = "output/recognizer.pickle",
            _le = "output/le.pickle",_confidence=0.5):
            self._detector_path = _detector
            self._emb_model = _emb_model
            self.confidence =_confidence
            self.scale = 3/5
            self.client = udp_client.SimpleUDPClient("localhost", 8999)
            self.detector = self._load_serialized_model()
            self.embedder = self._load_face_recognizer()
            self.recognizer = pickle.loads(open(_recognizer, "rb").read())
            self.le = pickle.loads(open(_le, "rb").read())
            with open('faceSizes.pickle', 'rb') as handle:
                self.faceSizes = pickle.load(handle)
            self.tick = 0
            self.vs =
    def add_face_size(self,name):
        self.faceSizes[name] = 155
        with open('faceSizes.pickle', 'wb') as handle:
            pickle.dump(self.faceSizes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_serialized_model(self):
        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join([self._detector_path, "deploy.prototxt"])
        modelPath = os.path.sep.join([self._detector_path,"res10_300x300_ssd_iter_140000.caffemodel"])
        print(protoPath,modelPath)
        return cv2.dnn.readNetFromCaffe(protoPath,modelPath)

    def _load_face_recognizer(self):
        # load our serialized face embedding model from disk
        print("[INFO] loading face recognizer...")
        return cv2.dnn.readNetFromTorch(self._emb_model)

    def label_image(self, image):
        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(image, width=600)
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
                faceWidth = abs(startX-endX);
                depth = self.scale*4*self.faceSizes[name]/faceWidth
                #depth=1;
                text = "{}: {:.2f}% d:{:.1f}".format(name, proba * 100,depth)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


                self.client.send_message("/faces", [name,int(midX),int(depth*10)] )
            cv2.imshow("Frame", frame)


    def extract_training_faces(self, image, name):
        frame = imutils.resize(image, width=600)
        (h, w) = frame.shape[:2]
        if time.time() - self.tick < 2:
            if time.time() - self.tick < .5:
                cv2.rectangle(frame, (0,0), (w, h),
                    (0, 255, 0), 10)
            cv2.imshow("Hello Friend", frame)
            return 0

        cv2.imshow("Hello Friend", frame)

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > self.confidence:
                path = os.path.join("dataset",name,f"{name}_{time.time()}.jpeg")
                cv2.imwrite(path, image)
                self.tick = time.time()
        if time.time() - self.tick < 1:
            return 1
        return 0


# s = Streamer()
# s.main_loop()
