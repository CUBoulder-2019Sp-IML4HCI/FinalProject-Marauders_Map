'''
recognize along with depth
'''

from centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse

import imutils
import pickle
import time
import cv2
import os
import math
from pythonosc import udp_client
from websocket import create_connection
import json
import shutil
from embedder import Embedder
from trainer import ModelTrainer
from tkinter import *

import collections

class Streamer(object):
    def __init__(self,_detector="face_detection_model",\
          _emb_model = "face_detection_model/openface_nn4.small2.v1.t7", \
            _recognizer = "output/recognizer.pickle",
            _le = "output/le.pickle",_confidence=0.5):
            self._detector_path = _detector
            self._emb_model = _emb_model
            self.confidence =_confidence
            #self.scale = 3/5
            #self.client = udp_client.SimpleUDPClient("localhost", 3000)
            self.ws= create_connection("ws://rhubarb-tart-58531.herokuapp.com/")
            #self.ws= create_connection("ws://localhost:3000") #use this for local testing
            self.detector = self._load_serialized_model()
            self.load_face_datas()
            with open('faceSizes.pickle', 'rb') as handle:
                self.faceSizes = pickle.load(handle)
            self.vs = None
            self.fps = None
            self.tick = 0
            with open('config.json') as json_camera:
                data = json.load(json_camera)
                self.scale = data['scale']
                self.cameraNum = data['camera']
                self.fov = data['fov']
                self.cameraAngle = data['cameraAngle']
                self.cameraX = data['cameraX']
                self.cameraY = data['cameraY']
                self.declination = data['cameraDeclination']
			
            self.train_name = None
            self.set_up_tk()

            self.ct = CentroidTracker()

            self.user = dict()
            self.user_buffer = collections.defaultdict(list)
            self.user_threshold = 50

            #self.fov = 70 #degrees

    def load_face_datas(self,_recognizer = "output/recognizer.pickle",
            _le = "output/le.pickle"):
        self.embedder = self._load_face_recognizer()
        self.recognizer = pickle.loads(open(_recognizer, "rb").read())
        self.le = pickle.loads(open(_le, "rb").read())
        
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
    
    def _add_face_size(self,name):
        self.faceSizes[name] = 155
        with open('faceSizes.pickle', 'wb') as handle:
            pickle.dump(self.faceSizes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def add_cv_box(self,frame,box,centroid,objectID=None,name=None,percentage=None):
        '''
        Two ways to call the functions:
        1) you know the objectId and call with that
        2) you don't know the objectId, but know a person's name
        '''
        
        startX,startY,endX,endY = box
        midX,midY = centroid
        if name is None: #super confident with the user
            depth = self.get_depth(self.user[objectID],abs(endX-startX))
            text = "{} d={:.2f}".format(self.user[objectID],depth)
            name = self.user[objectID]
        else:
            depth = self.get_depth(name,abs(endX-startX))
            text = "{} d={:.2f} {:.2f}".format(name,depth,percentage)

        faceX,faceY = self.calc_angles(midX,midY,depth)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        id_tag = "{}".format(objectID)
        cv2.putText(frame, id_tag, (midX - 10, midY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)     
        cv2.circle(frame, (midX, midY), 4, (0, 255, 0), -1)
        self.send_message(name,faceX,faceY)

    def add_user(self,objectID,name):
        '''
        buffers until is confident on the person 
        #TODO : to keep on removing a person every 30 seconds
        '''
        if objectID not in self.user:
            self.user_buffer[objectID].append(name)

        if len(self.user_buffer[objectID]) >= self.user_threshold:
            self.user[objectID] = max(set(self.user_buffer[objectID]), key=self.user_buffer[objectID].count)
            del self.user_buffer[objectID]
    
    def get_depth(self,name,faceWidth):
        '''
        returns the depth for that name
        '''
        return self.scale*self.faceSizes[name]/faceWidth

    def calc_angles(self,midX,midY,depth):
        '''
        return faceX,faceY used in website
        '''
        h,w = 337,600
        phiR = math.radians(self.declination)
        theta = self.fov*((w/2 - midX)/w) +self.cameraAngle
        thetaR = math.radians(theta) 
        faceY = math.cos(phiR)*math.cos(thetaR)*depth + self.cameraY
        faceX = math.cos(phiR)*math.sin(thetaR)*depth + self.cameraX
        return faceX,faceY
    
    def send_message(self,name,x,y):
        '''
        Sending data to website
        '''
        wsString = json.dumps([name,x,y,self.cameraNum])
        self.ws.send(wsString) #sending to website
                    
        
    def main_loop(self):
        print("[INFO] starting video stream...")
        self.vs = VideoStream(src=0).start()
        time.sleep(2.0)

        # start the FPS throughput estimator
        self.fps = FPS().start()

        # loop over frames from the video file stream
        while True:
            # grab the frame from the threaded video stream
            frame = self.vs.read()

            # resize the frame to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]
            
            # construct  a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)
            
            rects = list()
            centers= dict()

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
                    rects.append(box.astype("int"))
                    midX = int((startX + endX) / 2.0)
                    midY = int((startY + endY) / 2.0)
                    centers[(midX,midY)] = [startX, startY, endX, endY]
            
            detected_faces = None
            if len(rects)>0:
                detected_faces = self.ct.update(rects)
            if detected_faces:
                for (objectID,centroid) in detected_faces.items():
                    try:
                        bound_box = startX,startY,endX,endY = centers[(centroid[0],centroid[1])]
                    except KeyError:
                        continue
                    if objectID in self.user:
                        self.add_cv_box(frame,bound_box,centroid,objectID)
                    else:
                        #TODO : need to put this on each face after we get objects detected
                        # construct a blob for the face ROI, then pass the blob
                        # through our face embedding model to obtain the 128-d
                        # quantification of the face
                        # extract the face ROI
                        face = frame[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        # ensure the face width and height are sufficiently large
                        if fW < 20 or fH < 20:
                            continue
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                            (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        self.embedder.setInput(faceBlob)
                        vec = self.embedder.forward()

                        # perform classification to recognize the face
                        preds = self.recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = self.le.classes_[j]
                        self.add_user(objectID,name)
                        # print(name)       
                        # draw the bounding box of the face along with the
                        # associated probabilit
                        
                        
                        self.add_cv_box(frame,bound_box,centroid,None,name,proba)
                    
					
                    #wsString = json.dumps([name,int(midX),int(depth*10),self.cameraNum])
                    
            # update the FPS counter
            self.fps.update()
            cv2.imshow("Frame", frame)

            # show the output frame
            # print(startX-endX)
           
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            if key == ord("n"):
                self.user = dict()
                self.user_buffer = collections.defaultdict(list)
                self.add_images(self.get_frame)
            if key == ord("c"):
                self.user = dict()
                self.user_buffer = collections.defaultdict(list)
        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.vs.stop()


    def extract_training_faces(self, image, name):
        frame = imutils.resize(image, width=600)
        (h, w) = frame.shape[:2]
        if time.time() - self.tick < 2:
            if time.time() - self.tick < .5:
                cv2.rectangle(frame, (0,0), (w, h),
                    (0, 255, 0), 10)
            cv2.imshow("Frame", frame)
            return 0

        cv2.imshow("Frame", frame)

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
                # path = os.path.join('dataset',name,f'{name}_{time.time()}.jpeg')
                path = os.path.join('dataset',name,name+'_'+str(time.time())+'.jpeg')
                cv2.imwrite(path, image)
                self.tick = time.time()
        if time.time() - self.tick < 1:
            return 1
        return 0

    def cleanup_output(self):
        for file in os.listdir("output"):
            os.remove(os.path.join("output", file))

    def get_frame(self):
        return self.vs.read()

    def add_images(self,get_frame):
        self._run_tk_loop()
        name = self.train_name
        self._add_face_size(name)
        path = os.path.join("dataset", name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)
        detections = 0
        while detections < 6:
            frame = get_frame()
            detections += self.extract_training_faces(frame, name)
            # self.fps.update()

            key = cv2.waitKey(1) & 0xFF
            if detections == 2:
                text = "Turn right"
                cv2.putText(frame, text, (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            if key == ord("q"):
                return

        self.cleanup_output()

        e = Embedder()
        e.detect_faces_save()
        e.serialize_encodings()

        t = ModelTrainer()
        t.train_and_save()
        
        self.load_face_datas()
        # cv2.destroyWindow("Frame")
        
        return 

    def set_up_tk(self):
        self.root = Tk()
        self.root.title("Adding a user")
        self.root.geometry("640x640+0+0")
        heading= Label(self.root, text="Welcome!", font=("arial",40,"bold"), fg="steelblue") .pack()
        label1= Label(self.root, text="Enter your name: ",font=("arial",20,"bold"),fg="black").place(x=10,y=200)
        name=StringVar()
        entry_box= Entry(self.root, textvariable=name, width=25, fg="white" ,bg="steelblue").place(x=200, y=203)
        print("[INFO] Setting up tkinter for user input")
        def do_it():
            self.train_name = str(name.get()).strip().lower()
            self.root.destroy()
            self.root = None
        work= Button(self.root, text="ENTER", width=30, height=5, bg="steelblue", command=do_it).place(x=250,y=300)
        

    def _run_tk_loop(self):
        # print(self.root)
        if self.root is None:
            self.set_up_tk()
        self.root.mainloop()
