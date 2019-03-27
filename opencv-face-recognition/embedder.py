
'''
python extract_embeddings.py --dataset dataset \
	--embeddings output/embeddings.pickle \
	--detector face_detection_model \
	--embedding-model face_detection_model/openface_nn4.small2.v1.t7
'''	
from imutils import paths
import numpy as np
# import argparse
import imutils
import cv2
import os
import pickle


class Embedder(object):
    def __init__(self,_dataset="dataset",_embeddings="output/embeddings.pickle", \
                _detector="face_detection_model",_embedding_model="face_detection_model/openface_nn4.small2.v1.t7", \
                    _confidence = 0.4):
        '''
        @param : _dataset : path to input directory of faces + images
        @param : _embeddings : path to output serialized db of facial embeddings
        @param : _detector : path to OpenCV's deep learning face detector
        @param : _embedding_model : path to OpenCV's deep learning face embedding model
        '''
        self.dataset = _dataset
        self.embeddings = _embeddings
        self.detector_path =_detector
        self.embeddings_model = _embedding_model
        self.confidence = _confidence
        self.detector = self._load_serialized_model()
        self.embedder = self._load_face_recognizer()

    def _load_serialized_model(self):
        # load our serialized face detector from disk
        print("[INFO] loading face recognizer...")
        protoPath = os.path.sep.join([self.detector_path, "deploy.prototxt"])
        modelPath = os.path.sep.join([self.detector_path,"res10_300x300_ssd_iter_140000.caffemodel"])
        return cv2.dnn.readNetFromCaffe(protoPath,modelPath)

    def _load_face_recognizer(self):
        # load our serialized face embedding model from disk
        print("[INFO] loading face recognizer...")
        return cv2.dnn.readNetFromTorch(self.embeddings_model)

    def detect_faces_save(self):
        # grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(self.dataset))

        # initialize our lists of extracted facial embeddings and
        # corresponding people names
        self.knownEmbeddings = []
        self.knownNames = []

        # initialize the total number of faces processed
        self.total = 0

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1,
                len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            # load the image, resize it to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
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
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI and grab the ROI dimensions
                    face = image[startY:endY, startX:endX]
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

                    # add the name of the person + corresponding face
                    # embedding to their respective lists
                    self.knownNames.append(name)
                    self.knownEmbeddings.append(vec.flatten())
                    self.total += 1
                
    def serialize_encodings(self):
        print("[INFO] serializing {} encodings...".format(self.total))
        data = {"embeddings": self.knownEmbeddings, "names": self.knownNames}
        f = open(self.embeddings, "wb")
        f.write(pickle.dumps(data))
        f.close()

# e = Embedder()
# e.detect_faces_save()
# e.serialize_encodings()

