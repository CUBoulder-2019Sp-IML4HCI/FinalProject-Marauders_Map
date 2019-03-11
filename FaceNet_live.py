from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
# import matplotlib.pyplot as plt

import os

import cv2

faces = dict()

print("="*20)
print("Loading model")
print("="*20)
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# Load weights
from keras.models import model_from_json
model.load_weights('weights/vgg_face_weights.h5')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)




def loadFace():
    face_path = "faces/"
    for f in os.listdir(face_path):
        print("Loading {} ".format(f))
        name = f.split('.')[0]
        f_path = face_path+f
        faces[name] = vgg_face_descriptor.predict(preprocess_image(f_path))[0,:]

def verifyFace(img1, img2):
    epsilon = 0.40
    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0,:]
    
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    
    print("Cosine similarity: ",cosine_similarity)
    print("Euclidean distance: ",euclidean_distance)
    
    if(cosine_similarity < epsilon):
        print("verified... they are same person")
    else:
        print("unverified! they are not same person!")
    
    # f = plt.figure()
    # f.add_subplot(1,2, 1)
    # plt.imshow(image.load_img(img1))
    # plt.xticks([]); plt.yticks([])
    # f.add_subplot(1,2, 2)
    # plt.imshow(image.load_img(img2))
    # plt.xticks([]); plt.yticks([])
    # plt.show(block=True)
    print("-----------------------------------------")

def find_face(img1,epsilon=0.4):
    epsilon = epsilon
    query = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
    # todo: https://stackoverflow.com/a/10834984
    cosine_similarity = [findCosineSimilarity(x,query) for i,x in faces.items()]
    print (cosine_similarity)
    #todo : make it faster
    for i in range(len(cosine_similarity)):
        if cosine_similarity[i] < epsilon:
            cosine_similarity[i] = 0
    
    idx = cosine_similarity.index(max(cosine_similarity))
    try:
        return list(faces)[idx] # or sorted(dic)[n] if you want the keys to be sorted
    except IndexError:
        return "unknown"


loadFace()

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

#loading webcam
cap = cv2.VideoCapture(0)
ret,frame = cap.read()

frame_freq = 0
while(True):
    ret,frame = cap.read() # return a single frame in variable `frame
    # cv2.imshow('img1',frame) #display the captured image
    frame_freq += 1
    # if frame_freq % 5 ==0:
        # frame_freq = 0
    cv2.imwrite('images/c1.png',frame)
    name = find_face("images/c1.png")
    print(name)
    cv2.putText(frame,name, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)  
    cv2.imshow('img1',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
        cv2.imwrite('images/c1.png',frame)
        cv2.destroyAllWindows()
        break

cap.release()