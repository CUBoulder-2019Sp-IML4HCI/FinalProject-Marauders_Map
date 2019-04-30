# Maurader's Map Final Report 

Eric Minor, Supriya Naidu, Ashwin Sankaralingam, Nelson Mitchell, Talia Krause

## Project Desctiption

   Harry Potter captured the imagination of children everywhere with its depiction of a world where magic could be used to accomplish extraordinary things. One item in particular, the Marauder's Map, allowed the user to see the location of everyone at school on a simple piece of parchement. Although the ethics of tracking people without consent are questionable, the ability to see everyone's location is an incredible boon. As long as everyone invovled has consented and is comfortable with having their location available to others, such a map would save huge quantities of time in workspaces that both encourage collaboration and allow people to move around to a variety of locations. The process of finding a knowledgeable colleague would be reduced to finding a marking on a map. Sorcerous abilities notwithstanding, the creation of such a map is possible with a synergistic usage of machine learning, conventional computer vision techniques, some trigonometry, and an aggregrative webserver.

   In simplest terms, the goal of the project is to use computer vision and machine learning to track people in a room or rooms and place them onto a 2D grid, marking their location. This technology would be useful for anyone trying to find someone else in room/building, plus it would be fun to have a real life version of the Marauder's Map from Harry Potter.

   A common problem in large buildings is trying to figure out the location of certain people who you need to talk to or deliver things to. Using cameras situated throughout a building, a map with the location of everyone in the building can be constructed, making locating individuals easier.

   Specifically, this technology would be used in a collaborative workspace where people want to be available to help others. Some jobs require people to spend time in a variety of places. For instance, a researcher might spend time in a lab, in a machine shop creating equipment, or in a computer room analyzing data. Providing a real-time map of workers and their locations would make it easier for colleagues to find one other. This would not compromise privacy as a worker is supposed to be available while at work and not hidden. It would merely save a colleague a few minutes of searching around to find the person they needed to talk to. The technology could also be used to keep track of which professionals are currently available in a given helproom or other place where assistance is expected to be given. If a person marks themselves to be tracked in such a room, it is expected that they are there to render assistance and thus would not want to be hidden.

   Collaboration focused workspaces rely upon being able to quickly and easily access the expertise of your coworkers. Messaging apps allow for quick digital communication, but it is often more efficient to simply have a conversation in person. The system we developed allows for consenting individuals to train a webcam system to identify and track their face in order to create a map of everyone’s location in a collaborative workspace. 

   Using a simple interface, a python script records images of a person’s face, extracts features, and uses transfer learning to create a recognizer. This can be done with multiple computers and cameras, which all connect via websockets to a Node.js server that aggregates the data and keeps a dictionary containing everyone’s position. The node server also serves a website that displays a map of everyone’s location.




## User Interface
   There are two components to the user interface. The first is the facial identification interface on each tracking computer. The second is map displayed over the internet at https://atlasmaraudermap.herokuapp.com/ .

   Each computer running the facial tracking software displays an image of what the computer sees along with a box around each identifed face and its label. By pressing n on the keyboard when interacting with this display, a dialogue will be opened prompting the user to enter their name. The camera will then record images of the person currently in front of the camera and label those images with the entered name. The algorithm is then trained to identify the new person. A user can also press c to reset the tracking algorithm if someone is mislabeled or press q to quit the program.
   
   The webserver which aggregrates all tracking data also services a website, where it will output the locations of detected people. By going to the afformentioned website, a user can view live tracking data in the form of a Marauder's map. The website can also play music and has a link to the project github page for interested parties.



## Challenges Faced in the project
The challenges in the project came from three main sources:

* Face tracking and identification
* Depth estimation and Absolute Position Estimation
* Data aggregration and Display


### Face tracking and identification 🧒🏻: 

In order to get a personalized effect for the user to use our system, we planned to identify each person seperately instead of tagging them as person `x`. This was quiet challenging to do in real time. After some digging we planned to use transfer learning over resnet layers to utilize the convolutional layer to produce the abstract representation of the face.

1)  ❌Tranfer learning using resnet :  Renet [?] is one state of the art object recognition network. Using Resnet weights we tried train the last couple of fully connected layer with our trainable faces. Although it was accuracte, and capable of identifying if a person is present in the scene, it was not able to localize the presence, and it was computation intesive everytime we need to retrain a new person. It also had a very bad fps, compared to other models we tried. 

2)  ✅ Single Shot detector : OpenCV has released a new way of identifying objects and localizing those objects using transfer learning principles on facenet architecture. Facenet is one of the state-of-art facial recognition networks, capable of identifying face unaffected by pose, illumination and contrast.

    * After loading the facenet architecture with pretrained weights, we can simply perform a single forward propogation to get a list of localized regions of interst with a probability score depicting how probable it is face. This stage is useful for localization of faces, and to detect if it is a face or not. 

    * Facenet model computes a 128 dimension embedding that quantifies as an abstract representation of the face. Each person's 128 dimension would differ from another person, and it would be possible to linearly seperate these embeddings in `nth` dimension to classify each person trained by the system.

    * Using SVM and linear classifier, we classify the detections based on the person's embeddings. As only the regions of interest is passed through the neural network, it is quiet fast in most of the CPU, and real time.


    * For each user, we have a trainable module, where they can enter their name and capture upto 5 pics, that will then undergo the embedding process for classification.

    //insert image after review

3) ✅ Centroid Tracking : Due to the use of SVM for classification, the classification was jittery as the faces that lie near to the margin was often confused. In order to increase the confidence in the classification, polling method was introduced to ID a particular face after a majority. In 40 consecutive frames, if a face has a majority name, then that face gets that name for the next 400 frames. Once again on the 400th frame, we flush the registered faces, as we wanted to the classification system to be active and correct itself if it got the face wrong the first time.

    * Centroid Tracking uses a simple method of comparing euclidean distances between various centroid of object on the scene. It is a effective algorithm, to track multiple moving objects in the scene without loosing track.

    * For every face in the scene, a centroid is identified and kept in memory. 

    * When a person moves in a subsequent frame, the euclidean distance will be smaller compared to other objects in the scene. If there was a new object in a subsequent frame, the system will assign a new ID. If an object is not present in the scene, after some buffer time, it will lose its ID and will be assigned a new ID when it enters the scene.

    //insert image after review

### Depth Estimation and Absolute Position Estimation📏: 
1) ❌Kinect Depth : We planned to use Kinect 360 to get the depth details on the face we had already localized using SSD algorithm. The depth map generated by kinect was not very accurate with respect to real world measures, which led us to drop the kinect to use for depth. Moreover since the discontinue of Kinect by Microsoft, the support for kinect seems to have reduced amongst the community.

2)  ✅ Face size and angle : Using the formula Depth ∝ (Actual Face Width)/(Face Width in Frame), the relative depth of face could be estimated. Each camera has to be manually calibrated by measuring the ratio at 1 meter and dividing by a constant to make the program output equal 1 meter when the face is at 1 meter. The calibration remains accurate at all distances. 

In order to translate the depth of a face into planar coordinates for usage in real-world mappings, the angle of the face in the camera frame was calculated (based on left-right centroid location). Multipling depth by the sine of the angle yields the x coordinate of the face, and multiplying the depth by the cosine of the angle yields the y coordinate of face, relative to the camera. Each camera contains its absolute position in a map, and its pointing angle. These are used to translate face locations to absolute locations in the map. Testing shows that this estimator for distance is accurate to ~2 meters, at which point the detector beings to fail

Data Aggregration and Display

### Data aggregration and Display: 
❌ Centralized Image Process: An initial idea for the project involved streaming all webcam video feeds to one machine which would contain the trained model. Although this method would slightly streamline the process, doing so would require a very powerful computer in order to process all the video data. As it stands, running the model for a single video feed used up approximately 60% of a 4th generation intel i7 processor as tested on a laptop computer. A centralized processing computer would need to handle a large number of video feeds, which would require an unfeasibly fast processor. Instead, a decentralized model where each computer analyzed its own video feed and sent face coordinates to a central server was adopted. This cut down on the amount of data the server needed to handle immensely, making it possible to use a free heroku server.

✅ Data Aggregration: The server is successfully able to collect and process face coordinates from several computers. The largest stress test we were capable of generating used five computers sending data to the server simultaneously, with each computer detecting at least one face. The server experienced no slowdown from this. When the server receives information about a face detection, it stores that face along with its last known location and the detecting camera in a dictionary If multiple cameras claim to detect the same person at multiple locations, the the server only uses data from the camera that originally detected the face. If a face is not detected from 1.5 seconds, the face is dropped from the dictionary and can be detected by another camera. The server also calculates and stores the distance traveled between locations and the direction.

✅ Display: Website Discussion


## Technologies and Libraries
1) Python
    * imutils (for video interfacing)
    * OpenCV (for face detection and webcam)
    * Caffe (for loading models)
    * websocket-client (for sending messages to website)
    * tkinter (for user input)
    
2) Node.js
    * websocket (for communication with detector and website)
    * express (for handling web requests)
    
3) Frontend
   * Semantic UI (for nice visuals)
   * p5 (sound and map drawing)
   * websocket (for communication with server)
   
   
   
## Risk faced and mitigated

* Real time face detection : We were facing the problem of the system to be slow to run as well as train new faces. To avoid that we introduced SSD(Single Shot Detector) utilizing facenet weights, that speeded up the detection to real-time.

* Personalized identification : Utilizing the facenet embeddings, and SVM classification, we were able to classify multiple of faces.

* Jittery face detections : Using SVM, we faced a problem of faces being confused when it was closer to the margin. To reduce the sensitivity we added a polling mechanism with Centroid tracking to avoid jitters in detection.

* website problems : The node server handles conflicts that result from multiple cameras transmitting face coordinates. Occasionally, two cameras will detect the same face and the server needs to decide which to display. The current implementation gives precendence to the first camera to detect the face. This can occassionally result in a person being displayed in the wrong location.


## Project Outcomes

We feel we learnt to develop an user end-to-end trainable computer vision system capable of training on each user and capture their world coordinates efficiently.

* Utiling transfer learning to reuse weight from an already trained classifier.

* Using SVM to classify a embedding input into new multiple classes to provide user trainable feature.

* On creating a system capable of loading, training, inferencing computer vision models in real time.

* Setting up heroku and git submodules for communicating with multiple projects

* Using websockets for communication.

* Using a node server to aggregrate data from multiple cameras

* Using a website to display detections from multiple cameras



## References
[1]Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

[2] J. C. Nascimento, A. J. Abrantes and J. S. Marques, "An algorithm for centroid-based tracking of moving objects," 1999 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings. ICASSP99 (Cat. No.99CH36258), Phoenix, AZ, USA, 1999, pp. 3305-3308 vol.6.

[3] Tutorials from https://www.pyimagesearch.com/
