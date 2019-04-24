#Maurader's Map Final Report 

Eric Minor, Supriya Naidu, Ashwin Sankaralingam, Nelson Mitchell, Talia Krause

## Project Desctiption

The goal is to use computer vision to track people in a room or rooms and place them onto a 2D grid, marking their location. This technology would be useful for anyone trying to find someone else in room/building, plus it would be fun to have a real life version of the Marauder's Map from Harry Potter.

A common problem in large buildings is trying to figure out the location of certain people who you need to talk to or deliver things too. Using cameras situated throughout a building, a map with the location of everyone in the building could be constructed, making locating individuals easier.

Specifically, this technology would be used in a collaborative workspace where people want to be available to help others. Some jobs require people to spend time in a variety of places. For instance, a researcher might spend time in a lab, in a machine shop creating equipment, or in a computer room analyzing data. Providing a real-time map of workers and their locations would make it easier for colleagues to find one other. This would not compromise privacy as a worker is supposed to be available while at work and not hidden. It would merely save a colleague a few minutes of searching around to find the person they needed to talk to. The technology could also be used to keep track of which professionals are currently available in a given helproom or other place where assistance is expected to be given. If a person marks themselves to be tracked in such a room, it is expected that they are there to render assistance and thus would not want to be hidden.


## Challenges Faced in the project
We had 2 main problems in the entire project: 

* Face tracking and identification
* Depth estimation

// TO change ? Following are some of the techniques we tried, and will be explaining why we chose some. 

### Face tracking and identification : 

In order to get a personalized effect for the user to use our system, we planned to identify each person seperately instead of tagging them as person `x`. This was quiet challenging to do in real time. After some digging we planned to use transfer learning over resnet layers to utilize the convolutional layer to produce the abstract representation of the face.

1)  ❌Tranfer learning using resnet :  Renet [?] is one state of the art object recognition network. Using Resnet weights we tried train the last couple of fully connected layer with our trainable faces. Although it was accuracte, and capable of identifying if a person is present in the scene, it was not able to localize the presence, and it was computation intesive everytime we need to retrain a new person. It also had a very bad fps, compared to other models we tried. 

2)  ✅ Single Shot detector : OpenCV has released a new way of identifying objects and localizing those objects using transfer learning principles on facenet architecture. Facenet is one of the state-of-art facial recognition networks, capable of identifying face unaffected by pose, illumination and contrast.

    * After loading the facenet architecture with pretrained weights, we can simply perform a single forward propogation to get a list of localized regions of interst with a probability score depicting how probable it is face. This stage is useful for localization of faces, and to detect if it is a face or not. 

    * Facenet model computes a 128 dimension embedding that quantifies as an abstract representation of the face. Each person's 128 dimension would differ from another person, and if we could linearly classify these embeddings in


    After getting these regions of interests, it is sent to another convolutional network which have weights of resnet layer. The 
    
    After removing the last FC layer of the resnet, we send the 128 dimensional vector to a SVM to classify into the different labels representing various people trained in the system.

    We also aded a common label for unknown user, and added a bunch of celebrities pictures.

## Technologies used


## Risk to Failure


## Transfer Learning


## Face Tracking and Neural Network


## References
[]