# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.0.0] - 2019-03-20
### Added
- Added face recognization and  face boundary detection [@nelson](https://github.com/nelsonnn)
    * Using transfer learning and triplet loss on the embeddings to classify faces with few examples
    * Experimented other features and other models for the same, and using HOG features, speeds up real-time 
- Added video to training picture script [@eric](https://github.com/em370)
    * Frame-by-Frame splitting of a video, to store the train image for a person
    * Depth estimating using face size
    * Created scripts to simplify running and creating a model
- Visualizing the marauders map as a web-app [@supriya](https://github.com/supriyanaidu)
    * Created layout in p5js
    * updated requirements to ease setup
- Setting up kinect[@ashwin](https://github.com/ashwinroot)
    * Connected kinect and visualizing depth map
    * experimented with model types
    * Testing and video creation
- Project manager [@talia](https://github.com/takr3985)
    * worked with ashwin on enter name GUI pop up
    * working on commands for taking pictures of user (telling them to move face) 
    * Found and troubleshot bug related to use on Windows machines
    * Tested installation process on fresh macOS environments
    * Supervised other team members

- Results :[pictures](https://github.com/CUBoulder-2019Sp-IML4HCI/FinalProject-Marauders_Map/tree/master/prototype)
- Youtube :
  * [Demo](https://youtu.be/TrNAKGQKF4Q)
  * [How to Run](https://youtu.be/pv_LqElPHjc)

    
## [1.1.0] - 2019-04-04
### Added
- Refactored into classes[@ashwin](https://github.com/ashwinroot)
- Kinect connected to live feed with depth[@ashwin](https://github.com/ashwinroot) and [@talia](https://github.com/ashwinroot)
- Visualizing 2d map in pygame [@ashwin](https://github.com/ashwinroot) : not used for final iteration
- Made heroku website and added websocket code to the recognizer. just_runC.py runs without kinect code [@eric](https://github.com/em370)
- Ported and setup processing visualizer on web-app[@supriya](https://github.com/supriyanaidu)
- Training module for each user. Users able to add their faces [@nelson](https://github.com/nelsonnn)


## [2.0.0] - 2019-04-11
- Trying out kinect libraries and experimenting with various version of depth map @ashwin](https://github.com/ashwinroot) and [@talia](https://github.com/ashwinroot)
- Using angles,scales for relative distance [@nelson](https://github.com/nelsonnn) and [@eric](https://github.com/em370)
- Fixed the web app and visualizer [@supriya](https://github.com/supriyanaidu)
- Using camera.json config for multiple camera support [@eric](https://github.com/em370)


## [3.0.0] - 2019-04-18
- Removed kinect support as it was buggy
- Centroid tracking: tracking if a person enters a scene [@ashwin](https://github.com/ashwinroot)
- Working on google drive interaction for a distributed system, for communicating a newly trained person  [@nelson](https://github.com/nelsonnn) 
- Redid website to support multiple cameras in multiple canvas [@supriya](https://github.com/supriyanaidu)
- Fixed the data pipeline. Scaled-Position and person name calculation was moved to client side.[@eric](https://github.com/em370)
- tkinter prompt to add a new user [@talia](https://github.com/takr3985)

## [4.0.0] - 2019-04-25
- Adding thresholds for distances and feet angle calculation[@eric](https://github.com/em370)
- Redoing the website with the map representing the expo room along with real-time scaling[@supriya](https://github.com/supriyanaidu)
- Directions to train a new face on the frame. [@talia](https://github.com/takr3985)
- Optimization and increasing the confidence in the classification of the person [@ashwin](https://github.com/ashwinroot)
- Poster : all
- Video : [@nelson](https://github.com/nelsonnn) 

## [5.0.0] - 2019-05-02
- Redoing the map for class [@supriya](https://github.com/supriyanaidu),[@eric](https://github.com/em370)
- Report : all


