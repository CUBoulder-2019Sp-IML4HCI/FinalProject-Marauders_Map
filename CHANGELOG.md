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
- Project manager [@talia](link)
    * 3d modeling and printing a rotatable stand for the kinect
    * Found and troubleshot bug related to use on Windows machines
    * Tested installation process on fresh macOS environments
    * Supervised other team members

- Results :[pictures](https://github.com/CUBoulder-2019Sp-IML4HCI/FinalProject-Marauders_Map/tree/master/prototype)
- Youtube :
  * [Demo](https://youtu.be/TrNAKGQKF4Q)
  * [How to Run](https://youtu.be/pv_LqElPHjc)

    
## [1.1.0] - 2019-04-03
### Added
- Refactored into classes[@ashwin](https://github.com/ashwinroot)
- Kinect connected to live feed with depth[@ashwin](https://github.com/ashwinroot) and [@talia](https://github.com/ashwinroot)
- Visualizing 2d map in pygame [@ashwin](https://github.com/ashwinroot)
- Made heroku website and added websocket code to the recognizer. just_runC.py runs without kinect code [@eric](https://github.com/em370)
- Ported and setup processing visualizer on web-app[@supriya](https://github.com/supriyanaidu)
- fixed the web app and visualizer [@eric](https://github.com/em370)



