# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.0.0] - 2019-03-20
### Added
- Added face recognization and  face boundary detection [@nelson]
    * Using transfer learning and triplet loss on the embeddings to classify faces with few examples
    * Experimented other features and other models for the same, and using HOG features, speeds up real-time 
- Added video to training picture script [@eric](https://github.com/em370)
    * Frame-by-Frame splitting of a video, to store the train image for a person
- Visualizing the marauders map as a web-app [@supriya](https://github.com/supriyanaidu)
    * Created layout in p5js
- Setting up kinect[@ashwin](https://github.com/ashwinroot)
    * Connected kinect and visualizing depth map
    