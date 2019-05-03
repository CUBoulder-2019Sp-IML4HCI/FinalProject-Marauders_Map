# Marauders_Map : Facial Recognition and Data Aggregation for Multi-Camera Person Tracking

This is part of final project for CSCI/ATLS 4889/5880 Machine Learning for Human-Computer Interaction, advised by Ben Shapiro. 

## Team
* Talia Krause:[@takr3985](https://github.com/takr3985)
* Eric Minor:[@em370](https://github.com/em370)
* Nelson Mitchell:[@nelsonnn](https://github.com/nelsonnn)
* Supriya Naidu:[@supriyanaidu](http://github.com/supriyanaidu)
* Ashwin Sankaralingam:[@ashwinroot](https://github.com/ashwinroot)

## Proposal : [link](https://github.com/CUBoulder-2019Sp-IML4HCI/FinalProject-Marauders_Map/blob/master/proposal.md)

## Final_report : [link](https://github.com/CUBoulder-2019Sp-IML4HCI/FinalProject-Marauders_Map/blob/master/final_report.md)

## Poster : [link](https://github.com/CUBoulder-2019Sp-IML4HCI/FinalProject-Marauders_Map/blob/master/Poster.pdf)

## Final Release
- Youtube : [youtube]()  
- Website : https://atlasmaraudermap.herokuapp.com/map


## Usage
This projects uses python3.6.

For installing the requirements needed in the project install the requirements.txt using :

```
pip install -r requirements.txt
```

In order to use the facial tracking, you need run the recognizer locally. Navigate to the folder `FinalProject-Marauders_Map/opencv-face-recognition` and run the command 

```
python3 just_run.py
```

- If you need to retrain a new model with a new set of embeddings add option -e

After a few seconds, a window should appear showing a live feed from your webcam with boxes around all detected faces. Initially, your face should appear as unknown or be incorrectly labeled. In order to train the model to recognize your face, press `n`. A dialog box will appear where you can enter your name. Make sure no other faces are in frame and press enter. Slightly tilt your face at each prompt. When training is finished, the program will automatically reload and your face should be correctly labeled. If you feel the face detected is not right click on `c` to refresh the computer prediction and it will restart the process of classification.

To view the location of everyone detected, navigate to https://atlasmaraudermap.herokuapp.com/map . A map should display feet for everyone detected within the last second. If you want to change the location and orientation of your camera, locally modify the file FinalProject-Marauders_Map/opencv-face-recognition/config.json. Camera angle changes what direction your camera is facing when translating camera coordinates to display coordinates. 0 degrees points to the bottom of the map and angles progress counterclockwise. CameraX and CameraY refer to the camera's x and y coordinates on the map. (0,0) is the top left corner of the map.

Example of config.json : 

```
{
	"camera": 5,
	"scale": 0.601748352,
	"fov": 70,
	"cameraX":5.58,
	"cameraY": 3.45,
	"cameraAngle": 0 ,
	"cameraDeclination": 0
}
```



