#from recognizer_kinect import Streamer as S_1
from recognizer import Streamer
from embedder import Embedder
from trainer import ModelTrainer
from imutils.video import VideoStream
from imutils.video import FPS
import time
import cv2
import os, shutil

def cleanup_output():
    for file in os.listdir("output"):
        os.remove(os.path.join("output", file))

cleanup_output()
e = Embedder()
e.detect_faces_save()
e.serialize_encodings()

t = ModelTrainer()
t.train_and_save()

s = Streamer()

# loop over frames from the video file stream
def main_loop(vs, fps, streamer):
    while True:
        frame = vs.read()

        streamer.label_image(frame)

        # update the FPS counter
        fps.update()

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            return
        elif key == ord("n"):
            streamer = new_loop(vs, fps, streamer)

def new_loop(vs, fps, s):
    name = input("What is your name? ")
    s.add_face_size(name)
    path = os.path.join("dataset", name)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    detections = 0
    while detections < 6:
        frame = vs.read()
        detections += s.extract_training_faces(frame, name)
        fps.update()

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return Streamer()

    cleanup_output()

    e = Embedder()
    e.detect_faces_save()
    e.serialize_encodings()

    t = ModelTrainer()
    t.train_and_save()

    cv2.destroyWindow("Hello Friend")

    S = Streamer()
    return S

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

main_loop(vs, fps, s)

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
