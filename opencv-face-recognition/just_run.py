from recognizer import Streamer as S_2
from embedder import Embedder
from trainer import ModelTrainer
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embedd",action="store_true",
	help="to embedd or not to embedd")
ap.add_argument("-t", "--train",action="store_true",
	help="to train or not to train")
ap.add_argument("-k", "--kinect",action="store_true",
	help="to train or not to train")
args = vars(ap.parse_args())

if args['embedd']:
    e = Embedder()
    e.detect_faces_save()
    e.serialize_encodings()
if args['embedd'] or args['train']:
    t = ModelTrainer()
    t.train_and_save()
if args['kinect']:
    from recognizer_kinect import Streamer_kinect as S_1
    s= S_1()
    s.main_loop_kinnect()
else:
    s = S_2()
    s.main_loop()
