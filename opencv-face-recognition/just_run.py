from recognizer import Streamer
from embedder import Embedder
from trainer import ModelTrainer
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embedd",action="store_true",
	help="to embedd or not to embedd")
ap.add_argument("-t", "--train",action="store_true",
	help="to train or not to train")
args = vars(ap.parse_args())

if args['embedd']:
    e = Embedder()
    e.detect_faces_save()
    e.serialize_encodings()
if args['embedd'] or args['train']:
    t = ModelTrainer()
    t.train_and_save()

s = Streamer()
s.main_loop()
