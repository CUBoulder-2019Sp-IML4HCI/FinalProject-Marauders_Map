from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

class ModelTrainer():
    def __init__(self,_emb_path="output/embeddings.pickle",\
        _recognizer_path="output/recognizer.pickle",
        _le = "output/le.pickle"):
        '''
        @param : _emb_path : Path to the embeddings saved in prev step
        @param : _recognizer : Path where the recognizer is saved. It is a SVM
        @param : _le : Label encoder path
        '''
        self._emb_path = _emb_path
        self._recognizer_path = _recognizer_path
        self._le_path = _le
        self.recognizer = None
        self.data = self._load_face_embeddings()
        self.labels = self.encode_labels()

    def _load_face_embeddings(self):
        # load the face embeddings
        return pickle.loads(open(self._emb_path, "rb").read())
    
    def encode_labels(self):
        # encode the labels
        print("[INFO] encoding labels...")
        self.le = LabelEncoder()
        labels = self.le.fit_transform(self.data["names"])
        return labels
    
    def train_and_save(self):
        #todo : play with gaussian kernel and alpha
        print("[INFO] training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(self.data["embeddings"], self.labels)
        # write the actual face recognition model to disk
        f = open(self._recognizer_path, "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        # write the label encoder to disk
        f = open(self._le_path, "wb")
        f.write(pickle.dumps(self.le))
        f.close()

t = ModelTrainer()
t.train_and_save()
