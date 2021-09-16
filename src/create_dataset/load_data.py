import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class data_frame():
    def __init__(self, folder_path, func):
        func(self, folder_path)

    def init_stratify(self, folder_path):
        self.X_train = pd.read_csv(os.path.join(folder_path, 'train/samples.csv')).drop('Unnamed: 0', axis=1).drop('subject', axis=1)
        self.y_train = pd.read_csv(os.path.join(folder_path, 'train/labels.csv')).drop('Unnamed: 0', axis=1)
        self.X_test = pd.read_csv(os.path.join(folder_path, 'test/samples.csv')).drop('Unnamed: 0', axis=1).drop('subject', axis=1)
        self.y_test = pd.read_csv(os.path.join(folder_path, 'test/labels.csv')).drop('Unnamed: 0', axis=1)

    def init_base(self, folder_path):
        X = pd.DataFrame(np.load(os.path.join(folder_path, "samples.npy")))
        y = pd.DataFrame(np.load(os.path.join(folder_path, "labels.npy")))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

    def init_phase1(self, folder_path):
        X = pd.DataFrame(np.load(os.path.join(folder_path, "samples.npy")))
        y = pd.DataFrame(np.load(os.path.join(folder_path, "labels.npy")))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
    

def load_stratify(src, k):
    data = []
    for i in range(0, k):
        data.append(data_frame(os.path.join(src, str(i)), data_frame.init_stratify))
    return data

def load_base(src, k):
    data = []
    for i in range(0, k):
        data.append(data_frame(src, data_frame.init_base))
    return data

def load_phase1(src, k):
    data = []
    for i in range(0, k):
        data.append(data_frame(src, data_frame.init_phase1))
    return data