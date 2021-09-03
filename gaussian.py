import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score

# Description: Gaussian Process Classifier

# Load data
samples = np.load('data/samples.npy')
labels = np.load('data/labels.npy')

# Train & Evaluate model
model = GaussianProcessClassifier()
scores = cross_val_score(model, samples, labels, cv=20)
print("Average accuracy:", sum(scores) / len(scores))
