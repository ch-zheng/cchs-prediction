import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score

# Description: Ridge regression classifier
# Accuracy: 81.94%

# Load data
samples = np.load('data/samples.npy')
labels = np.load('data/labels.npy')

# Train & Evaluate model
model = RidgeClassifier()
scores = cross_val_score(model, samples, labels, cv=100)
print("Average accuracy:", sum(scores) / len(scores))
