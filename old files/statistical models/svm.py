import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

# Description: Support Vector Machine
# Accuracy: 68.68%

# Load data
samples = np.load('data/samples.npy')
labels = np.load('data/labels.npy')

# Train & Evaluate model
model = svm.SVC()
scores = cross_val_score(model, samples, labels, cv=100, n_jobs=-1)
print("Average accuracy:", sum(scores) / len(scores))
