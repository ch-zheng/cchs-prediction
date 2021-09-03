import numpy as np
from sklearn import neighbors
from sklearn.model_selection import cross_val_score

# Description: K-Nearest Neighbors
# Accuracy: 71.52%

# Load data
samples = np.load('data/samples.npy')
labels = np.load('data/labels.npy')

# Train & Evaluate model
model = neighbors.KNeighborsClassifier(n_jobs=-1)
scores = cross_val_score(model, samples, labels, cv=100)
print("Average accuracy:", sum(scores) / len(scores))
