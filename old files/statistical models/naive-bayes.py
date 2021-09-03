import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# Description: Gaussian Process Classifier
# Accuracy: 53.77%

# Load data
samples = np.load('data/samples.npy')
labels = np.load('data/labels.npy')

# Train & Evaluate model
model = GaussianNB()
scores = cross_val_score(model, samples, labels, cv=100)
print("Average accuracy:", sum(scores) / len(scores))
