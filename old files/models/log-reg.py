import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Description: Logistic regression classifier
# Accuracy: 78.00%

# Load data
samples = np.load('data/samples.npy')
labels = np.load('data/labels.npy')

# Train & Evaluate model
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, samples, labels, cv=100, n_jobs=-1)
print("Average accuracy:", sum(scores) / len(scores))
