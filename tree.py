import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Description: Decision Tree
# Accuracy: 70.04%

# Load data
samples = np.load('data/samples.npy')
labels = np.load('data/labels.npy')

# Train & Evaluate model
model = DecisionTreeClassifier()
scores = cross_val_score(model, samples, labels, cv=100)
print("Average accuracy:", sum(scores) / len(scores))
