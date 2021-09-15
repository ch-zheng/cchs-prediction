import utils
import zoo
from pathlib import Path
import json
import numpy as np

# Load data
samples = np.load('data/individualized/samples.npy')
labels = np.load('data/individualized/labels.npy')
X, y = utils.augment(samples, labels)
splits = utils.split(10, X, y)

# Model list
models = {
    'ridge': zoo.Ridge(),
    'logistic': zoo.Logistic(),
    'svm': zoo.SVM(),
    'neighbors': zoo.Neighbors(),
    'bayes': zoo.NaiveBayes(),
    'tree': zoo.DecisionTree()
}

for name, model in models.items():
    # Hyperparameter search
    parameters = model.grid_search(X, y, cv=splits)
    print(name.capitalize(), 'parameters:', parameters)
    # Save hyperparameters
    with open(Path('hyperparameters', name + '.json'), 'w') as f:
        json.dump(parameters, f)
