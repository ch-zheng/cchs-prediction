# Local
from mlp import MLP
import utils
import transform

import statistics
from pathlib import Path
import numpy as np

# Cross-validate MLP
X = np.load('data/individualized/samples.npy')
y = np.load('data/individualized/labels.npy')
X, y = utils.oversample(X, y)
X, y = utils.augment(X, y, 3)
X = transform.relativize(X, 3)
X = transform.normalize(X, 3)
splits = utils.split(10, X, y)
X = np.delete(X, 0, 1)
splits = utils.create_splits(X, y, splits)

# Evaluation
scores = []
for i, split in enumerate(splits):
    training = split[0]
    test = split[1]
    model = MLP(X.shape[1])
    model.fit(training[0], training[1])
    score = model.score(test[0], test[1])
    scores.append(score)
    print('Score', i, '=', score)
    model.save(Path('pretrained', f'split{i}.pt'))
print(scores)
print(statistics.mean(scores), statistics.stdev(scores))
