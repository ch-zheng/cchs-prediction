import numpy as np
import utils

samples = np.load('data/individualized/samples.npy')
labels = np.load('data/individualized/labels.npy')
#X, y = utils.augment(samples, labels)
X, y = samples, labels
splits = utils.split(10, X, y)
X = np.delete(X, 0, 1)
test_sets = []
for split in splits:
    training = set(split[0])
    test = set(split[1])
    test_sets.append(test)
for t in test_sets:
    print(test_sets[0].isdisjoint(t))
