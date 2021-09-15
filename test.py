import utils
import numpy as np

samples = np.load('data/original dataset/samples.npy')
labels = np.load('data/original dataset/labels.npy')
X, y = utils.augment(samples, labels)
splits = utils.split(10, X, y)
