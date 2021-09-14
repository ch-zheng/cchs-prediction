import utils
#import encode
import numpy as np

# Encode table
#encodings = encode.encode('data/samples_updated.csv')
#encode.save_encodings('data/individualized', encodings[0], encodings[1])

samples = np.load('data/individualized/samples.npy')
labels = np.load('data/individualized/labels.npy')
samples, labels = utils.augment(samples, labels)
splits = utils.split(10, samples, labels)
