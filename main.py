from sys import exit
from utils import oversample
import numpy as np

# Load data
samples = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
labels = np.array([0, 1, 0, 0])
print('Samples', samples)
print('Labels', labels)
a, b = oversample(samples, labels)
print('a', a)
print('b', b)
exit(0)

add_count = 2*int(sum(labels)) # Additional row count
uber_samples = np.empty((add_count, samples.shape[1]))
uber_labels = np.empty(add_count)
k = 0
for i in range(len(samples)):
    if labels[i] == 1:
        uber_samples[k] = samples[i] 
        uber_labels[k] = labels[i]
        uber_samples[k+1] = samples[i]
        uber_labels[k+1] = labels[i]
        k += 2
uber_samples = np.row_stack((samples, uber_samples))
uber_labels = np.row_stack((labels, uber_labels))
print(uber_samples)
print(uber_labels)
