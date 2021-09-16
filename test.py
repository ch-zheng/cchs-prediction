import utils
import numpy as np
import matplotlib.pyplot as plt
#import zoo
#import json
#from pathlib import Path

# Load data
samples = np.load('data/individualized/samples.npy')
labels = np.load('data/individualized/labels.npy')
X, y = utils.augment(samples, labels)
X = np.delete(X, range(3), 1)

c = 1500
a = X[c]
b = X[c+2888]

plt.figure()
plt.axis([0, 1, 0, 1])
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_aspect(1)
plt.scatter(a[::2], a[1::2], c='#FF0000')
plt.scatter(b[::2], b[1::2], c='#00FF00')
plt.show()
