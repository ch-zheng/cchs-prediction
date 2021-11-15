import utils
import numpy as np
from sklearn.model_selection import cross_val_predict
from pathlib import Path
from zoo import Ridge

# Load model
model = Ridge()
model.load(Path('pretrained/ridge.pickle'))

# Load data
X = np.load('data/pruned/samples.npy')
y = np.load('data/pruned/labels.npy')
#X, y = utils.augment(X, y, 3)
splits = utils.split(10, X, y)
X = np.delete(X, 0, 1)

predictions = cross_val_predict(model.model, X, y, cv=splits, n_jobs=-1)
i = 0
print('ROW/TRUE/FALSE')
for a, b in zip(y, predictions):
    if a != b:
        print(i, a, b)
    i += 1
