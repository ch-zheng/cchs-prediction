import utils
import encode
import numpy as np

# Create dataset of landmarks with points pruned
# Row format: (subject, race, age, landmarks...)
X = np.load('data/individualized/samples.npy')
y = np.load('data/individualized/labels.npy')
X, y = utils.augment(X, y)

# Points to use
targets = {
    0,
    28,
    29,
    30,
    31,
    35,
    38,
    43,
    49,
    53,
    38,
    16,
    0,
    21,
    22,
    64,
    57,
    27,
    25,
    18,
    40,
    47,
}
'''
    55,
    59,
    5,
    11,
    60,
    56,
    58,
    32,
    34
}
'''

X = utils.filter_landmarks(X, targets)

# Save
encode.save_encodings('data/pruned', X, y)
