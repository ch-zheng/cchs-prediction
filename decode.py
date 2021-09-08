# Native
import csv
import random
# External
import numpy as np

# Description: Find CSV table entry based on the initial entries of an encoded row.
# Used for finding the corresponding image filename for misclassified samples.

# Target format: (race, age, landmarks...)
targets = (
    (-1, 7, 0, 2.6296e-01),
    (-1, 0, 0, 0.1130),
    (-1, 1, 0, 0.3194),
    (-1, 4, 0, 0.2147),
    (1, 0, 0, 0.2457),
    (-1, 1, 0, 0.1913),
    (0, 1, 0, 0.1226),
    (-1, 1, 0.0226, 0.0904)
)

# Encodings
label_encoding = {'CONTROL': 0, 'CCHS': 1}
race_encoding = {'Other': 0, 'UNKNOWN': 0, 'African American': 1, 'Caucasian': -1}
age_encoding = {
    '0-1': 0,
    '2-3': 1,
    '4-5': 2,
    '6-8': 3,
    '9-11': 4,
    '12-14': 5,
    '15-17': 6,
    '18-25': 7,
    '26-35': 8
}

# Read CSV file
print('Opening CSV file')
with open('data/samples.csv') as csv_file:
    rows = iter(csv.reader(csv_file))
    next(rows) # Skip header row
    for row_idx, row in enumerate(rows):
        # Get data
        label = label_encoding[row[1]]
        race = race_encoding[row[2]]
        age = age_encoding[row[3]]
        landmarks = np.array(tuple(map(int, row[4:])), dtype=np.single)
        # Standardize landmarks to [0, 1]
        landmarks_x = landmarks[::2]
        landmarks_x = landmarks_x - landmarks_x.min()
        landmarks[::2] = landmarks_x
        landmarks_y = landmarks[1::2]
        landmarks_y = landmarks_y - landmarks_y.min()
        landmarks[1::2] = landmarks_y
        landmarks = landmarks / landmarks.max()
        # Aggregate features
        sample = np.empty(2 + landmarks.size, dtype=np.single)
        sample[0] = race
        sample[1] = age
        sample[2:] = landmarks
        # Compare to targets
        for target in targets:
            match = True
            for i, x in enumerate(target):
                if abs(x - float(sample[i])) >= 0.0001:
                    match = False
                    break
            if match:
                print(row[0])
