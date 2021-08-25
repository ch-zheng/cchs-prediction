# Native
import csv
import random
# External
import numpy as np

# Description: Find CSV table entry based on the initial entries of an encoded row
# Target format: (race, age, landmarks...)
targets = (
    (-1, 0, 0, 0.3122),
    (0, 2, 0, 0.1364),
    (-1, 3, 0, 0.2260),
    (0, 3, 0, 0.2338),
    (-1, 1, 0, 0.2014),
    (-1, 1, 0, 0.1178),
    (-1, 0, 0, 0.3651),
    (-1, 1, 0, 0.1629),
    (-1, 1, 0, 0.2695),
    (-1, 0, 0, 0.2051)
)

# Encodings
label_encoding = {'control': 0, 'cchs': 1}
race_encoding = {'other': 0, 'unknown': 0, 'black': 1, 'white': -1}
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
