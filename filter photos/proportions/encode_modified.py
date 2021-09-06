# Native
import csv
import random
# External
import numpy as np

# Description: Encode CSV table into NumPy array files

samples = []
labels = []

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
with open('data/proportions/samples_final.csv') as csv_file:
    rows = iter(csv.reader(csv_file))
    next(rows) # Skip header row
    count = 0
    for row in rows:
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
        # Append to result
        samples.append(sample)
        labels.append(label)
        count = count + 1
print('Finished reading', count, 'table entries')

# Save as .npy files
samples = np.stack(samples)
np.save('data/proportions/samples.npy', samples)
labels = np.array(labels, dtype=np.single)
np.save('data/proportions/labels.npy', labels)
print('Arrays saved')
