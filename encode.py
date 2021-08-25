# Native
import csv
import random
# External
import numpy as np

# Description: Encode CSV table into NumPy array

# Training set
training_samples = [] # Row format: [race, age, landmarks...]
training_labels = []
# Test set
test_samples = []
test_labels = []

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
    count = 0
    for row in rows:
        # Get data
        label = label_encoding[row[0]]
        race = race_encoding[row[1]]
        age = age_encoding[row[2]]
        landmarks = np.array(tuple(map(int, row[3:])), dtype=np.single)
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
        if random.random() < 0.05 and len(test_samples) < 100:
            test_samples.append(sample)
            test_labels.append(label)
        else:
            training_samples.append(sample)
            training_labels.append(label)
        count = count + 1
print('Finished reading', count, 'table entries')
print('Training set size:', len(training_samples))
print('Test set size:', len(test_samples))

# Save as .npy files
# Training set
training_samples = np.stack(training_samples)
np.save('data/training/samples.npy', training_samples)
training_labels = np.array(training_labels, dtype=np.single)
np.save('data/training/labels.npy', training_labels)
# Test set
test_samples = np.stack(test_samples)
np.save('data/test/samples.npy', test_samples)
test_labels = np.array(test_labels, dtype=np.single)
np.save('data/test/labels.npy', test_labels)
print('Arrays saved')
