# Native
import csv
import os
from typing import Tuple
from pathlib import Path
# External
import numpy as np

# Description: Encode CSV table into NumPy array files

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

# Read CSV table & convert to NumPy arrays
def encode(input_table: str) -> Tuple[np.ndarray, np.ndarray]:
    samples = []
    labels = []
    # Read CSV file
    # Assumed row format: (filename, subject, group, race, age, landmarks...)
    with open(input_table) as csv_file:
        rows = iter(csv.reader(csv_file))
        next(rows) # Skip header row
        for row in rows:
            # Get data
            subject = row[1]
            label = label_encoding[row[2]]
            race = race_encoding[row[3]]
            age = age_encoding[row[4]]
            landmarks = []
            for i in range(68):
                landmarks.append(row[5+2*i]) # x
                landmarks.append(row[6+2*i]) # y
            landmarks = np.array(tuple(map(int, landmarks)), dtype=np.single)
            # Standardize landmarks to [0, 1]
            landmarks_x = landmarks[::2]
            landmarks_x -= landmarks_x.min()
            landmarks_y = landmarks[1::2]
            landmarks_y -= landmarks_y.min()
            landmarks /= landmarks.max()
            # Aggregate features
            sample = np.empty(3 + landmarks.size, dtype=np.single)
            sample[0] = subject
            sample[1] = race
            sample[2] = age
            sample[3:] = landmarks
            # Append to result
            samples.append(sample)
            labels.append(label)
    samples = np.stack(samples)
    labels = np.array(labels, dtype=np.single)
    return samples, labels

# Save encodings as .npy files
def save_encodings(output_dir: str, samples: np.ndarray, labels: np.ndarray):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(Path(output_dir, 'samples.npy'), samples)
    np.save(Path(output_dir, 'labels.npy'), labels)
