# Native
import math
from typing import Tuple, List
# External
import numpy as np

# Description: Data transformation utilities

# Oversample to make samples roughly proportionate to labels
def oversample(samples: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Invariant: Assumes positive label proportion <= 0.5
    multiplier = math.floor(len(labels) / sum(labels) - 1)
    additional = int((multiplier - 1) * sum(labels))
    # Uber := Extra samples
    uber_samples = np.empty((additional, samples.shape[1]), dtype=samples.dtype)
    uber_labels = np.empty(additional, dtype=labels.dtype)
    j = 0 # Uber index
    for i in range(len(samples)):
        if labels[i] == 1:
            for k in range(multiplier - 1):
                uber_samples[j+k] = samples[i]
                uber_labels[j+k] = labels[i]
            j += multiplier - 1
    uber_samples = np.row_stack((samples, uber_samples))
    uber_labels = np.hstack((labels, uber_labels))
    return uber_samples, uber_labels

# Augment facial landmark data
# Assumed row format: (subject, group, race, age, landmarks...)
def augment(samples: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Horizontally flip landmarks
    flipped_samples = samples.copy()
    for row in flipped_samples:
        x_coords = row[4::2]
        max_x = x_coords.max()
        x_coords *= -1
        x_coords += max_x
    # Append to original data
    result_samples = np.row_stack((samples, flipped_samples))
    result_labels = np.tile(labels, 2)
    return result_samples, result_labels

# Note: How to remove 'subject' column from samples array
# np.delete(samples, 0, 1)

def split(k: int, samples: np.ndarray, labels: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Count duplicates of each CCHS subject
    pos_subjects = set() # CCHS
    neg_subjects = set() # Control
    subject_counts = {}
    for i in range(len(samples)):
        subject = samples[i][0]
        group = labels[i]
        # Add to appropriate set
        if group == 1:
            pos_subjects.add(subject)
        else:
            neg_subjects.add(subject)
        # Increment count
        if subject in subject_counts:
            subject_counts[subject] += 1
        else:
            subject_counts[subject] = 1

    # Partition subjects into splits
    splits = []
    split_sizes = []
    for _ in range(k):
        splits.append(set())
        split_sizes.append(0)
    for subjects in (pos_subjects, neg_subjects):
        while len(subjects) > 0:
            # Find smallest split
            smallest_split = 0
            for i in range(len(splits)):
                if split_sizes[i] == min(split_sizes):
                    smallest_split = i
                    break
            # Insert subject into smallest split
            s = subjects.pop()
            splits[smallest_split].add(s)
            split_sizes[smallest_split] += subject_counts[s]

    # Assemble resulting arrays
    result = []
    for i in range(k):
        size = split_sizes[i]
        a = np.empty((size, samples.shape[1]), dtype=samples.dtype)
        b = np.empty(size, dtype=labels.dtype)
        count = 0
        split = splits[i]
        for j in range(len(samples)):
            if samples[j][0] in split:
                a[count] = samples[j]
                b[count] = labels[j]
                count += 1
        if count < size:
            raise Exception('Miscount') # Debugging test
        result.append((a, b))
    return result
