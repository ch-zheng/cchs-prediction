# Native
import math
from typing import Tuple, List
# External
import numpy as np

# Description: Data transformation utilities

# Pairs of landmarks to be horizontally swapped
_swaps = (
    # Jaw
    (0, 16),
    (1, 15),
    (2, 14),
    (3, 13),
    (4, 12),
    (5, 11),
    (6, 10),
    (7, 9),
    # Brow
    (17, 26),
    (18, 25),
    (19, 24),
    (20, 23),
    (21, 22),
    # Nostrils
    (31, 35),
    (32, 34),
    # Eyes
    (36, 45),
    (37, 44),
    (38, 43),
    (39, 42),
    (40, 47),
    (41, 46),
    # Mouth
    (48, 54),
    (49, 53),
    (50, 52),
    (55, 59),
    (56, 58),
    # Inner lips
    (61, 63),
    (65, 67),
)

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
def augment(samples: np.ndarray, labels: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
    # Horizontally flip landmarks
    flipped_samples = samples.copy()
    for row in flipped_samples:
        x_coords = row[offset::2]
        max_x = x_coords.max()
        x_coords *= -1
        x_coords += max_x
        # Swap landmark positions
        for swap in _swaps:
            a = offset + 2*swap[0]
            b = offset + 2*swap[1]
            tmp = row[a:a+2].copy()
            row[a:a+2] = row[b:b+2]
            row[b:b+2] = tmp.copy()
    # Append to original data
    result_samples = np.row_stack((samples, flipped_samples))
    result_labels = np.tile(labels, 2)
    return result_samples, result_labels

# Assumed row format: (subject, race, age, landmarks...)
def filter_landmarks(X, targets):
    samples = np.empty((X.shape[0], 3+2*len(targets)), dtype=X.dtype)
    for src, dst in zip(X, samples):
        dst[:3] = src[:3]
        k = 3
        for t in targets:
            i = 3 + 2*t
            dst[k:k+2] = src[i:i+2]
            k += 2
    return samples

# Note: How to remove 'subject' column from samples array
# np.delete(samples, 0, 1)

# Create splits in the format expected by sklearn cross-validation
# Assumed row format: (subject, ...)
def split(k: int, samples: np.ndarray, labels: np.ndarray) -> List[Tuple[List[int], List[int]]]:
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

    # Partition subjects into buckets
    buckets = []
    bucket_sizes = []
    for _ in range(k):
        buckets.append(set())
        bucket_sizes.append(0)
    for subjects in map(
        lambda x: sorted(x, key=lambda y: subject_counts[y], reverse=True),
        (pos_subjects, neg_subjects)):
        for s in subjects:
            # Find smallest bucket
            smallest_bucket = 0 # Index into buckets
            min_size = min(bucket_sizes)
            for i in range(k):
                if bucket_sizes[i] == min_size:
                    smallest_bucket = i
                    break
            # Insert subject into smallest bucket
            buckets[smallest_bucket].add(s)
            bucket_sizes[smallest_bucket] += subject_counts[s]

    # Assemble splits
    splits = []
    for bucket in buckets:
        training = []
        test = []
        for i in range(len(samples)):
            subject = samples[i][0]
            if subject in bucket:
                test.append(i)
            else:
                training.append(i)
        splits.append((training, test))
    return splits

# Create array splits from indices
def create_splits(X: np.ndarray, y: np.ndarray, splits: List[Tuple[List[int], List[int]]]) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    result = []
    for split in splits:
        data_sets = [] # [Training, Test]
        for i in range(2):
            set_size = len(split[i])
            data_set = (
                np.empty((set_size, X.shape[1]), dtype=X.dtype),
                np.empty(set_size, dtype=y.dtype)
            )
            idx = 0
            for row_idx in split[i]:
                data_set[0][idx] = X[row_idx]
                data_set[1][idx] = y[row_idx]
                idx += 1
            data_sets.append(data_set)
        result.append(tuple(data_sets))
    return result
