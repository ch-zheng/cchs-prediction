import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load data
X = torch.from_numpy(np.load('data/phase1/samples.npy'))
y = torch.from_numpy(np.load('data/phase1/labels.npy'))

print(X)
