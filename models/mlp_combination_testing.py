# Native
import csv
import math
import signal
import sys
import time
from pathlib import Path
# External
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Description: Train a multilayer perceptron neural net.

# Load data
X = torch.from_numpy(np.load('data/phase1/samples.npy'))
y = torch.from_numpy(np.load('data/phase1/labels.npy'))

points_list = {

}


