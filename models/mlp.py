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
training = True # Training or evaluation

# Define model
model = nn.Sequential(
    nn.Linear(138, 100),
    nn.ReLU(),
    nn.Linear(100, 60),
    nn.ReLU(),
    nn.Linear(60, 1),
    nn.Sigmoid(),
)
model.cuda()
loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epoch = 0

# Load model
model_path = Path('models/mlp.pt')
if model_path.is_file():
    print('Loading model')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

# Load data
X = torch.from_numpy(np.load('data/samples.npy'))
y = torch.from_numpy(np.load('data/labels.npy'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
if training:
    samples = X_train
    labels = y_test
else:
    samples = X_test
    labels = y_test
data = TensorDataset(samples.cuda(), labels.cuda())
dataloader = DataLoader(
    data,
    batch_size=32,
    shuffle=True
)

# Register exit callback
def sigint_handler(sig_num, frame):
    # Save model
    print('\nSaving model')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, model_path)
    sys.exit(0)

# Training loop
def train():
    global epoch
    model.train()
    signal.signal(signal.SIGINT, sigint_handler)
    last_print_time = time.time() # Limit console printing rate
    total_batches = math.floor(len(dataloader.dataset) / dataloader.batch_size)
    while True:
        for batch, (X, y) in enumerate(dataloader):
            prediction = model(X).squeeze() # Prediction
            loss = loss_func(prediction, y) # Compute loss
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print information
            if time.time() - last_print_time > 0.1:
                print(f'Epoch={epoch}, Batch={batch}/{total_batches}, Loss={loss.item()}', end='\r')
                last_print_time = time.time()
        epoch += 1

# Evaluation
def evaluate():
    model.eval()
    total = len(dataloader.dataset)
    correct = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            prediction = model(X).squeeze()
            comparisons = y.greater(0.5) == prediction.greater(0.5)
            # Get incorrect predictions
            for i, c in enumerate(comparisons):
                if c.item() == False:
                    # Print initial 6 entries of offending sample
                    print(32 * batch + i, X[i][:6], y[i])
            correct += comparisons.sum().item()
    print(f'Accuracy={correct/total}')

# Run main loop
if training:
    train()
else:
    evaluate()
