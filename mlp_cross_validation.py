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
X = torch.from_numpy(np.load('data/samples.npy'))
y = torch.from_numpy(np.load('data/labels.npy'))

# Create csv file
OUTPUT_FILE = "data/accuracy_results.csv"
with open(OUTPUT_FILE, "a", newline='') as csvfile:
    csvWriter = csv.writer(csvfile)
    header = ['trial', 'accuracy']
    csvWriter.writerow(header)

def load_data():
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    # Create dataloaders
    data_train = TensorDataset(X_train, y_train)
    dataloader_train = DataLoader(
        data_train,
        batch_size=32,
        shuffle=True
    )
    data_test = TensorDataset(X_test, y_test)
    dataloader_test = DataLoader(
        data_test,
        batch_size=32,
        shuffle=True
    )

    return dataloader_train, dataloader_test

def create_model():
    # Define model
    model = nn.Sequential(
        nn.Linear(138, 100),
        nn.ReLU(),
        nn.Linear(100, 60),
        nn.ReLU(),
        nn.Linear(60, 1),
        nn.Sigmoid(),
    )
    model
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    return model, loss_func, optimizer

# Training loop
def train(model, loss_func, optimizer, dataloader):
    epoch = 0
    model.train()
    last_print_time = time.time() # Limit console printing rate
    total_batches = math.floor(len(dataloader.dataset) / dataloader.batch_size)
    while epoch <= 60000:
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
    
    return model

# Evaluation
def evaluate(model, dataloader):
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
    accuracy = correct/total
    print(f'Accuracy for run {trial}={accuracy}')
    with open(OUTPUT_FILE, "a", newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerow([trial, accuracy])

# Run main loop
k = 10 # 10-fold cross validation
trial = 0
while trial < k:
    dataloader_train, dataloader_test = load_data()
    model, loss_func, optimizer = create_model()
    model = train(model, loss_func, optimizer, dataloader_train)
    evaluate(model, dataloader_test)
    trial += 1
