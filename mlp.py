from model import Model
# Native
import math
import os
#from pathlib import Path
# External
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sklearn.metrics as metrics

# Description: Train a multilayer perceptron neural net.
training = True # Training or evaluation

class MLP(Model):
    def __init__(self, features: int):
        self.model = nn.Sequential(
            nn.Linear(features, 2*math.floor(features/3)),
            nn.ReLU(),
            nn.Linear(2*math.floor(features/3), math.floor(features/3)),
            nn.ReLU(),
            nn.Linear(math.floor(features/3), 1),
            nn.Sigmoid(),
        )
        self.model.cuda()
        self.loss_func = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
    def fit(self, X, y):
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        # Create dataloader
        data = TensorDataset(X.cuda(), y.cuda())
        dataloader = DataLoader(
            data,
            batch_size=64,
            shuffle=True
        )
        # Training loop
        self.model.train()
        for epoch in range(20000):
            print('Epoch', epoch, end='\r')
            for X, y in dataloader:
                prediction = self.model(X).squeeze() # Prediction
                loss = self.loss_func(prediction, y) # Compute loss
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print()
    def predict(self, X):
        self.model.eval()
        X = torch.from_numpy(X).cuda()
        with torch.no_grad():
            predictions = self.model(X).squeeze()
        predictions = predictions.cpu().numpy()
        return np.around(predictions)
    def score(self, X, y) -> dict[str, float]:
        predictions = self.predict(X)
        results = {
            'accuracy': metrics.accuracy_score(y, predictions),
            'balanced_accuracy': metrics.balanced_accuracy_score(y, predictions),
            'recall': metrics.recall_score(y, predictions),
            'precision': metrics.precision_score(y, predictions)
        }
        return results
    # Disk ops
    def save(self, file: os.PathLike):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, file)
    def load(self, file: os.PathLike):
        checkpoint = torch.load(file)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
