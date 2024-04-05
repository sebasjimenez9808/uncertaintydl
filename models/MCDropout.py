import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from models.Bootstrap import RegressionMLP
from utils.data_generation import RegressionDataset


class MCDropoutNet(RegressionMLP):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 reg_fct: callable, n_hidden: int = 4, dropout_p=0.3):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim,
                         output_dim=output_dim, n_hidden=n_hidden)
        self.dropout = nn.Dropout(p=dropout_p)  # Apply dropout to all layers
        self.data_set = RegressionDataset(reg_fct, n_samples=1000)
        self.model_loss = None

    def forward(self, x, i=1):
        torch.manual_seed(i + 42)
        x = torch.relu(self.fc1(x))
        for layer in self.hidden_layers.values():
            x = torch.relu(self.dropout(layer(x)))
        x = self.fc_final(x)
        return x

    def train_model(self, loss_fct: callable, n_epochs: int = 100, lr: float = 0.01,
                      batch_size: int = 32):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        data_loader = DataLoader(self.data_set, batch_size=batch_size, shuffle=True)
        lost_all_epochs = []
        for epoch in range(n_epochs):
            loss_this_epoch = []
            for x, y in data_loader:
                x = x.float()
                y = y.float()
                optimizer.zero_grad()
                predictions = self(x, epoch + 100)
                loss = loss_fct(predictions, y)
                loss.backward()
                loss_this_epoch.append(loss.item())
                optimizer.step()
            lost_all_epochs.append(np.mean(loss_this_epoch))
        self.model_loss = lost_all_epochs
