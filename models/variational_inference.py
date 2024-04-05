import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.data_generation import RegressionDataset


def kl_loss_function(model, predictions, y, kl_weight=0.7):
    # Compute negative log likelihood (NLL) loss
    nll_loss = nn.functional.mse_loss(predictions, y)

    # Compute KL divergence between variational posterior and prior
    kl_loss = 0
    for layer in model.children():
        kl_loss += torch.distributions.kl.kl_divergence(
            Normal(layer.weight_mu, torch.log(1 + torch.exp(layer.weight_rho))),
            Normal(torch.zeros_like(layer.weight_mu), torch.ones_like(layer.weight_rho))
        ).sum()

        kl_loss += torch.distributions.kl.kl_divergence(
            Normal(layer.bias_mu, torch.log(1 + torch.exp(layer.bias_rho))),
            Normal(torch.zeros_like(layer.bias_mu), torch.ones_like(layer.bias_rho))
        ).sum()

    # Combine NLL and KL losses
    loss = (1 - kl_weight) * nll_loss + kl_weight * kl_loss

    return loss


class BayesianLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.in_features = input_dim
        self.out_features = output_dim

        # Define variational parameters for weight and bias
        self.weight_mu = nn.Parameter(torch.randn(output_dim, input_dim))
        self.weight_rho = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias_mu = nn.Parameter(torch.randn(output_dim))
        self.bias_rho = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        # Sample weight and bias from their variational distributions
        weight = Normal(self.weight_mu, torch.log(1 + torch.exp(self.weight_rho))).rsample()
        bias = Normal(self.bias_mu, torch.log(1 + torch.exp(self.bias_rho))).rsample()

        return torch.nn.functional.linear(x, weight, bias)


class BayesianNet(BayesianLinear):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 reg_fct: callable,
                 n_hidden: int = 4, kl_weight=0.7):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.hidden_layers = {
            f"fc{i + 2}": BayesianLinear(hidden_dim, hidden_dim) for i in range(n_hidden)
        }
        self.fc_final = BayesianLinear(hidden_dim, output_dim)
        self.kl_weight = kl_weight
        self.data_set = RegressionDataset(reg_fct, n_samples=1000)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        for layer in self.hidden_layers.values():
            x = torch.relu(layer(x))
        x = self.fc_final(x)
        return x

    def train_model(self, n_epochs: int = 100, lr: float = 0.01,
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
                predictions = self(x)
                loss = kl_loss_function(self, predictions, y,
                                        kl_weight=self.kl_weight)
                loss.backward()
                loss_this_epoch.append(loss.item())
                optimizer.step()
            lost_all_epochs.append(np.mean(loss_this_epoch))
        self.model_loss = lost_all_epochs




