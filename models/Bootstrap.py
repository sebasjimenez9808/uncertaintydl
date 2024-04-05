import torch
import torch.nn as nn
import numpy as np
from utils.data_generation import BootstrapDataset, RegressionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class RegressionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, n_hidden: int = 1,
                 seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = {
            f"fc{i + 2}": nn.Linear(hidden_dim, hidden_dim) for i in range(n_hidden)
        }
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        for layer in self.hidden_layers.values():
            x = torch.relu(layer(x))
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))
        return mu, sigma



class BootstrapEnsemble(RegressionMLP):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 reg_fct: callable,
                 n_hidden: int = 4, n_models: int = 100,
                 bootstrap_size: float = 0.6):
        super().__init__(input_dim, hidden_dim, output_dim, n_hidden)
        self.n_models = n_models
        self.bootstrap_size = bootstrap_size
        self.models = [RegressionMLP(input_dim, hidden_dim, output_dim, n_hidden) for _ in range(n_models)]
        self.data_set = RegressionDataset(reg_fct, n_samples=1000)
        self.model_loss = None

    def train_model(self, loss_fct: callable, n_epochs: int = 100, lr: float = 0.01,
                    batch_size: int = 32):
        bootstrap_sample = BootstrapDataset(self.data_set.x_train,
                                            self.data_set.y_train,
                                            self.bootstrap_size, self.n_models)
        loss_array = {i: [] for i in range(self.n_models)}
        for i, model in enumerate(tqdm(self.models)):
            bootstrap_sample.index_bootstrap = i
            data_loader = DataLoader(bootstrap_sample, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_all_epochs = []
            for epoch in range(n_epochs):
                loss_this_epoch = []
                for x, y in data_loader:
                    x = x.float()
                    y = y.float()
                    optimizer.zero_grad()
                    predictions = model(x)
                    loss = loss_fct(predictions, y)
                    loss.backward()
                    optimizer.step()
                    loss_this_epoch.append(loss.item())
                loss_all_epochs.append(sum(loss_this_epoch) / len(loss_this_epoch))
            loss_array[i] = loss_all_epochs
        self.model_loss = loss_array

    def forward(self, x):
        predictions = []
        for model in self.models:
            with torch.no_grad():
                prediction = model(x)
            predictions.append(prediction.detach().numpy())
        return np.asarray(predictions)


