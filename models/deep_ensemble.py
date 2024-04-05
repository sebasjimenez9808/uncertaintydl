from tqdm import tqdm
from models.Bootstrap import RegressionMLP
from utils.data_generation import RegressionDataset, BootstrapDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import copy


class DeepEnsemble(RegressionMLP):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 reg_fct: callable,
                 n_hidden: int = 4, n_models: int = 100):
        super().__init__(input_dim, hidden_dim, output_dim, n_hidden, seed=42)
        self.n_models = n_models
        self.models = [RegressionMLP(input_dim, hidden_dim, output_dim,
                                     n_hidden, seed=_ + 42) for _ in range(n_models)]
        self.data_set = RegressionDataset(reg_fct, n_samples=1000)
        self.output_dim = output_dim
        self.model_loss = None

    def train_model(self, loss_fct: callable, n_epochs: int = 100, lr: float = 0.01,
                    batch_size: int = 32):
        torch.manual_seed(42)
        shuffle_index = torch.randperm(len(self.data_set.y_train))
        shuffle_data = copy.deepcopy(self.data_set)
        shuffle_data.x_train = shuffle_data.x_train[shuffle_index]
        shuffle_data.y_train = shuffle_data.y_train[shuffle_index]

        loss_array = {i: [] for i in range(self.n_models)}

        for i, model in enumerate(tqdm(self.models)):
            data_loader = DataLoader(shuffle_data, batch_size=batch_size, shuffle=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_all_epochs = []
            for epoch in range(n_epochs):
                loss_this_epoch = []
                for x, y in data_loader:
                    x = x.float()
                    y = y.float()
                    optimizer.zero_grad()
                    mu, sigma = model(x)
                    loss = loss_fct((mu, sigma), y)
                    loss.backward()
                    optimizer.step()
                    loss_this_epoch.append(loss.item())
                loss_all_epochs.append(sum(loss_this_epoch) / len(loss_this_epoch))
            loss_array[i] = loss_all_epochs
        self.model_loss = loss_array

    def forward(self, x):
        x_test = torch.from_numpy(x).float().view(-1, 1)
        predictions = torch.stack([torch.stack(list(m(x_test)), dim=0).view(-1, 2) for m in self.models], dim=0)
        mean = torch.mean(predictions[:, :, 0], dim=0)
        variance = torch.mean(predictions[:, :, 0]** 2 + predictions[:, :, 1] ** 2, dim=0) - mean ** 2
        return mean, variance
