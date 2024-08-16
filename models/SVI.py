from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import torch
import torch.nn as nn
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from torch.utils.data import DataLoader
from tqdm.auto import trange
from pyro.infer import Predictive
import numpy as np

from utilities.data_generation import RegressionData


class PyroBNN(PyroModule):
    def __init__(self, input_dim: int, output_dim: int, n_hidden: int,
                 hidden_dim: int, prior_scale: float = 1.):
        super().__init__()
        self.activation = nn.ReLU()

        self.layer_sizes = [input_dim] + [hidden_dim] * n_hidden + [output_dim]

        layer_list = [PyroModule[nn.Linear](self.layer_sizes[i], self.layer_sizes[i + 1]) for i in
                      range(len(self.layer_sizes) - 1)]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        for i, layer in enumerate(self.layers):
            layer.weight = PyroSample(dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[i])).expand(
                [self.layer_sizes[i + 1], self.layer_sizes[i]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., prior_scale).expand([self.layer_sizes[i + 1]]).to_event(1))

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.activation(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))
        pred = self.layers[-1](x)
        mu = pred[:, 0]
        sigma = torch.exp(pred[:, 1]).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.reshape(-1, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        pred = self.fc3(x)
        mu = pred[:, 0]
        sigma = torch.exp(pred[:, 1])
        return mu, sigma


class VariationalInference(PyroBNN):
    def __init__(self, input_dim: int, output_dim: int, n_hidden: int,
                 hidden_dim: int, reg_fct: callable, prior_scale: float = 1.,
                 n_samples: int = 1000, test_n_samples: int = 1000,
                 wandb_active: bool = False,
                 n_samples_predictions: int = 500,
                 test_interval: tuple = (-5, 5),
                 train_interval: tuple = (-3, 3),
                 heteroscedastic: bool = False):
        super().__init__(input_dim, output_dim, n_hidden, hidden_dim, prior_scale)
        self.epistemic_uncertainty = None
        self.aleatoric_uncertainty = None
        self.standard_deviation = None
        self.mean_predictions = None
        self.guide = AutoDiagonalNormal(self)
        self.data_set = RegressionData(reg_fct, n_samples=n_samples, test_n_samples=test_n_samples,
                                       train_interval=train_interval, test_interval=test_interval,
                                       heteroscedastic=heteroscedastic)
        self.wandb_active = wandb_active
        self.n_samples_predictions = n_samples_predictions

    def train_model(self, n_epochs: int = 100, batch_size: int = 32, lr: float = 0.01, loss_fct=Trace_ELBO()):
        data_loader = DataLoader(self.data_set.train_data, batch_size=batch_size, shuffle=True)
        optimizer = pyro.optim.Adam({"lr": lr})
        svi = SVI(self, self.guide, optimizer, loss=Trace_ELBO())
        pyro.clear_param_store()
        progress_bar = trange(n_epochs)
        for epoch in progress_bar:
            # for x, y in data_loader:
            #     loss = svi.step(x.float().squeeze(), y.float().squeeze())
            loss = svi.step(self.data_set.train_data.x, self.data_set.train_data.y)
            progress_bar.set_postfix(loss=f"{loss:.3f}")

    def make_predictions_on_test(self):
        predictive = Predictive(self, guide=self.guide, num_samples=self.n_samples_predictions)
        preds = predictive(self.data_set.test_data.x)
        # self.mean_predictions = preds['obs'].T.detach().numpy().mean(axis=1)
        # self.standard_deviation = preds['obs'].T.detach().numpy().std(axis=1)

        all_predictions = []
        all_predictions_var = []
        for i in range(self.n_samples_predictions):
            simple_nn = SimpleNN(1, 10, 2)
            with torch.no_grad():
                simple_nn.fc1.weight.copy_(preds['layers.0.weight'][i, :, :, :].squeeze().view(10, 1))
                simple_nn.fc1.bias.copy_(preds['layers.0.bias'][i, :, :].squeeze())
                simple_nn.fc2.weight.copy_(preds['layers.1.weight'][i, :, :, :].squeeze().view(10, 10))
                simple_nn.fc2.bias.copy_(preds['layers.1.bias'][i, :, :].squeeze())
                simple_nn.fc3.weight.copy_(preds['layers.2.weight'][i, :, :, :].squeeze().view(2, 10))
                simple_nn.fc3.bias.copy_(preds['layers.2.bias'][i, :, :].squeeze())

            mean_pred, var_pred = simple_nn(self.data_set.test_data.x)
            all_predictions.append(mean_pred)
            all_predictions_var.append(var_pred)

        all_predictions = torch.stack(all_predictions)
        all_predictions_var = torch.stack(all_predictions_var)
        self.mean_predictions = all_predictions.detach().numpy().mean(axis=0)
        self.aleatoric_uncertainty = all_predictions_var.detach().numpy().mean(axis=0)
        self.epistemic_uncertainty = all_predictions.var(axis=0).detach().numpy()

#https://arxiv.org/pdf/1806.05978