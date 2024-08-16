import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from utilities.data_generation import RegressionData


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
    def __init__(self, input_dim: int, output_dim: int,
                 parent=None,
                 n_batches: int = 1,
                 bias: bool = True):
        super().__init__()
        self.in_features = input_dim
        self.out_features = output_dim

        # Define variational parameters for weight and bias
        self.weight_mu = nn.Parameter(torch.randn(input_dim, output_dim))
        self.weight_rho = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias_mu = nn.Parameter(torch.randn(output_dim))
        self.bias_rho = nn.Parameter(torch.randn(output_dim))
        self.include_bias = bias
        self.n_batches = n_batches
        self.parent = parent

    def reparameterize(self, mu, p):
        sigma = torch.log(1 + torch.exp(p))
        eps = torch.randn_like(sigma)
        return mu + (eps * sigma) + 1e-6

    def kl_divergence(self, z, mu_theta, p_theta, prior_sd=1):
        log_prior = Normal(0, prior_sd).log_prob(z)
        log_p_q = Normal(mu_theta, torch.log(1 + torch.exp(p_theta))).log_prob(z)
        return (log_p_q - log_prior).sum()

    def forward(self, x):
        w = self.reparameterize(self.weight_mu, self.weight_rho)

        if self.include_bias:
            b = self.reparameterize(self.bias_mu, self.bias_rho)
        else:
            b = 0

        z = x @ w + b

        self.parent.accumulated_kl_div += self.kl_divergence(w,
                                                             self.weight_mu,
                                                             self.weight_rho,
                                                             )
        if self.include_bias:
            self.parent.accumulated_kl_div += self.kl_divergence(b,
                                                                 self.bias_mu,
                                                                 self.bias_rho,
                                                                 )
        return z


@dataclass
class KL:
    accumulated_kl_div = 0


class BayesianNet(BayesianLinear):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 reg_fct: callable, n_samples: int = 1000, test_n_samples: int = 1000,
                 n_hidden: int = 4, kl_weight=0.7, batch_size=32,
                 wandb_active: bool = False):

        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.test_mse = None
        self.standard_deviation_predictions = None
        self.variance_predictions = None
        self.mean_predictions = None
        self.n_batches = int(n_samples / batch_size)
        self.batch_size = batch_size
        self.kl_loss = KL
        self.fc1 = BayesianLinear(input_dim, hidden_dim, n_batches=self.n_batches, parent=self.kl_loss)
        self.hidden_layers = {
            f"fc{i + 2}": BayesianLinear(hidden_dim, hidden_dim, n_batches=self.n_batches, parent=self.kl_loss) for i in
            range(n_hidden)
        }
        self.fc_final = BayesianLinear(hidden_dim, output_dim, n_batches=self.n_batches, parent=self.kl_loss)
        self.kl_weight = kl_weight
        self.data_set = RegressionData(reg_fct, n_samples=n_samples, test_n_samples=test_n_samples)
        self.wandb_active = wandb_active

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        for layer in self.hidden_layers.values():
            x = torch.relu(layer(x))
        x = self.fc_final(x)
        return x

    @property
    def accumulated_kl_div(self):
        return self.kl_loss.accumulated_kl_div

    def reset_kl_div(self):
        self.kl_loss.accumulated_kl_div = 0

    def train_model(self, n_epochs: int = 100, lr: float = 0.01, batch_size: int = 32 ,**kwargs):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        data_loader = DataLoader(self.data_set.train_data, batch_size=batch_size, shuffle=True)
        lost_all_epochs = []
        lost_log = []
        lost_kl = []
        for epoch in range(n_epochs):
            loss_this_epoch = []
            loss_log_this_epoch = []
            loss_kl_this_epoch = []

            for x, y in data_loader:
                x = x.float()
                y = y.float()
                optimizer.zero_grad()
                predictions = self(x)
                loss_kl, loss_log = det_loss(predictions, y, self, kl_weight=self.kl_weight)
                loss = loss_kl + loss_log
                loss.backward()
                loss_kl_this_epoch.append(loss_kl.item())
                loss_log_this_epoch.append(loss_log.item())
                loss_this_epoch.append(loss.item())
                optimizer.step()
            lost_kl.append(np.mean(loss_kl_this_epoch))
            lost_log.append(np.mean(loss_log_this_epoch))
            lost_all_epochs.append(np.mean(loss_this_epoch))
            # wandb.log({"kl_loss": np.mean(lost_kl_this_epoch), "log_loss": np.mean(lost_log_this_epoch),
            #            "loss": np.mean(lost_all_epochs)})
        self.model_loss = (lost_kl, lost_log, lost_all_epochs)


    def get_mse(self, test_vals, predictions):
        return np.mean((test_vals.squeeze() - predictions.squeeze()).detach().numpy() ** 2)

    def make_predictions_on_test(self):
        x = self.data_set.test_data.x.unsqueeze(1).unsqueeze(1)
        with torch.no_grad():
            trace = np.array([self(x).flatten().numpy() for _ in range(1000)]).T
        self.mean_predictions = trace.mean(axis=1)
        self.variance_predictions = trace.var(axis=1)
        self.standard_deviation_predictions = np.sqrt(self.variance_predictions)
        y_values_numpy = self.data_set.test_data.y.squeeze().numpy()
        self.test_mse = np.mean((y_values_numpy - self.mean_predictions) ** 2)


def det_loss(y, y_pred, model: BayesianNet, kl_weight=0.01):
    batch_size = y.shape[0]
    reconstruction_error = -Normal(y_pred, .1).log_prob(y).sum()
    kl = model.accumulated_kl_div
    model.reset_kl_div()
    return reconstruction_error*(1-kl_weight), kl*kl_weight
