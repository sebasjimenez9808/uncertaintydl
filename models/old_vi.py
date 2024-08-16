from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import time
import torch.distributions as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from utilities.data_generation import RegressionData
import torch.utils.data as data_utils


class LinearVariational(nn.Module):
    """
    Mean field approximation of nn.Linear
    """

    def __init__(self, in_features, out_features, parent, n_batches, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_bias = bias
        self.parent = parent
        self.n_batches = n_batches

        if getattr(parent, 'accumulated_kl_div', None) is None:
            parent.accumulated_kl_div = 0

        # Initialize the variational parameters.
        # 𝑄(𝑤)=N(𝜇_𝜃,𝜎2_𝜃)
        # Do some random initialization with 𝜎=0.001
        self.w_mu = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(mean=0, std=0.001)
        )
        # proxy for variance
        # log(1 + exp(ρ))◦ eps
        self.w_p = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(mean=-2.5, std=0.001)
        )
        if self.include_bias:
            self.b_mu = nn.Parameter(
                torch.zeros(out_features)
            )
            # proxy for variance
            self.b_p = nn.Parameter(
                torch.zeros(out_features)
            )

    def reparameterize(self, mu, p):
        torch.manual_seed(int(time.time() * 1000))
        sigma = torch.log(1 + torch.exp(p))
        eps = torch.randn_like(sigma)
        return mu + (eps * sigma)

    def kl_divergence(self, z, mu_theta, p_theta, prior_sd=1):
        log_prior = dist.Normal(0, prior_sd).log_prob(z)
        log_p_q = dist.Normal(mu_theta, torch.log(1 + torch.exp(p_theta))).log_prob(z)
        return (log_p_q - log_prior).mean() / self.n_batches

    def forward(self, x):
        w = self.reparameterize(self.w_mu, self.w_p)

        if self.include_bias:
            b = self.reparameterize(self.b_mu, self.b_p)
        else:
            b = 0

        z = x @ w + b

        self.parent.accumulated_kl_div += self.kl_divergence(w,
                                                             self.w_mu,
                                                             self.w_p,
                                                             )
        if self.include_bias:
            self.parent.accumulated_kl_div += self.kl_divergence(b,
                                                                 self.b_mu,
                                                                 self.b_p,
                                                                 )
        return z


@dataclass
class KL:
    accumulated_kl_div = 0


class VIModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_hidden: int,
                 hidden_dim: int, reg_fct: callable, prior_scale: float = 1.,
                 n_samples: int = 1000, test_n_samples: int = 1000,
                 wandb_active: bool = False,
                 n_samples_predictions: int = 500,
                 test_interval: tuple = (-5, 5),
                 train_interval: tuple = (-3, 3),
                 heteroscedastic: bool = False,
                 problem: str = 'regression',
                 seed: int = 42, add_sigmoid: bool = False):
        super().__init__()
        torch.manual_seed(seed)
        self.epistemic_uncertainty = None
        self.aleatoric_uncertainty = None
        self.mean_predictions = None
        self.kl_loss = KL
        self.n_batches = int(n_samples / 32)
        sequential_layers = [LinearVariational(input_dim, hidden_dim, self.kl_loss, self.n_batches),
                             nn.ReLU()]
        for _ in range(n_hidden):
            sequential_layers.extend([LinearVariational(hidden_dim, hidden_dim, self.kl_loss, self.n_batches),
                                      nn.ReLU()])
        sequential_layers.append(LinearVariational(hidden_dim, output_dim, self.kl_loss, self.n_batches))
        if add_sigmoid:
            sequential_layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(
            *sequential_layers
        )
        self.data_set = RegressionData(reg_fct, n_samples=n_samples, test_n_samples=test_n_samples,
                                       train_interval=train_interval, test_interval=test_interval,
                                       heteroscedastic=heteroscedastic,
                                       problem=problem)
        self.n_samples_predictions = n_samples_predictions
        self.training_time = None

    @property
    def accumulated_kl_div(self):
        return self.kl_loss.accumulated_kl_div

    def reset_kl_div(self):
        self.kl_loss.accumulated_kl_div = 0

    def forward(self, x):
        return self.layers(x)

    def loss_function(self, y_pred, y, loss_fct: callable):
        kl = self.accumulated_kl_div
        self.reset_kl_div()
        return loss_fct(predictions=y_pred, y=y, return_vector=True).sum() + kl

    def train_model(self, loss_fct: callable, n_epochs: int = 100, batch_size: int = 32, lr: float = 0.01):
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.data_loader = DataLoader(data_utils.TensorDataset(self.data_set.train_data.x.unsqueeze(-1),
                                                               self.data_set.train_data.y.unsqueeze(-1)),
                                      batch_size=batch_size)

        for epoch in tqdm(range(n_epochs)):
            for x, y in self.data_loader:
                optim.zero_grad()
                y_pred = self(x)
                loss = self.loss_function(y_pred=y_pred, y=y, loss_fct=loss_fct)
                loss.backward()
                optim.step()
            # optim.zero_grad()
            # y_pred = self(self.data_set.train_data.x.unsqueeze(1))
            # loss = self.loss_function(y_pred=y_pred, y=self.data_set.train_data.y.unsqueeze(1),
            #                           loss_fct=loss_fct)
            # loss.backward()
            # optim.step()

    def make_predictions_on_test(self):
        mean_predictions = []
        log_vars_predictions = []
        for i in range(self.n_samples_predictions):
            with torch.no_grad():
                pred = self(self.data_set.test_data.x.unsqueeze(1))
                mean_predictions.append(pred[:, 0])
                log_vars_predictions.append(pred[:, 1])

        mean_predictions = torch.stack(mean_predictions)
        log_vars_predictions = torch.stack(log_vars_predictions)

        self.mean_predictions = mean_predictions.detach().numpy().mean(axis=0)
        self.aleatoric_uncertainty = torch.exp(log_vars_predictions).detach().numpy().mean(axis=0)
        self.epistemic_uncertainty = mean_predictions.detach().numpy().var(axis=0)

    def make_predictions_on_test_classification(self):
        predictions = []
        for i in range(self.n_samples_predictions):
            with torch.no_grad():
                pred = self(self.data_set.test_data.x.unsqueeze(1))
                predictions.append(pred)
        accuracy = self.get_accuracy(self.data_set.test_data.y, predictions)
        print(f"Accuracy: {accuracy}")
        return accuracy

    def make_predictions_on_test_classification_information(self):
        predictions = []
        for i in range(self.n_samples_predictions):
            with torch.no_grad():
                pred = self(self.data_set.test_data.x.unsqueeze(1))
                pred = torch.sigmoid(pred)
                predictions.append(pred)
        self.get_information_theoretical_decomposition(predictions)

    def get_accuracy(self, y_values, predictions, stacked: bool = False):
        if not stacked:
            predictions = torch.stack(predictions, dim=0)
        # prediction has dimensions (n_models, n_samples, n_classes * 2)
        num_classes = int(predictions.size(2) / 2)  # there are two outputs per class: mean and variance

        mean_predictions = np.array(predictions[:, :, : num_classes])
        var_predictions = np.array(predictions[:, :, num_classes:])
        means = np.mean(mean_predictions, axis=0)
        vars = np.mean(var_predictions, axis=0)

        mean_tensor = torch.tensor(means, dtype=torch.float32)
        log_var_tensor = torch.tensor(vars, dtype=torch.float32)

        # Number of samples to draw
        num_samples_to_draw = 1000
        n_samples = mean_predictions.shape[1]

        # Step 1: Sample from the normal distribution
        # Here we need to sample for each mean and variance
        epsilon = torch.randn(num_samples_to_draw, n_samples,
                              2)  # Shape (num_samples_to_draw, n_samples_from_data, categories)
        std_tensor = torch.exp(log_var_tensor / 2)  # Standard deviation tensor (n_samples, categories)
        var_tensor = torch.exp(log_var_tensor)  # Variance tensor (n_samples, categories)

        # Broadcast mean and std to match epsilon shape for element-wise operation
        mean_broadcasted = mean_tensor.unsqueeze(0).expand(num_samples_to_draw, -1, -1)
        std_broadcasted = std_tensor.unsqueeze(0).expand(num_samples_to_draw, -1, -1)

        # Sample from the normal distribution
        sampled_tensor = mean_broadcasted + std_broadcasted * epsilon  # Shape (num_samples_to_draw, n_samples, categories)

        # Step 2: Apply softmax to obtain probabilities
        # Apply softmax along the last dimension
        probabilities_tensor = F.softmax(sampled_tensor, dim=-1)

        probabilities_mean = torch.mean(probabilities_tensor, dim=0)
        variance_mean = torch.var(probabilities_tensor, dim=0)

        self.mean_predictions = probabilities_mean[:, 1]
        self.epistemic_uncertainty = variance_mean[:, 1]
        self.aleatoric_uncertainty = var_tensor[:, 1]

        # use probabilities_mean to calculate entropy
        entropy = -torch.sum(probabilities_mean * torch.log(probabilities_mean), dim=-1)
        self.aleatoric_entropy = entropy

        # make a tensor of dim (500, 1) with the index of the greatest value for each sample
        predictions_cat = torch.argmax(probabilities_mean, dim=-1)

        true_labels = y_values.squeeze().numpy()
        predictions_labels = predictions_cat.numpy()

        accuracy = np.mean(true_labels == predictions_labels)
        return accuracy

    def get_information_theoretical_decomposition(self, predictions, stacked: bool = False):
        if not stacked:
            predictions = torch.stack(predictions, dim=0)
        prediction_class_zero = (1 - predictions)

        self.aleatoric_entropy = -torch.mean(predictions * torch.log(predictions) +
                                             prediction_class_zero * torch.log(prediction_class_zero), dim=0)
        predictions_mean = torch.mean(predictions, dim=0)
        predictions_mean_class_zero = torch.mean(prediction_class_zero, dim=0)

        self.total_entropy = -(predictions_mean * torch.log(predictions_mean) +
                               predictions_mean_class_zero * torch.log(predictions_mean_class_zero))

        self.epistemic_entropy = self.total_entropy - self.aleatoric_entropy

        self.predictions = predictions
        self.mean_predictions = predictions_mean
