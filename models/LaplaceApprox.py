from models.base_model import RegressionMLP
from laplace.baselaplace import FullLaplace
from laplace.curvature.backpack import BackPackGGN
from laplace import Laplace, marglik_training

from utils.data_generation import RegressionData
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import torch


class LaplaceRegLA(RegressionMLP):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 reg_fct: callable, n_hidden: int = 4, n_samples: int = 1000,
                 test_n_samples: int = 1000, wandb_active: bool = False):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim,
                         output_dim=output_dim, n_hidden=n_hidden)
        self.data_loader = None
        self.test_mse = None
        self.variance_predictions = None
        self.laplace_approximation = None
        self.params_hmc_gpu = None
        self.tau_list = None
        self.tau_out = None
        self.lower = None
        self.upper = None
        self.lower_al = None
        self.upper_al = None
        self.mean_predictions = None
        self.standard_deviation_predictions = None
        self.standard_deviation_aleatoric = None
        self.test_data_set = None
        self.data_set = RegressionData(reg_fct, n_samples=n_samples, test_n_samples=test_n_samples,
                                       test_interval=(-5, 5), train_interval=(-3, 3))
        self.reg_fct = reg_fct

        self.model_loss = None

        self.sequential_model = torch.nn.Sequential(
            self.fc1, torch.nn.Tanh(), self.fc2,
            torch.nn.Tanh(), self.fc_mu)
        self.wandb_active = wandb_active

    def train_map(self, loss_fct: callable, n_epochs: int = 100, lr: float = 0.01,
                  batch_size: int = 32):
        optimizer = torch.optim.Adam(self.sequential_model.parameters(), lr=lr)
        # self.data_loader = DataLoader(self.data_set.train_data, batch_size=batch_size)
        self.data_loader = DataLoader(data_utils.TensorDataset(self.data_set.train_data.x.unsqueeze(-1),
                                                               self.data_set.train_data.y.unsqueeze(-1)),
                                      batch_size=batch_size)
        lost_all_epochs = []
        for epoch in range(n_epochs):
            loss_this_epoch = []
            for x, y in self.data_loader:
                x = x.float()
                y = y.float()
                optimizer.zero_grad()
                pred = self.sequential_model(x)
                loss = loss_fct(pred, y)
                loss.backward()
                loss_this_epoch.append(loss.item())
                optimizer.step()
            lost_all_epochs.append(np.mean(loss_this_epoch))
        self.model_loss = lost_all_epochs

    def laplace_approx(self, n_epochs: int = 100):

        la = Laplace(self.sequential_model, 'regression', subset_of_weights='all', hessian_structure='full')
        la.fit(self.data_loader)
        log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
        losses = []
        for i in range(n_epochs):
            hyper_optimizer.zero_grad()
            neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
            losses.append(neg_marglik.item())
            neg_marglik.backward()
            hyper_optimizer.step()

        self.model_loss = losses
        self.laplace_approximation = la

    def train_model(self, n_epochs: int = 100, batch_size: int = 32, lr: float = 0.01,
                    loss_fct: callable = torch.nn.MSELoss()):
        self.train_map(loss_fct=torch.nn.MSELoss(), n_epochs=n_epochs, lr=lr, batch_size=batch_size)
        self.laplace_approx(n_epochs=n_epochs)

    def make_predictions_on_test(self):
        f_mu, f_var = self.laplace_approximation(self.data_set.test_data.x.unsqueeze(-1))
        self.mean_predictions = f_mu.squeeze().detach().cpu().numpy()
        #self.epistemic_uncertainty = f_var[:, 0, 0]  # f var is the car covar matrix of mu and sigma - this is variance of means
        #self.mean_predictions = f_mu[:, 0].cpu().numpy()
        #self.aleatoric_uncertainty = f_mu[:, 1].cpu().numpy()
        y_values_numpy = self.data_set.test_data.y.squeeze().numpy()
        self.test_mse = np.mean((y_values_numpy - self.mean_predictions) ** 2)
