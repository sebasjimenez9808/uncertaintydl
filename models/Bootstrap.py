import sys
import os
conf_path = os.getcwd()
print(conf_path)
sys.path.append(conf_path)
import torch
import torch.nn as nn
import numpy as np

from models.base_model import RegressionMLP, EvaluationModel
from utilities.data_generation import BootstrapDataset, RegressionData
from torch.utils.data import DataLoader
from tqdm import tqdm
#import wandb
import torch.utils.data as data_utils


class BootstrapEnsemble(EvaluationModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 reg_fct: callable,
                 n_hidden: int = 4, n_models: int = 100,
                 n_samples: int = 1000,
                 test_n_samples: int = 1000,
                 bootstrap_size: float = 0.6,
                 wandb_active: bool = False,
                 heteroscedastic: bool = False,
                 test_interval: tuple = (-5, 5),
                 train_interval: tuple = (-3, 3),
                 problem: str = 'regression',
                 add_sigmoid: bool = False, seed: int = 42,
                 **kwargs):
        self.total_uncertainty = None
        self.aleatoric_uncertainty = None
        self.epistemic_uncertainty = None
        self.test_mse = None
        self.training_mse = None
        self.standard_deviation_predictions = None
        self.variance_predictions = None
        self.mean_predictions = None
        self.n_models = n_models
        self.bootstrap_size = bootstrap_size
        self.models = [RegressionMLP(input_dim=input_dim, hidden_dim=hidden_dim,
                                     output_dim=output_dim, n_hidden=n_hidden,
                                     add_sigmoid=add_sigmoid, seed=seed + _) for _ in range(n_models)]
        self.data_set = RegressionData(reg_fct, n_samples=n_samples, test_n_samples=test_n_samples,
                                       heteroscedastic=heteroscedastic,
                                       test_interval=test_interval,
                                       train_interval=train_interval,
                                       problem=problem
                                       )
        self.model_loss = None
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples
        self.wandb_active = wandb_active
        self.output_dim = output_dim
        self.training_time = None

    def activate_wandb(self):
        if self.wandb_active:
            wandb.init(
                project="Thesis",
                name="Bootstrap",
                config={
                    "architecture": "Bootstrap",
                    "dataset": "regression",
                    "hidden_dim": self.hidden_dim,
                    "n_hidden": self.n_hidden,
                    "n_samples": self.n_samples,
                }
            )

    def finish_wandb(self):
        if self.wandb_active:
            wandb.finish()

    def log_wandb(self, epoch: int, **kwargs):
        if self.wandb_active:
            wandb.log(kwargs, step=epoch)

    def get_sample_loader(self, bootstrap_sample: BootstrapDataset, batch_size: int):
        data_loader_by_model = {}
        # set seed 123
        torch.manual_seed(123)
        for index, model in enumerate(self.models):
            bootstrap_sample.index_bootstrap = index
            data_loader_by_model[index] = DataLoader(
                data_utils.TensorDataset(bootstrap_sample.x[index, :].unsqueeze(-1),
                                         bootstrap_sample.y[index, :].unsqueeze(-1)),
                batch_size=batch_size, shuffle=True)
        return data_loader_by_model

    def get_optimizer(self, lr: float):
        optimizer_by_model = {}
        for index, model in enumerate(self.models):
            optimizer_by_model[index] = torch.optim.Adam(model.parameters(), lr=lr)
        return optimizer_by_model

    def train_model(self, loss_fct: callable, n_epochs: int = 100, lr: float = 0.01,
                    batch_size: int = 32):
        bootstrap_sample = BootstrapDataset(self.data_set.train_data.x,
                                            self.data_set.train_data.y,
                                            self.bootstrap_size, self.n_models)

        data_loader = self.get_sample_loader(bootstrap_sample, batch_size)
        optimizer = self.get_optimizer(lr)
        self.activate_wandb()

        for i, model in enumerate(tqdm(self.models)):
            model.to('cuda')
            mse_this_model = []
            loss_this_model = []
            for epoch in range(n_epochs):
                for x, y in data_loader[i]:
                    optimizer[i].zero_grad()
                    pred = model(x)
                    # mu, sigma = pred[:, 0], pred[:, 1]
                    loss = loss_fct(pred, y)
                    loss.backward()
                    optimizer[i].step()
                    loss_this_model.append(loss.item())
                    # mse_this_model.append(self.get_mse(y, mu))
                # loss_this_model = sum(loss_this_model) / len(loss_this_model)
                # self.log_wandb(epoch=epoch, loss=loss_this_model)

            # self.log_wandb(epoch=epoch, loss=mean_loss_this_epoch, mse=mean_mse_this_epoch)

        # self.model_loss = loss_by_epoch
        # self.training_mse = mse_by_epoch

        self.finish_wandb()

    def get_mse(self, test_vals, predictions):
        return np.mean((test_vals.squeeze() - predictions.squeeze()).detach().numpy() ** 2)

    def forward_test(self, x):
        """ make predictions of the shape (mu, sigma) """
        predictions_mu = []
        predictions_sigma = []
        for model in self.models:
            with torch.no_grad():
                prediction = model(x.squeeze(-1))
            predictions_mu.append(prediction[:, 0])
            predictions_sigma.append(torch.exp(prediction[:, 1]))
        return torch.stack(predictions_mu, dim=0), torch.stack(predictions_sigma, dim=0)

    def make_predictions_on_test(self):
        x_test = self.data_set.test_data.x.unsqueeze(1).unsqueeze(1)
        predictions_mu, predictions_sigma = self.forward_test(x_test)
        mean_vector = np.array(predictions_mu.squeeze())
        variance_vector = np.array(predictions_sigma.squeeze())
        self.mean_predictions = np.mean(mean_vector, axis=0)
        self.aleatoric_uncertainty = np.mean(variance_vector, axis=0)
        self.epistemic_uncertainty = np.var(mean_vector, axis=0)
        self.total_uncertainty = self.epistemic_uncertainty + self.aleatoric_uncertainty
        self.standard_deviation_predictions = np.sqrt(self.epistemic_uncertainty)
        y_values_numpy = self.data_set.test_data.y.squeeze().numpy()
        self.test_mse = np.mean((y_values_numpy - self.mean_predictions) ** 2)

    def make_predictions_on_test_classification(self):
        x_test = self.data_set.test_data.x.unsqueeze(1).unsqueeze(1)
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x_test.squeeze(-1))
                predictions.append(pred)
        self.accuracy = self.get_accuracy(self.data_set.test_data.y, predictions)
        print('Accuracy:', self.accuracy)

    def make_predictions_on_test_classification_information(self):
        x_test = self.data_set.test_data.x.unsqueeze(1).unsqueeze(1)
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x_test.squeeze(-1))
                pred_sig = torch.sigmoid(pred.squeeze(1))
                predictions.append(pred_sig)

        self.get_information_theoretical_decomposition(predictions)
