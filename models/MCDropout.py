import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import wandb

from models.base_model import RegressionMLP, EvaluationModel
from utilities.data_generation import RegressionData
import torch.utils.data as data_utils


class MCDropoutNet(EvaluationModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 reg_fct: callable, n_samples: int = 1000,
                 test_n_samples: int = 1000,
                 n_hidden: int = 4, dropout_p=0.3,
                 problem: str = 'regression',
                 train_interval: tuple = (-3, 3),
                 test_interval: tuple = (-5, 5),
                 samples_prediction: int = 100,
                 wandb_active: bool = False, heteroscedastic: bool = False,
                 seed: int = 42, add_sigmoid: bool = False):
        self.base_model = RegressionMLP(input_dim=input_dim, hidden_dim=hidden_dim,
                                        output_dim=output_dim, n_hidden=n_hidden,
                                        add_sigmoid=add_sigmoid, seed=seed,
                                        add_dropout=True, dropout_rate=dropout_p)
        self.total_uncertainty = None
        self.test_mse = None
        self.training_mse = None
        self.standard_deviation_predictions = None
        self.variance_predictions = None
        self.mean_predictions = None
        self.dropout = nn.Dropout(p=dropout_p)  # Apply dropout to all layers
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
        self.dropout_p = dropout_p
        self.samples_prediction = samples_prediction
        self.training_time = None

    def activate_wandb(self):
        if self.wandb_active:
            wandb.init(
                project="Thesis",
                name="MC Dropout",
                config={
                    "architecture": "MC Dropout",
                    "dataset": "regression",
                    "hidden_dim": self.hidden_dim,
                    "n_hidden": self.n_hidden,
                    "n_samples": self.n_samples,
                    "dropout_p": self.dropout_p
                }
            )

    def finish_wandb(self):
        if self.wandb_active:
            wandb.finish()

    def log_wandb(self, epoch: int, **kwargs):
        if self.wandb_active:
            wandb.log(kwargs, step=epoch)

    def train_model(self, loss_fct: callable, n_epochs: int = 100, lr: float = 0.01,
                    batch_size: int = 32):
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=lr)
        data_loader = DataLoader(data_utils.TensorDataset(self.data_set.train_data.x.unsqueeze(-1),
                                                          self.data_set.train_data.y.unsqueeze(-1)),
                                 batch_size=batch_size)
        self.activate_wandb()

        lost_all_epochs = []
        mse_all_epochs = []
        for epoch in range(n_epochs):
            loss_this_epoch = []
            mse_this_epoch = []
            for x, y in data_loader:
                x = x.float()
                y = y.float()
                optimizer.zero_grad()
                pred = self.base_model(x)
                mu = pred[:, 0]
                loss = loss_fct(pred, y)
                loss.backward()
                mse_this_epoch.append(self.get_mse(y, mu))
                loss_this_epoch.append(loss.item())
                optimizer.step()
            loss_this_epoch = sum(loss_this_epoch) / len(loss_this_epoch)
            mse_this_epoch = sum(mse_this_epoch) / len(mse_this_epoch)
            lost_all_epochs.append(loss_this_epoch)
            mse_all_epochs.append(mse_this_epoch)
            self.log_wandb(epoch, loss=loss_this_epoch, mse=mse_this_epoch)

        self.model_loss = lost_all_epochs
        self.training_mse = mse_all_epochs

        self.finish_wandb()

    def get_mse(self, test_vals, predictions):
        return np.mean((test_vals.squeeze() - predictions.squeeze()).detach().numpy() ** 2)

    def forward_test_with_dropout(self, x):
        """ Forward pass data with dropout for test data """
        predictions_mean = []
        predictions_sigma = []
        for i in range(self.samples_prediction):
            with torch.no_grad():
                pred = self.base_model(x.squeeze(-1))
            predictions_mean.append(pred[:, 0])
            predictions_sigma.append(torch.exp(pred[:, 1]))
        return torch.stack(predictions_mean, dim=0), torch.stack(predictions_sigma, dim=0)

    def make_predictions_on_test(self):
        x_test = self.data_set.test_data.x.unsqueeze(1).unsqueeze(1)
        prediction = self.forward_test_with_dropout(x_test)
        predictions_mu, predictions_sigma = prediction
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
        for i in range(self.samples_prediction):
            with torch.no_grad():
                pred = self.base_model(x_test.squeeze(-1))
                predictions.append(pred)
        self.accuracy = self.get_accuracy(self.data_set.test_data.y, predictions)
        print('Accuracy:', self.accuracy)

    def make_predictions_on_test_classification_information(self):
        x_test = self.data_set.test_data.x.unsqueeze(1).unsqueeze(1)
        predictions = []
        for i in range(self.samples_prediction):
            with torch.no_grad():
                pred = self.base_model(x_test.squeeze(-1))
                predictions.append(pred)
        self.get_information_theoretical_decomposition(predictions)
