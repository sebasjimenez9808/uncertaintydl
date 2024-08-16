from tqdm import tqdm
from models.base_model import RegressionMLP, EvaluationModel
from utilities.data_generation import RegressionData
from torch.utils.data import DataLoader
import torch
import numpy as np
import copy
import wandb
import torch.utils.data as data_utils


class DeepEnsemble(EvaluationModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 reg_fct: callable, n_samples: int = 1000, test_n_samples: int = 1000,
                 n_hidden: int = 4, n_models: int = 100,
                 wandb_active: bool = False,
                 heteroscedastic: bool = False,
                 resample: bool = False, problem: str = 'regression',
                 test_interval: tuple = (-5, 5), train_interval: tuple = (-3, 3),
                 seed: int = 42, add_sigmoid: bool = False,
                 **kwargs):
        self.test_mse = None
        self.standard_deviation_predictions = None
        self.variance_predictions = None
        self.mean_predictions = None
        self.n_models = n_models
        self.models = [RegressionMLP(input_dim=input_dim, hidden_dim=hidden_dim,
                                     output_dim=output_dim, n_hidden=n_hidden,
                                     add_sigmoid=add_sigmoid, seed=seed + _) for _ in range(n_models)]
        self.data_set = RegressionData(reg_fct, n_samples=n_samples, test_n_samples=test_n_samples,
                                       heteroscedastic=heteroscedastic,
                                       test_interval=test_interval,
                                       train_interval=train_interval,
                                       problem=problem)
        self.add_sigmoid = add_sigmoid
        self.problem = problem
        self.output_dim = output_dim
        self.model_loss = None
        self.n_samples = n_samples
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.wandb_active = wandb_active
        self.resample = resample
        self.reg_fct = reg_fct
        self.heteroscadastic = heteroscedastic
        self.test_interval = test_interval
        self.train_interval = train_interval
        self.training_time = None


    def activate_wandb(self):
        if self.wandb_active:
            wandb.init(
                project="Thesis",
                name="Deep Ensemble",
                config={
                    "architecture": "Deep Ensemble",
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

    def get_sample_loader(self, batch_size: int, resample: bool = False):
        data_loader_by_model = {}

        for index, model in enumerate(self.models):
            torch.manual_seed(123 + index)
            if self.resample:
                new_data_set = RegressionData(self.reg_fct, n_samples=self.n_samples,
                                              heteroscedastic=self.heteroscadastic,
                                              test_interval=self.test_interval,
                                              train_interval=self.train_interval,
                                              seed=42 + index,
                                              problem=self.problem)
                shuffle_index = torch.randperm(len(new_data_set.train_data.y))
                shuffle_data = copy.deepcopy(new_data_set.train_data)
            else:
                shuffle_index = torch.randperm(len(self.data_set.train_data.y))
                shuffle_data = copy.deepcopy(self.data_set.train_data)
            shuffle_data.x = shuffle_data.x[shuffle_index]
            shuffle_data.y = shuffle_data.y[shuffle_index]
            data_loader_by_model[index] = DataLoader(data_utils.TensorDataset(shuffle_data.x.unsqueeze(-1),
                                                                              shuffle_data.y.unsqueeze(-1)),
                                                     batch_size=batch_size, shuffle=True)
        return data_loader_by_model

    def get_optimizer(self, lr: float):
        optimizer_by_model = {}
        for index, model in enumerate(self.models):
            optimizer_by_model[index] = torch.optim.Adam(model.parameters(), lr=lr)
        return optimizer_by_model

    def train_model(self, loss_fct: callable, n_epochs: int = 100, lr: float = 0.01,
                    batch_size: int = 32, resample: bool = False, **kwargs):

        data_loader = self.get_sample_loader(batch_size, resample=resample)
        optimizer = self.get_optimizer(lr)
        self.activate_wandb()

        loss_by_epoch = []
        mse_by_epoch = []
        for epoch in tqdm(range(n_epochs)):
            loss_this_epoch = []
            mse_this_epoch = []
            for i, model in enumerate(self.models):
                loss_this_model = []
                mse_this_model = []
                for x, y in data_loader[i]:
                    optimizer[i].zero_grad()
                    pred = model(x)
                    #mu, sigma = pred[:, 0], pred[:, 1]
                    loss = loss_fct(pred, y)
                    loss.backward()
                    optimizer[i].step()
                    loss_this_model.append(loss.item())
                    #mse_this_model.append(self.get_mse(y, mu))
                loss_this_model = sum(loss_this_model) / len(loss_this_model)
                #mse_this_model = sum(mse_this_model) / len(mse_this_model)

                loss_this_epoch.append(loss_this_model)
                #mse_this_epoch.append(mse_this_model)

            mean_loss_this_epoch = sum(loss_this_epoch) / len(loss_this_epoch)
            #mean_mse_this_epoch = sum(mse_this_epoch) / len(mse_this_epoch)

            loss_by_epoch.append(mean_loss_this_epoch)
            #mse_by_epoch.append(mean_mse_this_epoch)
            #elf.log_wandb(epoch=epoch, loss=mean_loss_this_epoch, mse=mean_mse_this_epoch)

        self.model_loss = loss_by_epoch
        self.training_mse = mse_by_epoch

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
                prediction_mu, prediction_sigma = prediction[:, 0], prediction[:, 1]
            predictions_mu.append(prediction_mu)
            predictions_sigma.append(torch.exp(prediction_sigma))
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
                prediction = model(x_test.squeeze(-1))
                predictions.append(prediction)
        accuracy = self.get_accuracy(self.data_set.test_data.y, predictions)
        print('Accuracy:', accuracy)
        return accuracy

    def make_predictions_on_test_classification_information(self):
        x_test = self.data_set.test_data.x.unsqueeze(1).unsqueeze(1)
        predictions = []
        for model in self.models:
            with torch.no_grad():
                prediction = model(x_test.squeeze(-1))
                prediction_sig = torch.sigmoid(prediction.squeeze(1))
                predictions.append(prediction_sig)

        self.get_information_theoretical_decomposition(predictions)
