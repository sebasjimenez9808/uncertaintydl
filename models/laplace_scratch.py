from models.base_model import RegressionMLP
from utils.data_generation import RegressionData
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import torch


class LaplaceReg(RegressionMLP):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 reg_fct: callable, n_hidden: int = 4, n_samples: int = 1000,
                 test_n_samples: int = 1000, wandb_active: bool = False,
                 heteroscedastic: bool = False, train_interval: tuple = (-2, 2),
                 test_interval: tuple = (-3, 3), problem: str = 'regression'):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim,
                         output_dim=output_dim, n_hidden=n_hidden)
        self.epistemic_uncertainty = None
        self.aleatoric_uncertainty = None
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
                                       test_interval=test_interval, train_interval=train_interval,
                                       problem=problem, heteroscedastic=heteroscedastic)
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

    def train_model(self, loss_fct: callable, n_epochs: int = 100, batch_size: int = 32, lr: float = 0.01):
        self.train_map(loss_fct=loss_fct, n_epochs=n_epochs, lr=lr, batch_size=batch_size)
        self.train_laplace(loss_fct=loss_fct, prior_precision=1, sigma_noise=2)

    def train_laplace(self, loss_fct: callable, prior_precision: float = 1., 
                      sigma_noise: float = 0.9):
        
        self.last_layer = False
        self.param_vector = self.params_to_vector()
        self.prior_precision = prior_precision
        self.sigma_noise = sigma_noise
        self.compute_mean_and_cov(self.data_loader, loss_fct)

    def make_predictions_on_test(self):
        pred_la, pred_map = self.predict(self.data_set.test_data.x.unsqueeze(-1), num_samples=1000)
        mean = pred_la[:, :, 0]
        variance_preds = pred_la[:, :, 1]
        variance = torch.exp(variance_preds)
        mean_predictions = mean.mean(dim=0).squeeze().detach().numpy()
        variance_predictions = mean.var(dim=0).squeeze().detach().numpy()
        self.mean_predictions = mean_predictions
        self.aleatoric_uncertainty = variance.mean(dim=0).squeeze().detach().numpy()
        self.epistemic_uncertainty = variance_predictions

    def make_predictions_on_test_classification(self):
        pred_la, pred_map = self.predict(self.data_set.test_data.x.unsqueeze(-1), num_samples=1000)
        self.accuracy = self.get_accuracy(self.data_set.test_data.y, pred_la, stacked=True)

    def make_predictions_on_test_classification_information(self):
        pred_la, pred_map = self.predict(self.data_set.test_data.x.unsqueeze(-1), num_samples=1000)
        self.get_information_theoretical_decomposition(pred_la, stacked=True)

    def params_to_vector(self):
        """
        returns a vector of all model parameters as a stacked vector
        model
        """
        if not self.last_layer:
            param_vector = torch.cat([param.view(-1) for param in self.sequential_model.parameters()])
        else:
            last_layer = list(self.sequential_model.children())[-1]
            param_vector = torch.cat([param.view(-1) for param in last_layer.parameters()])

        self.num_params = param_vector.shape[0]

        return param_vector

    def vector_to_params(self, param_vector):
        """
        returns the individual parameters from a vector

        Args:
            param_vector - given parameter vector to put into model

        """
        weight_idx = 0

        if not self.last_layer:
            param_iterator = self.sequential_model
        else:
            param_iterator = list(self.sequential_model.children())[-1]  # last layer

        for param in param_iterator.parameters():
            param_len = param.numel()

            # updata parameter with param_vector slice
            param.data = param_vector[weight_idx: weight_idx + param_len].view_as(param).data

            weight_idx += param_len

    def gradient(self, model):
        """Collects gradient of model output with respect to parameters.

        Args:
            model: model of which to gather derivates of parameters
        """
        grad = torch.cat([p.grad.data.flatten() for p in model.parameters()])
        return grad.detach()

    def jacobian_params(self, model, data, k=True):
        """Compute Jacobian of parameters.

        Args:
            model: model whose parameters to take gradient of
            data: input data to model

        Returns:
            Jacobian of model output w.r.t. to model parameters

        """
        model.zero_grad()
        output = model(data)
        Jacs = list()
        for i in range(output.shape[1]):
            jacobian = []
            for j in range(output.shape[0]):
                model.zero_grad()  # Zero the gradients before each backward pass

                # Compute the gradient of the output[j][i] with respect to the model parameters
                output[j, i].backward(retain_graph=True)

                # Collect the gradients and append to the jacobian list
                grad = torch.cat([p.grad.data.flatten() for p in model.parameters()])
                grad = grad.detach()

                jacobian.append(grad)

        Jacs.append(torch.stack(jacobian))
        Jacs = torch.cat(Jacs, dim=0)
        return Jacs.detach().squeeze(), output.detach()

    def last_layer_jacobian(self, model, X):
        """Compute Jacobian only of last layer

        Args:
            model: model of which to take the last layer
            X: model input

        Returns:
            Jacobian of model output w.r.t. to last layer parameters

        """
        model_no_last_layer = torch.nn.Sequential(*list(model.children())[:-1])
        last_layer = list(model.children())[-1]
        input_to_last_layer = model_no_last_layer(X)
        jac, map = self.jacobian_params(last_layer, input_to_last_layer)
        return jac, map

    def compute_mean_and_cov(self, train_loader, criterion):
        """
        Compute mean and covariance for laplace approximation with general gauss newton matrix

        Args:
            train_loader: DataLoader
            criterion: Loss criterion used for training
        """
        precision = torch.eye(self.num_params) * self.prior_precision

        self.loss = 0
        self.n_data = len(train_loader.dataset)

        for X, y in train_loader:
            m_out = self.sequential_model(X)
            batch_loss = criterion(m_out, y)
            # jac is of shape N x num_params
            if not self.last_layer:
                jac, _ = self.jacobian_params(self.sequential_model, X)
            else:
                jac, _ = self.last_layer_jacobian(self.sequential_model, X)

            # hess is diagonal matrix of shape of NxN, where N is X.shape[0] or batch_size
            hess = torch.eye(X.shape[0])
            precision += jac.T @ hess @ jac

            self.loss += batch_loss.item()

        self.n_data = len(train_loader.dataset)
        self.map_mean = self.params_to_vector()
        self.H = precision
        self.cov = torch.linalg.inv(precision)

    def linear_sampling(self, X, num_samples=100):
        """Prediction method with linearizing models and mc samples

        Args:
            X: prediction input
            num_samples: how many mc samples to draw

        Returns:
            laplace prediction and model map prediction

        """
        theta_map = self.params_to_vector()

        if not self.last_layer:
            jac, model_map = self.jacobian_params(self.sequential_model, X)
        else:

            jac, model_map = self.last_layer_jacobian(self.sequential_model, X)

        offset = model_map - jac @ theta_map.unsqueeze(-1)

        # reparameterization trick
        covs = self.cov @ torch.randn(len(theta_map), num_samples)

        theta_samples = theta_map + covs.T  # num_samples x num_params
        preds = list()

        for i in range(num_samples):
            pred = offset + jac @ theta_samples[i].unsqueeze(-1)
            preds.append(pred.detach())

        preds = torch.stack(preds)

        return preds, model_map

    def predict(self, X, sampling_method="linear", num_samples=10000):
        """Prediction wrapper method to accustom different sampling methods

        Args:
            X: model input
            sampling_method: which sampling method to choose
            num_samples: how many mc samples to draw

        Returns:
            laplace prediction and model map prediction

        """
        if sampling_method == "linear":
            preds_la, preds_map = self.linear_sampling(X, num_samples)

        return preds_la, preds_map

