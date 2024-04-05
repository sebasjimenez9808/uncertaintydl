import numpy as np
from scipy.stats import norm, bernoulli
import random

from torch.utils.data import Dataset
import torch

random.seed(42)


def get_uniform_data(x_range, n_samples):
    return np.random.uniform(x_range[0], x_range[1], n_samples)


def generate_bootstrap_samples(x, y, bootstrap_size: float = 0.6, n_models: int = 100):
    """Generates bootstrap samples from the training data.
    Args:
        x (torch.Tensor): The independent variable data.
        y (torch.Tensor): The dependent variable data.
        num_samples (int, optional): The number of bootstrap samples to generate. Defaults to 100.

    Returns:
        tuple: The generated bootstrap samples for x and y.
        :param n_models:
        :param bootstrap_size:
    """
    x_samples = []
    y_samples = []
    for _ in range(n_models):
        np.random.seed(_ + 1 * 10 + 42)
        indices = np.random.choice(len(x), int(len(x) * bootstrap_size),
                                   replace=True)  # Generate random indices with replacement
        x_samples.append(torch.from_numpy(x[indices]))
        y_samples.append(torch.from_numpy(y[indices]))

    return torch.stack(x_samples), torch.stack(y_samples)


class BootstrapDataset(Dataset):

    def __init__(self, x, y,
                 bootstrap_size: float = 0.6, n_models: int = 100):
        self.x, self.y = generate_bootstrap_samples(x, y, bootstrap_size, n_models)

        # make index_bootstrap a class attribute
        self.index_bootstrap = 0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[self.index_bootstrap, idx].view(-1, 1),
                self.y[self.index_bootstrap, idx].view(-1, 1))


class RegressionDataset(Dataset):
    def __init__(self,
                 fct,
                 n_samples: int = 1000,
                 eps_std: float = 0.4,
                 test_interval: tuple = (0, 12),
                 train_interval: tuple = (0, 7),
                 test_n_samples: int = 1000):
        self.fct = fct
        self.n_samples = n_samples
        self.eps_std = eps_std
        self.test_interval = test_interval
        self.train_interval = train_interval
        self.test_n_samples = test_n_samples
        self.x_train = get_uniform_data(self.train_interval, self.n_samples)
        self.y_train = fct(self.x_train) + norm.rvs(size=len(self.x_train), scale=self.eps_std)
        self.x_test = get_uniform_data(self.test_interval, self.test_n_samples)
        self.y_test = fct(self.x_test) + norm.rvs(size=len(self.x_test), scale=self.eps_std)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return torch.tensor(self.x_train[idx], dtype=torch.float32).view(-1, 1), torch.tensor(self.y_train[idx],
                                                                                              dtype=torch.float32).view(
            -1, 1)


class ClassificationDataset(Dataset):
    def __init__(self,
                 fct,
                 n_samples: int = 1000,
                 test_interval: tuple = (0, 12),
                 train_interval: tuple = (0, 7),
                 test_n_samples: int = 1000):
        self.fct = fct
        self.n_samples = n_samples
        self.test_interval = test_interval
        self.train_interval = train_interval
        self.test_n_samples = test_n_samples
        self.x_train = get_uniform_data(self.train_interval, self.n_samples)
        self.y_train = bernoulli.rvs(size=len(self.x_train), p=self.fct(self.x_train))
        self.x_test = get_uniform_data(self.test_interval, self.test_n_samples)
        self.y_test = bernoulli.rvs(size=len(self.x_test), p=self.fct(self.x_test))

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx, train=True):
        if train:
            return torch.tensor(self.x_train[idx], dtype=torch.float32), torch.tensor(self.y_train[idx],
                                                                                      dtype=torch.float32)
        else:
            return torch.tensor(self.x_test[idx], dtype=torch.float32), torch.tensor(self.y_test[idx],
                                                                                     dtype=torch.float32)


def target_function(x):
    """Calculates the value of the function (1/2) * (sin³(2 * pi * x) - 1)

    Args:
        x (float or numpy.ndarray): The input value(s) for the function.

    Returns:
        float or numpy.ndarray: The output value(s) of the function.
    """
    # return 0.5 * (np.sin(2 * np.pi * x) ** 3 + 1)
    return 0.5 * x ** 3

    # Generate independent variable data
