import numpy as np
import copy
from scipy.stats import norm, bernoulli
from torch.utils.data import Dataset
import torch


def get_uniform_data(x_range, n_samples):
    np.random.seed(123)
    return np.random.uniform(x_range[0], x_range[1], n_samples)


def generate_y_data_regression(heteroscedastic: bool, fct, x, eps_std, train_interval):
    if not heteroscedastic:
        return fct(x) + norm.rvs(size=len(x), scale=eps_std)
    else:
        return fct(x) + heterocedastic_noise(x, train_interval)


def generate_labels_classification(fct, x, heteroscedastic: bool = False, train_interval: tuple = (0, 7)):
    p_unstable = fct(x)
    max_value = p_unstable.max()
    min_value = p_unstable.min()
    if heteroscedastic:
        p_unstable = p_unstable + heterocedastic_noise(x, train_interval)
        # take the minimum between p_unstable and max_value
        p_unstable = np.minimum(p_unstable, max_value)
        # take the maximum between p_unstable and min_value
        p_unstable = np.maximum(p_unstable, min_value)

    return p_unstable, bernoulli.rvs(size=len(x), p=p_unstable)


def generate_regression_data(fct, n_samples: int = 1000, eps_std: float = 0.4, train_interval: tuple = (0, 7),
                             heteroscedastic: bool = False):
    x = get_uniform_data(train_interval, n_samples)
    y = generate_y_data_regression(heteroscedastic, fct, x, eps_std, train_interval)
    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def generate_classification_data(fct, n_samples: int = 1000, train_interval: tuple = (0, 7), heteroscedastic: bool = False):
    x = get_uniform_data(train_interval, n_samples)
    p, y = generate_labels_classification(fct, x, train_interval=train_interval, heteroscedastic=heteroscedastic)
    return torch.from_numpy(x).float(), torch.from_numpy(y).float(), p


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
    # set seed 123
    np.random.seed(123)
    for _ in range(n_models):
        np.random.seed(_ + 1 * 10 + 42)
        indices = np.random.choice(len(x), int(len(x) * bootstrap_size),
                                   replace=True)  # Generate random indices with replacement
        indices = torch.from_numpy(indices)
        x_samples.append(x[indices])
        y_samples.append(y[indices])

    return torch.stack(x_samples), torch.stack(y_samples)


class BootstrapDataset(Dataset):

    def __init__(self, x: np.array, y: np.array,
                 bootstrap_size: float = 0.6, n_models: int = 100):
        self.x, self.y = generate_bootstrap_samples(x, y, bootstrap_size, n_models)

        # make index_bootstrap a class attribute
        self.index_bootstrap = 0

    def __len__(self):
        return len(self.x[0,])

    def __getitem__(self, idx):
        return (self.x[idx, :].view(-1, 1),
                self.y[idx, :].view(-1, 1))


class RegressionDataset(Dataset):
    def __init__(self,
                 fct,
                 n_samples: int = 1000,
                 eps_std: float = 0.4,
                 train_interval: tuple = (0, 7),
                 heteroscedastic: bool = False,
                 problem: str = 'regression'):
        self.fct = fct
        self.n_samples = n_samples
        self.eps_std = eps_std
        self.train_interval = train_interval
        if problem == 'regression':
            self.x, self.y = generate_regression_data(fct, n_samples=n_samples, eps_std=eps_std,
                                                      train_interval=train_interval,
                                                      heteroscedastic=heteroscedastic)
        else:
            self.x, self.y, self.p = generate_classification_data(fct, n_samples=n_samples,
                                                                  train_interval=train_interval,
                                                                  heteroscedastic=heteroscedastic)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx].view(-1, 1), self.y[idx].view(-1, 1))


class RegressionData:
    def __init__(self,
                 fct,
                 n_samples: int = 1000,
                 eps_std: float = 0.4,
                 test_interval: tuple = (-3, 3),
                 train_interval: tuple = (-2, 2),
                 test_n_samples: int = 1000,
                 heteroscedastic: bool = False,
                 problem: str = 'regression'):
        self.train_data = RegressionDataset(fct, n_samples=n_samples, eps_std=eps_std, train_interval=train_interval,
                                            heteroscedastic=heteroscedastic, problem=problem)
        self.test_data = RegressionDataset(fct, n_samples=test_n_samples, eps_std=eps_std, train_interval=test_interval,
                                           heteroscedastic=heteroscedastic, problem=problem)


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
    return 0.5 * x ** 3


def target_function_sine(x):
    return 0.5 * (np.sin(2 * np.pi * x) + 1)


def heterocedastic_noise(x: np.array, interval: tuple, eps_std: float = 0.7):
    """ creates an array of noise that depends on the distance between the input and the center interval"""
    return (0.3 + interval[1] - np.round(np.abs(x - np.mean(interval)), 1)) * norm.rvs(size=len(x), scale=eps_std)
