import pymc as pm
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
from models.data_generation import generate_synthetic_data
import torch

torch.manual_seed(42)
np.random.seed(42)

x_train, y_train, y_bernoulli_train = generate_synthetic_data()
# make this arrays tensors
with pm.Model() as model:
    x_training = pm.Deterministic("x_training", x_train.detach().numpy())
x_train = torch.Tensor(x_train).reshape(-1, 1)
y_train = torch.Tensor(y_train)

class Model(pm.Model):
    def __init__(self):
        super(Model, self).__init__()

        # Priors
        self.mu = pm.Normal('mu', mu=0, sigma=10)
        self.sigma = pm.HalfNormal('sigma', sigma=1)

        # Likelihood
        self.likelihood = pm.Normal('likelihood', mu=self.mu, sigma=self.sigma, observed=x_train)

# Create an MCMC object and sample
model = Model()
mcmc = pm.sampling.mcmc.sample(model=model, tune=1000, draws=1000, chains=4)

dict_samples = mcmc.to_dict()


with pm.Model() as model:
    #x_train = pm.Deterministic("x_train", var=x_train.detach().numpy())
    #y_train = pm.Deterministic("y_train", var=y_train.detach().numpy())
    # Priors for weights and biases (adapt shapes as needed)
    weights_prior = pm.Normal('weights_prior', mu=0, sigma=1, shape=(5, 4))  # Input to hidden layer
    biases_prior = pm.Normal('biases_prior', mu=0, sigma=1, shape=(4,))
    weights_out = pm.Normal('weights_out', mu=0, sigma=1, shape=(4, 1))  # Hidden to output
    biases_out = pm.Normal('biases_out', mu=0, sigma=1, shape=(1,))

    # Model layers with non-linear activations
    hidden = pm.Deterministic('hidden', pm.math.tanh(pm.math.dot(x_train, weights_prior) + biases_prior))
    output = pm.Deterministic('output', pm.math.sigmoid(pm.math.dot(hidden, weights_out) + biases_out))

    # Likelihood for binary classification
    likelihood = pm.Normal('likelihood', p=output, observed=y_train)

trace = pm.sampling.mcmc.sample(model=model, tune=1000, draws=1000, chains=4)  # Adjust parameters as needed
