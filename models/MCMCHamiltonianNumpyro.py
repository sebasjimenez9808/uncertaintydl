import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

from models.data_generation import generate_synthetic_data

# Generate synthetic data for illustration purposes
np.random.seed(42)


x_train, y_train, y_bernoulli_train = generate_synthetic_data()

# Define the Bayesian Neural Network model
def bayesian_nn(X, y=None):
    num_hidden = 2

    # Define the hyperparameters
    prior_std = 1.0
    num_inputs, num_outputs = 1000, 1000

    # Define the neural network architecture
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((num_inputs, num_hidden)), prior_std))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros((num_hidden,)), prior_std))
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((num_hidden, num_outputs)), prior_std))
    b2 = numpyro.sample("b2", dist.Normal(jnp.zeros((num_outputs,)), prior_std))

    # Define the neural network function
    def neural_network(X):
        h = jax.nn.relu(jnp.dot(X, w1) + b1)
        return jnp.dot(h, w2) + b2

    # Model prediction
    y_pred = numpyro.sample("y_pred", dist.Normal(neural_network(X), 0.1), obs=y)

# Compile the model
nuts_kernel = NUTS(bayesian_nn)
mcmc = MCMC(nuts_kernel, num_warmup=40, num_samples=100, num_chains=1)
mcmc.run(jax.random.PRNGKey(0), x_train, y_train)

# Get posterior samples
posterior_samples = mcmc.get_samples()


# Assuming you have posterior_samples from the MCMC run
def predict(X, posterior_samples):
    w1_samples = posterior_samples["w1"]
    b1_samples = posterior_samples["b1"]
    w2_samples = posterior_samples["w2"]
    b2_samples = posterior_samples["b2"]

    # Generate predictions using samples from the posterior
    predictions = []
    for i in range(len(w1_samples)):
        h = jax.nn.relu(jnp.dot(X, w1_samples[i]) + b1_samples[i])
        y_pred = jnp.dot(h, w2_samples[i]) + b2_samples[i]
        predictions.append(y_pred)

    return jnp.stack(predictions)

# Make predictions
X_new = np.random.rand(10, 1)  # Assuming new data
predicted_samples = predict(x_train, posterior_samples)

# Compute mean and standard deviation of predictions
mean_prediction = jnp.mean(predicted_samples, axis=0)
std_dev_prediction = jnp.std(predicted_samples, axis=0)

x_plot, mean_prediction, std_prediction = zip(*sorted(zip(x_train, mean_prediction, std_dev_prediction)))
x_plot = np.array(x_plot)
mean_prediction = np.array(mean_prediction)
std_prediction = np.array(std_prediction)

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, s=2, alpha=0.7)
plt.plot(x_plot, mean_prediction, label='Mean Prediction', color='red')
plt.fill_between(x_plot, mean_prediction - 2*std_dev_prediction, mean_prediction + 2*std_dev_prediction, alpha=0.2,
                 label='Uncertainty')
plt.xlabel("Independent Variable (x)")
plt.ylabel("Dependent Variable (y)")
plt.title("H. MCMC")
plt.legend()
plt.show()
plt.close()