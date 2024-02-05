import numpy as np
from scipy.stats import norm, bernoulli
import random

random.seed(42)


# Define the target function
def target_function(x):
    """Calculates the value of the function (1/2) * (sin³(2 * pi * x) - 1)

    Args:
        x (float or numpy.ndarray): The input value(s) for the function.

    Returns:
        float or numpy.ndarray: The output value(s) of the function.
    """
    # return 0.5 * (np.sin(2 * np.pi * x) ** 3 + 1)
    return ((0.5*x ** 2) + 1+np.sin(x))/3


# Generate independent variable data
def generate_independent_variable(num_samples=1000):
    """Generates independent variable data from the specified distribution.

    Args:
        num_samples (int, optional): The number of samples to generate. Defaults to 1000.

    Returns:
        numpy.ndarray: The generated independent variable data.
    """
    x = np.concatenate([
        np.random.uniform(0, 7, int(0.9 * num_samples)),  # 90% from [1, 11]
        np.random.uniform(7, 10, int(0.1 * num_samples))  # 10% from [11, 12]
    ])
    np.random.shuffle(x)  # Shuffle the data for randomness
    return x


# Generate dependent variable with noise
def generate_dependent_variable_with_noise(x):
    """Generates dependent variable data with additive normal noise.

    Args:
        x (numpy.ndarray): The independent variable data.

    Returns:
        numpy.ndarray: The generated dependent variable data with noise.
    """

    y = target_function(x) + norm.rvs(size=len(x), scale=0.1)  # Add normal noise
    return y


# Generate dependent variable from Bernoulli distribution
def generate_bernoulli_dependent_variable(x):
    """Generates dependent variable data from a Bernoulli distribution.

    Args:
        x (numpy.ndarray): The independent variable data.

    Returns:
        numpy.ndarray: The generated dependent variable data as Bernoulli trials.
    """
    p = target_function(x)  # Use target function as success probability
    # make p all positive
    p = (p - np.min(p)) / (np.max(p) - np.min(p))
    y = bernoulli.rvs(size=len(x), p=p)  # Generate Bernoulli trials
    return y


# Generate synthetic data with different dependent variables
def generate_synthetic_data(num_samples=1000):
    """Generates synthetic data with independent and dependent variables (with/without noise).

    Args:
        num_samples (int, optional): The number of samples to generate. Defaults to 1000.

    Returns:
        tuple: A tuple containing various combinations of independent and dependent data.
    """
    x = generate_independent_variable(num_samples)
    y_with_noise = generate_dependent_variable_with_noise(x)
    y_bernoulli = generate_bernoulli_dependent_variable(x)
    return x, y_with_noise, y_bernoulli

# Example usage
