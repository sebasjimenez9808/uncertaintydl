import numpy as np
import matplotlib.pyplot as plt

from utilities.data_generation import generate_regression_data, cubic_function, heterocedastic_noise


def generate_aleatoric_epistemic_plot():
    """
    Generates plot to ilustrate aleatoric and epistemic uncertainties by simulating epistemic
    to be a function of the data and aleatoric just the noise.
    """
    x_train, y_train, noise_train = generate_regression_data(fct=cubic_function, n_samples=100,
                                                             eps_std=0.4, train_interval=(-3, 3),
                                                             heteroscedastic=True, seed=123)

    x_test, y_test, noise_test = generate_regression_data(fct=cubic_function, n_samples=100,
                                                          eps_std=0.4, train_interval=(-4, 4),
                                                          heteroscedastic=True, seed=123)

    x_train = x_train.numpy().flatten()
    y_train = y_train.numpy().flatten()
    x_test = x_test.numpy().flatten()
    y_test = y_test.numpy().flatten()
    indices = np.argsort(x_train)
    x_train = x_train[indices]
    y_train = y_train[indices]
    indices = np.argsort(x_test)
    x_test = x_test[indices]
    y_test = y_test[indices]

    x_uniform = np.linspace(-4, 4, 100)
    y_target = cubic_function(x_uniform)
    epistemic_uncertainty = np.abs(x_uniform * 2)
    aleatoric_uncertainty = (0.15 + (4 - np.round(np.abs(x_uniform - np.mean((-4, 4))), 1)) * 2)

    plt.figure(figsize=(10, 6))

    plt.fill_between(x_uniform, y_target - epistemic_uncertainty,
                     y_target + epistemic_uncertainty, alpha=0.5,
                     label='Epistemic Uncertainty', color='green')

    plt.fill_between(x_uniform, y_target - aleatoric_uncertainty,
                     y_target + aleatoric_uncertainty, alpha=0.3,
                     label='Aleatoric Uncertainty', color='red')

    plt.scatter(x_test, y_test, s=15, label='Test', color='black',
                marker='x')
    plt.scatter(x_train, y_train, s=15, label='Train', color='blue')

    plt.xlabel("Independent Variable (x)")
    plt.ylabel("Dependent Variable (y)")
    plt.legend()
    plt.savefig(f'plots/simulation/aleatoric_epistemic.png')
