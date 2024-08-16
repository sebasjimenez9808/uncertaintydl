import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from utilities.metrics import get_kl_divergence


def plot_predictions_with_uncertainty(model, title: str = '', filename: str = '',
                                      y_lim: tuple = (0, 1),
                                      x_training_lim: tuple = (0, 0.75)):
    x_values_test = model['test_data']['x']
    y_values_test = model['test_data']['noise']
    predictions_mean = model['test_data']['predictions']
    aleatoric_unceratinty = model['test_data']['aleatoric_uncertainty']
    epistemic_uncertainty = model['test_data']['epistemic_uncertainty']

    plt.figure(figsize=(10, 6))

    # plt.fill_between(x_values_test, predictions_mean - epistemic_uncertainty,
    #                  predictions_mean + epistemic_uncertainty, alpha=0.3,
    #                  label='Epistemic Uncertainty', color='red')
    #
    # plt.fill_between(x_values_test, predictions_mean - aleatoric_unceratinty,
    #                  predictions_mean + aleatoric_unceratinty, alpha=0.3,
    #                  color='green', label='Aleatoric Uncertainty')

    plt.plot(x_values_test, predictions_mean, label='Predictions', color='purple')

    plt.scatter(x_values_test, y_values_test, s=12, label='Test', color='orange')

    # Add vertical lines
    plt.axvline(x=x_training_lim[0], color='gray', linestyle='--')
    plt.axvline(x=x_training_lim[1], color='gray', linestyle='--')

    plt.xlabel("Independent Variable (x)")
    plt.ylabel("Dependent Variable (y)")
    plt.title(f"{title} Predictions")
    plt.ylim(*y_lim)
    plt.legend()
#    plt.savefig(filename)
    plt.show()
    plt.close()
