import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from utilities.metrics import get_kl_divergence


def plot_predictions_with_uncertainty(model, title: str = '', filename: str = '',
                                      y_lim: tuple = (-40, 40),
                                      x_training_lim: tuple = (-4, 4)):
    x_values_test = model['test_data']['x']
    x_values_train = model['train_data']['x']
    y_values_test = model['test_data']['y']
    y_values_train = model['train_data']['y']
    predictions_mean = model['test_data']['predictions']
    aleatoric_unceratinty = model['test_data']['aleatoric_uncertainty']
    epistemic_uncertainty = model['test_data']['epistemic_uncertainty']

    plt.figure(figsize=(10, 6))

    plt.fill_between(x_values_test, predictions_mean - epistemic_uncertainty,
                     predictions_mean + epistemic_uncertainty, alpha=0.3,
                     label='Epistemic Uncertainty', color='red')

    plt.fill_between(x_values_test, predictions_mean - aleatoric_unceratinty,
                     predictions_mean + aleatoric_unceratinty, alpha=0.3,
                     color='green', label='Aleatoric Uncertainty')

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
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_uncertainties(model, title: str = '', filename: str = '',
                       y_lim: tuple = (-40, 40),
                       x_training_lim: tuple = (-4, 4)):
    x_values_test = model['test_data']['x']
    aleatoric_unceratinty = model['test_data']['aleatoric_uncertainty']
    epistemic_uncertainty = model['test_data']['epistemic_uncertainty']

    plt.figure(figsize=(10, 6))

    plt.scatter(x_values_test, aleatoric_unceratinty, s=12, label='Aleatoric Uncertainty', color='green')
    plt.scatter(x_values_test, epistemic_uncertainty, s=12, label='Epistemic Uncertainty', color='red')

    plt.axvline(x=x_training_lim[0], color='gray', linestyle='--')
    plt.axvline(x=x_training_lim[1], color='gray', linestyle='--')

    plt.xlabel("Independent Variable (x)")
    plt.ylabel("Uncertainty")
    plt.title(f"{title} Uncertainties")
    plt.ylim(*y_lim)
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_kl_divergence(model, model_ref, title: str = '', filename: str = '',
                       y_lim: tuple = (-40, 40),
                       x_training_lim: tuple = (-4, 4)):
    indices = [i for i in range(len(model['test_data']['x']))]
    kl_divergence = get_kl_divergence(model=model, model_ref=model_ref, indices=indices, return_array=True)

    plt.figure(figsize=(10, 6))

    plt.scatter(model['test_data']['x'], kl_divergence['all'], s=12, label='KL Divergence', color='blue')
    plt.scatter(model['test_data']['x'], kl_divergence['aleatoric'], s=12, label='Aleatoric KL Divergence', color='red')
    plt.scatter(model['test_data']['x'], kl_divergence['epistemic'], s=12, label='Epistemic KL Divergence',
                color='green')

    plt.axvline(x=x_training_lim[0], color='gray', linestyle='--')
    plt.axvline(x=x_training_lim[1], color='gray', linestyle='--')

    plt.xlabel("Independent Variable (x)")
    plt.ylabel("KL Divergence")
    plt.title(f"{title} KL Divergence")
    plt.ylim(*y_lim)
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()


def grid_three_plots(model, model_ref, title: str = '', filename: str = '',
                     y_lim: tuple = (-40, 40),
                     x_training_lim: tuple = (-4, 4)):
    """
    Creates a grid of three columns and one row with plots for predictions, uncertainties, and kl divergence
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    x_values_test = model['test_data']['x']
    y_values_test = model['test_data']['y']
    predictions_mean = model['test_data']['predictions']
    aleatoric_unceratinty = model['test_data']['aleatoric_uncertainty']
    epistemic_uncertainty = model['test_data']['epistemic_uncertainty']

    axs[0].fill_between(x_values_test, predictions_mean - epistemic_uncertainty,
                        predictions_mean + epistemic_uncertainty, alpha=0.3,
                        label='Epistemic Uncertainty', color='red')
    axs[0].fill_between(x_values_test, predictions_mean - aleatoric_unceratinty,
                        predictions_mean + aleatoric_unceratinty, alpha=0.3,
                        color='green', label='Aleatoric Uncertainty')
    axs[0].plot(x_values_test, predictions_mean, label='Predictions', color='purple')
    axs[0].scatter(x_values_test, y_values_test, s=12, label='Test', color='orange')
    axs[0].axvline(x=x_training_lim[0], color='gray', linestyle='--')
    axs[0].axvline(x=x_training_lim[1], color='gray', linestyle='--')
    axs[0].set_ylim(*y_lim)
    axs[0].set_title(f"{title} Predictions")
    axs[0].legend()

    axs[1].scatter(x_values_test, aleatoric_unceratinty, s=12, label='Aleatoric Uncertainty', color='green')
    axs[1].scatter(x_values_test, epistemic_uncertainty, s=12, label='Epistemic Uncertainty', color='red')
    axs[1].axvline(x=x_training_lim[0], color='gray', linestyle='--')
    axs[1].axvline(x=x_training_lim[1], color='gray', linestyle='--')
    axs[1].set_ylim(0, y_lim[1])
    axs[1].set_title(f"{title} Uncertainties")
    axs[1].legend()

    indices = [i for i in range(len(x_values_test))]
    kl_divergence = get_kl_divergence(model=model, model_ref=model_ref, indices=indices, return_array=True)

    axs[2].scatter(x_values_test, kl_divergence['all'], s=12, label='KL Divergence', color='blue')
    axs[2].scatter(x_values_test, kl_divergence['aleatoric'], s=12, label='Aleatoric KL Divergence', color='red')
    axs[2].scatter(x_values_test, kl_divergence['epistemic'], s=12, label='Epistemic KL Divergence',
                   color='green')
    axs[2].axvline(x=x_training_lim[0], color='gray', linestyle='--')
    axs[2].axvline(x=x_training_lim[1], color='gray', linestyle='--')
    axs[2].set_ylim(0, y_lim[1])
    axs[2].set_title(f"{title} KL Divergence")
    axs[2].legend()

    plt.savefig(filename)
    plt.show()
    plt.close()


def grid_model_list_plots(models_list, model_ref, title: str = '', filename: str = '',
                          model_names: list = None,
                          y_lim: tuple = (-40, 40),
                          x_training_lim: tuple = (-4, 4),
                          fig_size: tuple = (30, 5),
                          title_plots_fontsize: int = 20,
                          title_rows_fontsize: int = 25,
                          legends_fontsize: int = 20):
    """
    Creates a grid of three columns and one row with plots for predictions, uncertainties, and kl divergence for each
    model in model_list
    """
    fig, axs = plt.subplots(len(models_list), 3, figsize=(fig_size[0], fig_size[1]))

    for i, model in enumerate(models_list):
        model_ref_copy = copy.deepcopy(model_ref)
        x_values_test = model['test_data']['x']
        y_values_test = model['test_data']['y']
        predictions_mean = model['test_data']['predictions']
        aleatoric_unceratinty = model['test_data']['aleatoric_uncertainty']
        epistemic_uncertainty = model['test_data']['epistemic_uncertainty']

        axs[i][0].fill_between(x_values_test, predictions_mean - epistemic_uncertainty,
                               predictions_mean + epistemic_uncertainty, alpha=0.3,
                               label='Epistemic Uncertainty', color='red')
        axs[i][0].fill_between(x_values_test, predictions_mean - aleatoric_unceratinty,
                               predictions_mean + aleatoric_unceratinty, alpha=0.3,
                               color='green', label='Aleatoric Uncertainty')
        axs[i][0].plot(x_values_test, predictions_mean, label='Predictions', color='purple')
        axs[i][0].scatter(x_values_test, y_values_test, s=12, color='orange')
        axs[i][0].axvline(x=x_training_lim[0], color='gray', linestyle='--')
        axs[i][0].axvline(x=x_training_lim[1], color='gray', linestyle='--')
        axs[i][0].set_ylim(*y_lim)
        if i == 0:
            axs[i][0].set_facecolor('lightblue')
            axs[i][0].set_title(f"{title} Predictions", fontsize=title_plots_fontsize)
            axs[i][0].legend(fontsize=legends_fontsize)
        axs[i][0].set_ylabel(model_names[i], fontsize=title_rows_fontsize)

        axs[i][1].scatter(x_values_test, aleatoric_unceratinty, s=12, label='Aleatoric Uncertainty', color='green')
        axs[i][1].scatter(x_values_test, epistemic_uncertainty, s=12, label='Epistemic Uncertainty', color='red')
        axs[i][1].axvline(x=x_training_lim[0], color='gray', linestyle='--')
        axs[i][1].axvline(x=x_training_lim[1], color='gray', linestyle='--')
        axs[i][1].set_ylim(0, y_lim[1])
        if i == 0:
            axs[i][1].set_facecolor('lightblue')
            axs[i][1].set_title(f"{title} Uncertainties", fontsize=title_plots_fontsize)
            axs[i][1].legend(fontsize=legends_fontsize)

        indices = [i for i in range(len(x_values_test))]

        if model_names[i] == 'Reference':
            # compare with the data generating process
            model_ref_copy['test_data']['aleatoric_uncertainty'] = np.abs(model['test_data']['noise']) ** 2

        print(model_names[i])
        kl_divergence = get_kl_divergence(model=model, model_ref=model_ref_copy, indices=indices, return_array=True)

        if model_names[i] == 'Reference':
            kl_divergence['aleatoric'] = uniform_filter1d(kl_divergence['aleatoric'], 5, mode='constant')

        axs[i][2].scatter(x_values_test, kl_divergence['aleatoric'], s=12, label='Aleatoric KL Divergence', color='red')

        if model_names[i] != 'Reference':
            axs[i][2].scatter(x_values_test, kl_divergence['all'], s=12, label='KL Divergence', color='blue')
            axs[i][2].scatter(x_values_test, kl_divergence['epistemic'], s=12, label='Epistemic KL Divergence',
                              color='green')
        axs[i][2].axvline(x=x_training_lim[0], color='gray', linestyle='--')
        axs[i][2].axvline(x=x_training_lim[1], color='gray', linestyle='--')
        axs[i][2].set_ylim(0, y_lim[1])
        if i == 0:
            axs[i][2].set_facecolor('lightblue')
            axs[i][2].set_title(f"{title} KL Divergence", fontsize=title_plots_fontsize)
        if i <= 1:
            axs[i][2].legend(fontsize=legends_fontsize)

        axs[i][0].tick_params(axis='both', which='major', labelsize=legends_fontsize)  # Adjust labelsize as needed
        axs[i][1].tick_params(axis='both', which='major', labelsize=legends_fontsize)
        axs[i][2].tick_params(axis='both', which='major', labelsize=legends_fontsize)

    plt.savefig(filename)
    plt.show()
    plt.close()
