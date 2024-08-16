import matplotlib.pyplot as plt
from models.Bootstrap import BootstrapEnsemble
from models.MCDropout import MCDropoutNet
from models.MCMCHamiltorch import MCMCReg
from models.SVI import VariationalInference
from models.deep_ensemble import DeepEnsemble
from models.laplace_scratch import LaplaceReg
from models.old_vi import VIModel
from utilities.data_generation import cubic_function, target_function_sine
from utilities.losses import negative_log_likelihood, classification_loss, negative_log_likelihood_vector
import numpy as np
import copy
import time
import json

from utilities.metrics import *


def generate_model(model_type, input_dim, hidden_dim, output_dim, reg_fct, n_hidden, n_samples,
                   test_n_samples, wandb_active=False, heteroscedastic: bool = False, num_samples: int = 1000,
                   problem: str = 'regression', train_interval: tuple = (-3, 3), test_interval: tuple = (-5, 5),
                   loss_fct: callable = None, seed: int = 42, samples_prediction: int = 500, n_models: int = 100,
                   loss_fct_vector: callable = None, **kwargs):
    if model_type == "ensemble":
        return BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                 reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                                 test_n_samples=test_n_samples, wandb_active=wandb_active,
                                 heteroscedastic=heteroscedastic, train_interval=train_interval,
                                 test_interval=test_interval, seed=seed, n_models=n_models,
                                 **kwargs)

    elif model_type == "mcdropout":
        return MCDropoutNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                            reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                            test_n_samples=test_n_samples, heteroscedastic=heteroscedastic,
                            problem=problem, train_interval=train_interval,
                            test_interval=test_interval, seed=seed, samples_prediction=samples_prediction,
                            **kwargs)
    elif model_type == "variational":
        return VIModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                       test_n_samples=test_n_samples, heteroscedastic=heteroscedastic,
                       train_interval=train_interval, seed=seed,
                       test_interval=test_interval, n_samples_predictions=samples_prediction)
    elif model_type == "deep_ensemble":
        return DeepEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                            reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                            test_n_samples=test_n_samples, problem=problem,
                            heteroscedastic=heteroscedastic, train_interval=train_interval,
                            test_interval=test_interval, seed=seed, n_models=n_models,
                            **kwargs)
    elif model_type == 'hamiltonian':
        return MCMCReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                       test_n_samples=test_n_samples, num_samples=samples_prediction,
                       heteroscedastic=heteroscedastic, train_interval=train_interval,
                       test_interval=test_interval, loss_fct=loss_fct_vector, seed=seed)
    elif model_type == 'laplace':
        return LaplaceReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                          reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                          test_n_samples=test_n_samples, heteroscedastic=heteroscedastic, train_interval=train_interval,
                          test_interval=test_interval, seed=seed, num_samples=samples_prediction)
    else:
        raise ValueError("Model type not recognized")


def train_model(model, **kwargs):
    model.train_model(**kwargs)


def plot_model_loss(model, index_to_plot: int = 0):
    if isinstance(model, DeepEnsemble):
        loss = model.model_loss[index_to_plot]
    elif isinstance(model, BootstrapEnsemble):
        loss = model.model_loss
    elif isinstance(model, MCDropoutNet):
        loss = model.model_loss
    elif isinstance(model, VariationalInference):
        loss = model.model_loss
    elif isinstance(model, MCMCReg):
        loss = model.model_loss
    elif isinstance(model, LaplaceReg):
        loss = model.model_loss
    else:
        raise ValueError("Model type not recognized")
    plt.figure(figsize=(10, 6))
    plt.scatter([i for i in range(len(loss))], loss, s=2, alpha=0.7)
    plt.xlabel("Independent Variable (x)")
    plt.ylabel("Dependent Variable (y)")
    plt.title("Synthetic Data with Bimodal Distribution")
    # write code to save figure
    plt.savefig('data.png')
    plt.show()
    plt.close()


def train_and_test_model(model: str, input_dim: int, hidden_dim: int, output_dim: int,
                         reg_fct: callable, n_hidden: int, loss_fct: callable = None,
                         n_epochs: int = 100, lr: float = 0.01, batch_size: int = 32,
                         num_samples: int = 1000, problem: str = 'regression', seed: int = 42,
                         samples_prediction: int = 500, n_models: int = 100,
                         loss_fct_vector: callable = None, train_interval: tuple = (-3, 3),
                         test_interval: tuple = (-5, 5), **kwargs):
    model = generate_model(model_type=model, input_dim=input_dim, hidden_dim=hidden_dim,
                           output_dim=output_dim, reg_fct=reg_fct, n_hidden=n_hidden,
                           num_samples=num_samples, problem=problem, loss_fct=loss_fct,
                           samples_prediction=samples_prediction, n_models=n_models,
                           loss_fct_vector=loss_fct_vector, seed=seed, train_interval=train_interval,
                           test_interval=test_interval, **kwargs)
    train_model(model, loss_fct=loss_fct, n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    model.make_predictions_on_test()

    return model


def plot_predictions(model, title: str = ''):
    x_values = model.data_set.test_data.x.numpy()
    x_values_train = model.data_set.train_data.x.numpy()
    y_values = model.data_set.test_data.y.numpy()
    y_values_train = model.data_set.train_data.y.numpy()

    predictions_mean = model.mean_predictions
    aleatoric_unceratinty = model.aleatoric_uncertainty
    epistemic_uncertainty = model.epistemic_uncertainty

    indices = np.argsort(x_values)
    x_values = x_values[indices]
    y_values = y_values[indices]
    predictions_mean = predictions_mean[indices]
    aleatoric_unceratinty = aleatoric_unceratinty[indices]
    epistemic_uncertainty = epistemic_uncertainty[indices]

    plt.figure(figsize=(10, 6))

    plt.fill_between(x_values, predictions_mean - epistemic_uncertainty - aleatoric_unceratinty,
                     predictions_mean - epistemic_uncertainty, alpha=0.3,
                     label='Aleatoric Uncertainty', color='red')

    plt.fill_between(x_values, predictions_mean + epistemic_uncertainty,
                     predictions_mean + epistemic_uncertainty + aleatoric_unceratinty, alpha=0.3,
                     color='red')

    plt.fill_between(x_values, predictions_mean - epistemic_uncertainty,
                     predictions_mean + epistemic_uncertainty, label='Epistemic Uncertainty',
                     color='green')

    plt.plot(x_values, predictions_mean, label='Predictions', color='red')

    plt.scatter(x_values_train, y_values_train, s=4, label='Train', color='black')
    plt.scatter(x_values, y_values, s=4, label='Test', color='orange')
    plt.xlabel("Independent Variable (x)")
    plt.ylabel("Dependent Variable (y)")
    plt.title(f"{title.split('_')[1]} Predictions")
    plt.ylim(-20, 20)
    plt.legend()
    plt.savefig(f'plots/regression/{title}.png')
    plt.show()
    plt.close()


def plot_uncertainties(model, title: str = ''):
    """ Plot uncertainties in the y axis with respect to data instances in the x axis """
    x_values = model.data_set.test_data.x.numpy()
    aleatoric_unceratinty = model.aleatoric_uncertainty
    epistemic_uncertainty = model.epistemic_uncertainty

    indices = np.argsort(x_values)
    x_values = x_values[indices]
    aleatoric_unceratinty = aleatoric_unceratinty[indices]
    epistemic_uncertainty = epistemic_uncertainty[indices]

    plt.figure(figsize=(10, 6),
               )
    plt.scatter(x_values, aleatoric_unceratinty, s=2, alpha=0.7, label='Aleatoric Uncertainty')
    plt.scatter(x_values, epistemic_uncertainty, s=2, alpha=0.7, label='Epistemic Uncertainty')
    plt.xlabel("Independent Variable (x)")
    plt.ylabel("Uncertainty")
    plt.title(f"{title.split('_')[1]} Uncertainties")
    plt.ylim(0, 10)
    plt.legend()
    plt.savefig(f'plots/regression/{title}_uncertainties.png')
    plt.show()
    plt.close()
