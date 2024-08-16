import time
import pandas as pd

import matplotlib.pyplot as plt
from models.Bootstrap import BootstrapEnsemble
from models.MCDropout import MCDropoutNet
from models.MCMCHamiltorch import MCMCReg
from models.deep_ensemble import DeepEnsemble
from models.laplace_scratch import LaplaceReg
from models.old_vi import VIModel
from utilities.data_generation import cubic_function, target_function_sine
from utilities.losses import negative_log_likelihood, classification_loss, classification_loss_vector
import numpy as np

from utilities.metrics import *


def generate_model(model_type, input_dim, hidden_dim, output_dim, reg_fct, n_hidden, n_samples,
                   test_n_samples, wandb_active=False, heteroscedastic: bool = False, num_samples: int = 1000,
                   problem: str = 'regression', train_interval: tuple = (-3, 3), test_interval: tuple = (-5, 5),
                   loss_fct: callable = None, loss_fct_vector: callable = None, seed: int = 42,
                   n_models: int = 100, samples_prediction: int = 1000, add_sigmoid: bool = False,
                   dropout_p: float = 0.3, bootstrap_size: float = 0.6,
                   **kwargs):
    if model_type == "ensemble":
        return BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                 reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                                 test_n_samples=test_n_samples, wandb_active=wandb_active,
                                 heteroscedastic=heteroscedastic, problem=problem,
                                 train_interval=train_interval, test_interval=test_interval,
                                 seed=seed, n_models=n_models, bootstrap_size=bootstrap_size,
                                 add_sigmoid=add_sigmoid,
                                 **kwargs)

    elif model_type == "mcdropout":
        return MCDropoutNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                            reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                            test_n_samples=test_n_samples, heteroscedastic=heteroscedastic,
                            problem=problem, train_interval=train_interval, test_interval=test_interval,
                            dropout_p=dropout_p, samples_prediction=samples_prediction,
                            seed=seed, add_sigmoid=add_sigmoid,
                            **kwargs)
    elif model_type == "variational":
        return VIModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                       test_n_samples=test_n_samples, heteroscedastic=heteroscedastic,
                       problem=problem, train_interval=train_interval, test_interval=test_interval,
                       seed=seed, add_sigmoid=add_sigmoid, n_samples_predictions=samples_prediction,
                       **kwargs)
    elif model_type == "deep_ensemble":
        return DeepEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                            reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                            test_n_samples=test_n_samples, problem=problem,
                            heteroscedastic=heteroscedastic, train_interval=train_interval,
                            test_interval=test_interval, n_models=n_models, add_sigmoid=add_sigmoid,
                            seed=seed, wandb_active=wandb_active, **kwargs)
    elif model_type == 'hamiltonian':
        return MCMCReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                       test_n_samples=test_n_samples, num_samples=samples_prediction,
                       heteroscedastic=heteroscedastic, problem=problem, train_interval=train_interval,
                       test_interval=test_interval, loss_fct=loss_fct_vector, seed=seed, add_sigmoid=add_sigmoid,
                       **kwargs)
    elif model_type == 'laplace':
        return LaplaceReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                          reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                          test_n_samples=test_n_samples, heteroscedastic=heteroscedastic,
                          problem=problem, train_interval=train_interval, test_interval=test_interval,
                          seed=seed, add_sigmoid=add_sigmoid, num_samples=samples_prediction)
    else:
        raise ValueError("Model type not recognized")


def train_model(model, **kwargs):
    args = {**kwargs}
    if not isinstance(model, MCMCReg):
        args = {k: v for k, v in args.items() if k not in ['step_size', 'num_steps_per_sample', 'burn', 'tau']}
    model.train_model(**args)


def plot_model_loss(model, index_to_plot: int = 0):
    if isinstance(model, DeepEnsemble):
        loss = model.model_loss[index_to_plot]
    else:
        loss = model.model_loss
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
                         num_samples: int = 1000, problem: str = 'regression',
                         **kwargs):
    model = generate_model(model_type=model, input_dim=input_dim, hidden_dim=hidden_dim,
                           output_dim=output_dim, reg_fct=reg_fct, n_hidden=n_hidden,
                           num_samples=num_samples, problem=problem, **kwargs)
    train_model(model, loss_fct=loss_fct, n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    model.make_predictions_on_test_classification()
    return model


def plot_predictions(model, title: str = ''):
    x_values = model.data_set.test_data.x.numpy()
    x_values_train = model.data_set.train_data.x.numpy()
    if hasattr(model.data_set.test_data, 'p'):
        y_values = model.data_set.test_data.p
        y_values_train = model.data_set.train_data.p
    else:
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
    plt.plot(x_values, predictions_mean, label='Predictions', color='red')

    plt.scatter(x_values_train, y_values_train, s=4, alpha=0.7, label='Train')
    plt.scatter(x_values, y_values, s=4, alpha=0.7, label='Test')
    plt.fill_between(x_values, predictions_mean - epistemic_uncertainty,
                     predictions_mean + epistemic_uncertainty, alpha=0.3, label='Epistemic Uncertainty')
    plt.fill_between(x_values, predictions_mean - epistemic_uncertainty - aleatoric_unceratinty,
                     predictions_mean + epistemic_uncertainty + aleatoric_unceratinty, alpha=0.3,
                     label='Aleatoric Uncertainty')
    plt.xlabel("Independent Variable (x)")
    plt.ylabel("Dependent Variable (y)")
    plt.title(f"{title.split('_')[0]} Predictions")
    plt.ylim(-20, 20)
    plt.legend()
    plt.savefig(f'plots/classification/{title}.png')
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
    plt.title(f"{title.split('_')[0]} Uncertainties")
    plt.ylim(0, 10)
    plt.legend()
    plt.savefig(f'plots/classification/{title}_uncertainties.png')
    plt.show()
    plt.close()


# %%
params = {
    'input_dim': 1,
    'hidden_dim': 40,
    'output_dim': 2,
    'reg_fct': target_function_sine,
    'n_hidden': 1,
    'loss_fct': classification_loss,
    'loss_fct_vector': classification_loss_vector,
    'lr': 0.0005,
    'batch_size': 30,
    'n_samples': 1500,
    'test_n_samples': 500,
    'n_epochs': 1500,
    'problem': 'classification',
    'train_interval': (0, 0.75),
    'test_interval': (0, 1),
    'seed': 42,
    'heteroscedastic': False,
    'wandb_active': False,
    'add_sigmoid': True,
    'samples_prediction': 2000,  # samples to take in bayesian models
    'n_models': 300,  # n models for ensemble
    'dropout_p': 0.3,  # for mc dropout
    'step_size': 0.0005,  # for hamiltonian
    'num_steps_per_sample': 700,
    'burn': 100,
    'tau': 1,
    'bootstrap_size': 0.6,  # for ensemble
}

preffix_name = 'c1500'
suffix_name = '_inf'

model_vi = train_and_test_model("variational", **params)
plot_predictions(model_vi, f"{preffix_name}variational_ht")
plot_uncertainties(model_vi, f"{preffix_name}variational_ht")

model_la = train_and_test_model("laplace", **params)
print('accuracy:', model_la.accuracy)
print('aleatoric entropy', model_la.aleatoric_entropy.mean())
print('epistemic uncertainty', model_la.epistemic_uncertainty.mean())
print('aleatoric uncertainty', model_la.aleatoric_uncertainty.mean())
plot_predictions(model_la, f"{preffix_name}laplace_ht")
plot_uncertainties(model_la, f"{preffix_name}laplace_ht")

model_ham = train_and_test_model("hamiltonian", **params)
print('accuracy:', model_ham.accuracy)
print('aleatoric entropy', model_ham.aleatoric_entropy.mean())
print('epistemic uncertainty', model_ham.epistemic_uncertainty.mean())
print('aleatoric uncertainty', model_ham.aleatoric_uncertainty.mean())
plot_predictions(model_ham, f"{preffix_name}hamiltonian_ht")
plot_uncertainties(model_ham, f"{preffix_name}hamiltonian_ht")


time_start = time.time()
model_deep = train_and_test_model("deep_ensemble", **params)
time_end = time.time()
print(f"Time taken: {time_end - time_start}")
plot_predictions(model_deep, f"{preffix_name}deep_ensemble_ht")
plot_uncertainties(model_deep, f"{preffix_name}deep_ensemble_ht")

model_mc = train_and_test_model("mcdropout", **params)
model = model_mc
plot_predictions(model_mc, f"{preffix_name}mcdropout_ht")
plot_uncertainties(model_mc, f"{preffix_name}mcdropout_ht")

# ensemble
model_boots = train_and_test_model("ensemble", **params)
plot_predictions(model_boots, f"{preffix_name}ensemble_ht")
plot_uncertainties(model_boots, f"{preffix_name}ensemble_ht")

# # This is to get the reference distribution
model_ref = train_and_test_model("deep_ensemble", resample=True, **params)
plot_predictions(model_ref, f"{preffix_name}reference")


# %%

