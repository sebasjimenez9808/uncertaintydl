import matplotlib.pyplot as plt
from models.Bootstrap import BootstrapEnsemble
from models.MCDropout import MCDropoutNet
from models.MCMCHamiltorch import MCMCReg
from models.deep_ensemble import DeepEnsemble
from models.laplace_scratch import LaplaceReg
from models.old_vi import VIModel
from utils.data_generation import target_function, target_function_sine
from utils.losses import negative_log_likelihood, classification_loss
import numpy as np

from utils.metrics import *


def generate_model(model_type, input_dim, hidden_dim, output_dim, reg_fct, n_hidden, n_samples,
                   test_n_samples, wandb_active=False, heteroscedastic: bool = False, num_samples: int = 1000,
                   problem: str = 'regression', train_interval: tuple = (-3, 3), test_interval: tuple = (-5, 5),
                   **kwargs):
    if model_type == "ensemble":
        return BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                 reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                                 test_n_samples=test_n_samples, wandb_active=wandb_active,
                                 heteroscedastic=heteroscedastic, problem=problem,
                                 train_interval=train_interval, test_interval=test_interval,
                                 **kwargs)

    elif model_type == "mcdropout":
        return MCDropoutNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                            reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                            test_n_samples=test_n_samples, heteroscedastic=heteroscedastic,
                            problem=problem, train_interval=train_interval, test_interval=test_interval,
                            **kwargs)
    elif model_type == "variational":
        return VIModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                    reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                                    test_n_samples=test_n_samples, heteroscedastic=heteroscedastic,
                                    problem=problem, train_interval=train_interval, test_interval=test_interval)
    elif model_type == "deep_ensemble":
        return DeepEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                            reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                            test_n_samples=test_n_samples, problem=problem,
                            heteroscedastic=heteroscedastic,
                            **kwargs)
    elif model_type == 'hamiltonian':
        return MCMCReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                       test_n_samples=test_n_samples, num_samples=num_samples,
                       heteroscedastic=heteroscedastic, problem=problem, train_interval=train_interval,
                       test_interval=test_interval)
    elif model_type == 'laplace':
        return LaplaceReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                          reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                          test_n_samples=test_n_samples, heteroscedastic=heteroscedastic,
                          problem=problem, train_interval=train_interval, test_interval=test_interval)
    else:
        raise ValueError("Model type not recognized")


def train_model(model, **kwargs):
    model.train_model(**kwargs)


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
    if problem == 'regression':
        model.make_predictions_on_test()
    else:
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
    # plt.xlabel("Independent Variable (x)")
    # plt.ylabel("Dependent Variable (y)")
    # plt.title(f"{title.split('_')[0]} Predictions")
    # plt.ylim(-20, 20)
    plt.legend()
    # plt.savefig(f'{title}.png')
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
    plt.savefig(f'{title}_uncertainties.png')
    plt.show()
    plt.close()


# %%

model_vi = train_and_test_model("variational", input_dim=1, hidden_dim=10, output_dim=4,
                                reg_fct=target_function_sine, n_hidden=2, loss_fct=classification_loss,
                                n_epochs=5000, lr=0.0005, batch_size=32, kl_weight=0.12,
                                n_samples=1000, test_n_samples=500, heteroscedastic=False,
                                problem='classification',
                                train_interval=(0, 0.5), test_interval=(0, 1))
plot_predictions(model_vi, "variational_ht")
plot_uncertainties(model_vi, "variational_ht")

model_la = train_and_test_model("laplace", input_dim=1, hidden_dim=50, output_dim=4,
                                reg_fct=target_function_sine, n_hidden=2, loss_fct=classification_loss,
                                n_epochs=1500, lr=0.0005, batch_size=150, n_samples=500, test_n_samples=500,
                                heteroscedastic=True, problem='classification',
                                train_interval=(0, 0.5), test_interval=(0, 1))
print('accuracy:', model_la.accuracy)
print('aleatoric entropy', model_la.aleatoric_entropy.mean())
print('epistemic uncertainty', model_la.epistemic_uncertainty.mean())
print('aleatoric uncertainty', model_la.aleatoric_uncertainty.mean())
plot_predictions(model_la, "laplace_ht")
plot_uncertainties(model_la, 'laplace_ht')

model_ham = train_and_test_model("hamiltonian", input_dim=1, hidden_dim=40, output_dim=4,
                                 reg_fct=target_function_sine, n_hidden=1, num_samples=1000, step_size=0.0005,
                                 num_steps_per_sample=700, n_samples=1000, test_n_samples=500,
                                 burn=150, tau=1000, heteroscedastic=True, problem='classification',
                                 train_interval=(0, 0.5), test_interval=(0, 0.5))
print('accuracy:', model_ham.accuracy)
print('aleatoric entropy', model_ham.aleatoric_entropy.mean())
print('epistemic uncertainty', model_ham.epistemic_uncertainty.mean())
print('aleatoric uncertainty', model_ham.aleatoric_uncertainty.mean())
plot_predictions(model_ham, "hamiltonian_ht")
plot_uncertainties(model_ham, "hamiltonian_ht")

import time

time_start = time.time()
model_deep = train_and_test_model("deep_ensemble", input_dim=1, hidden_dim=40, output_dim=4,
                                  reg_fct=target_function_sine, n_hidden=4, loss_fct=classification_loss,
                                  n_epochs=1000, lr=0.0005, batch_size=40, n_samples=1000, test_n_samples=500,
                                  heteroscedastic=True, problem='classification', train_interval=(0, 0.5),
                                  test_interval=(0, 1))
time_end = time.time()
print(f"Time taken: {time_end - time_start}")
plot_predictions(model_deep, "deep_ensemble_ht")
plot_uncertainties(model_deep, "deep_ensemble_ht")

model_mc = train_and_test_model("mcdropout", input_dim=1, hidden_dim=40, output_dim=4,
                                reg_fct=target_function_sine, n_hidden=4, loss_fct=classification_loss,
                                n_epochs=1000, lr=0.0005, batch_size=40, dropout_p=0.2,
                                test_n_samples=400, n_samples=1000, wandb_active=False,
                                heteroscedastic=True, problem='classification',
                                train_interval=(0, 0.5), test_interval=(0, 1))
model = model_mc
plot_predictions(model_mc, "mcdropout_ht")
plot_uncertainties(model_mc, "mcdropout_ht")

# ensemble
model_boots = train_and_test_model("ensemble", input_dim=1, hidden_dim=40, output_dim=4,
                                   reg_fct=target_function_sine, n_hidden=4, loss_fct=classification_loss,
                                   n_epochs=1000, lr=0.0005, batch_size=32, n_models=100, bootstrap_size=0.6,
                                   n_samples=1000, test_n_samples=500, wandb_active=False,
                                   heteroscedastic=True, problem='classification',
                                   train_interval=(0, 0.5), test_interval=(0, 1))
plot_predictions(model_boots, "ensemble_ht")
plot_uncertainties(model_boots, "ensemble_ht")

# # This is to get the reference distribution
model_ref = train_and_test_model("deep_ensemble", input_dim=1, hidden_dim=40, output_dim=2,
                                 reg_fct=target_function, n_hidden=4, loss_fct=negative_log_likelihood,
                                 n_epochs=400, lr=0.035, batch_size=40, n_samples=200, test_n_samples=500,
                                 resample=True, heteroscedastic=True, wandb_active=False)
plot_predictions(model_ref, "deep_ensemble_ref_heteroscedastic")

# %%


# %%
kl_div_vi = get_kl_divergence(model_ref, model_vi)
kl_div_ham = get_kl_divergence(model_ref, model_ham)
kl_div_deep = get_kl_divergence(model_ref, model_deep)
kl_div_mc = get_kl_divergence(model_ref, model_mc)
kl_div_boots = get_kl_divergence(model_ref, model_boots)

wasserstein_vi = get_wasserstein_distance(model_ref, model_vi)
wasserstein_ham = get_wasserstein_distance(model_ref, model_ham)
wasserstein_deep = get_wasserstein_distance(model_ref, model_deep)
wasserstein_mc = get_wasserstein_distance(model_ref, model_mc)
wasserstein_boots = get_wasserstein_distance(model_ref, model_boots)

predictive_capacity_vi = predictive_capacity(model_vi)
predictive_capacity_ham = predictive_capacity(model_ham)
predictive_capacity_deep = predictive_capacity(model_deep)
predictive_capacity_mc = predictive_capacity(model_mc)
predictive_capacity_boots = predictive_capacity(model_boots)

predictive_accuracy_vi = accuracy(model_vi)
predictive_accuracy_ham = accuracy(model_ham)
predictive_accuracy_deep = accuracy(model_deep)
predictive_accuracy_mc = accuracy(model_mc)
predictive_accuracy_boots = accuracy(model_boots)

root_mean_squared_error_vi = rmsce(model_vi)
root_mean_squared_error_ham = rmsce(model_ham)
root_mean_squared_error_deep = rmsce(model_deep)
root_mean_squared_error_mc = rmsce(model_mc)
root_mean_squared_error_boots = rmsce(model_boots)

unceratainty_compared_to_data_vi = nipg(model_vi, model_ref)
unceratainty_compared_to_data_ham = nipg(model_ham, model_ref)
unceratainty_compared_to_data_deep = nipg(model_deep, model_ref)
unceratainty_compared_to_data_mc = nipg(model_mc, model_ref)
unceratainty_compared_to_data_boots = nipg(model_boots, model_ref)

# %%
import pandas as pd

distance_metrics = pd.DataFrame({
    'Model': ['Variational Inference', 'Hamiltonian Monte Carlo', 'Deep Ensemble', 'MC Dropout', 'Ensemble'],
    'KL Divergence': [kl_div_vi, kl_div_ham, kl_div_deep, kl_div_mc, kl_div_boots],
    'Wasserstein Distance': [wasserstein_vi, wasserstein_ham, wasserstein_deep, wasserstein_mc, wasserstein_boots]
})

# %%

metrics = pd.DataFrame({
    'Model': ['Variational Inference', 'Hamiltonian Monte Carlo', 'Deep Ensemble', 'MC Dropout', 'Ensemble'],
    'Predictive Capacity': [predictive_capacity_vi, predictive_capacity_ham, predictive_capacity_deep,
                            predictive_capacity_mc, predictive_capacity_boots],
    'Predictive Accuracy': [predictive_accuracy_vi, predictive_accuracy_ham, predictive_accuracy_deep,
                            predictive_accuracy_mc, predictive_accuracy_boots],
    'Root Mean Squared Error': [root_mean_squared_error_vi, root_mean_squared_error_ham, root_mean_squared_error_deep,
                                root_mean_squared_error_mc, root_mean_squared_error_boots],
    'Uncertainty Compared to Data': [unceratainty_compared_to_data_vi, unceratainty_compared_to_data_ham,
                                     unceratainty_compared_to_data_deep, unceratainty_compared_to_data_mc,
                                     unceratainty_compared_to_data_boots]
})
