import sys
import os
conf_path = os.getcwd()

from utilities.data_generation import target_function_sine
from utilities.losses import classification_error_information, classification_error_information_vector

model_params_template_information = {
    'input_dim': 1,
    'hidden_dim': 60,
    'output_dim': 1,
    'reg_fct': target_function_sine,
    'n_hidden': 1,
    'loss_fct': classification_error_information,
    'loss_fct_vector': classification_error_information_vector,
    'lr': 0.00035,
    'batch_size': 30,
    'n_samples': 1500,
    'test_n_samples': 500,
    'n_epochs': 1000,
    'problem': 'classification',
    'train_interval': (0, 0.5),
    'test_interval': (0, 1),
    'seed': 42,
    'heteroscedastic': False,
    'wandb_active': False,
    'add_sigmoid': True,
    'samples_prediction': 2000,  # samples to take in bayesian models
    'n_models': 100,  # n models for ensemble
    'dropout_p': 0.3,  # for mc dropout
    'step_size': 0.0005,  # for hamiltonian
    'num_steps_per_sample': 700,
    'burn': 100,
    'tau': 1,
    'bootstrap_size': 0.6,  # for ensemble
}