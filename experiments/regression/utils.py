import json
import numpy as np

from utilities.data_generation import cubic_function
from utilities.losses import negative_log_likelihood, negative_log_likelihood_vector


def save_json_model(config: dict, model, model_type: str,
                    problem: str = 'regression'):
    x_values_test = model.data_set.test_data.x.numpy()
    x_values_train = model.data_set.train_data.x.numpy()
    y_values_test = model.data_set.test_data.y.numpy()
    y_values_train = model.data_set.train_data.y.numpy()
    if problem == 'regression':
        nose_train = model.data_set.train_data.noise.numpy()
        nose_test = model.data_set.test_data.noise.numpy()
        aleatoric_unceratinty = model.aleatoric_uncertainty
        epistemic_uncertainty = model.epistemic_uncertainty
    else:
        nose_train = model.data_set.train_data.p.copy()
        nose_test = model.data_set.train_data.p.copy()
        aleatoric_unceratinty = model.aleatoric_entropy
        epistemic_uncertainty = model.epistemic_entropy

    predictions_mean = model.mean_predictions


    indices = np.argsort(x_values_test)
    x_values_test = x_values_test[indices]
    y_values_test = y_values_test[indices]
    nose_test = nose_test[indices]
    predictions_mean = predictions_mean[indices]
    aleatoric_unceratinty = aleatoric_unceratinty[indices]
    epistemic_uncertainty = epistemic_uncertainty[indices]

    indices_train = np.argsort(x_values_train)
    x_values_train = x_values_train[indices_train]
    y_values_train = y_values_train[indices_train]
    nose_train = nose_train[indices_train]

    json_data = {
        'train_data': {
            'x': x_values_train.tolist(),
            'y': y_values_train.tolist(),
            'noise': nose_train.tolist()
        },
        'test_data': {
            'x': x_values_test.tolist(),
            'y': y_values_test.tolist(),
            'noise': nose_test.tolist(),
            'predictions': predictions_mean.tolist(),
            'aleatoric_uncertainty': aleatoric_unceratinty.tolist(),
            'epistemic_uncertainty': epistemic_uncertainty.tolist()
        },
        'training_samples': config['n_samples'],
        'epochs': config['n_epochs'],
        'training_time': model.training_time,
        'model': model_type,
        'lr': config['lr'],
        'batch_size': config['batch_size'],
        'hidden_dim': config['hidden_dim'],
        'n_hidden': config['n_hidden'],
        'seed': config['seed'],
        'hetereoscedastic': config['heteroscedastic'],
        'bayesian_samples': config['samples_prediction'],
        'ensemble_models': config['n_models'],
        'train_interval': config['train_interval'],
        'test_interval': config['test_interval']
    }
    filename = get_model_filename(config=config, model_type=model_type,
                                  problem=problem)
    save_json(filename=filename, data=json_data)


model_params_template = {
    'input_dim': 1,
    'hidden_dim': 40,
    'output_dim': 2,
    'reg_fct': cubic_function,
    'n_hidden': 1,
    'loss_fct': negative_log_likelihood,
    'loss_fct_vector': negative_log_likelihood_vector,
    'lr': 0.01,
    'batch_size': 35,
    'n_samples': 100,
    'test_n_samples': 500,
    'n_epochs': 400,
    'problem': 'regression',
    'train_interval': (-4, 4),
    'test_interval': (-5, 5),
    'seed': 42,
    'heteroscedastic': True,
    'wandb_active': False,
    'samples_prediction': 1000,  # samples to take in bayesian models
    'n_models': 50  # n models for ensemble
}


def get_model_filename(config, model_type: str, folder: str = 'results', file_format: str = 'json',
                       extra_name_part: str = '', problem: str = 'regression'):
    """
    ts: traning_samples
    ep: epochs
    hd: hidden_dim
    nh: n_hidden
    nm: n_models
    lr: learning_rate
    sp: samples_prediction
    """
    if model_type in ['bootstrap', 'mcdropout', 'deep', 'reference']:
        n_models_name = f'nm_{config["n_models"]}'
    else:
        n_models_name = f'sp_{config["samples_prediction"]}'
    return f'{folder}/{problem}/{model_type}_het_{str(config["heteroscedastic"])}_ts_{config["n_samples"]}_ep_{config["n_epochs"]}_hd_{config["hidden_dim"]}_nh_{config["n_hidden"]}_{n_models_name}_lr_{config["lr"]}{extra_name_part}.{file_format}'


def read_json_file(filename: str):
    """Reads a JSON file and returns the data as a Python object.

    Args:
        filename: The name of the JSON file to read.

    Returns:
        The data from the JSON file as a Python object (e.g., dictionary or list).
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            data = transform_json_lists_to_array(data)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{filename}'.")
        return None


def save_json(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def transform_json_lists_to_array(data):
    for subset in ['train_data', 'test_data']:
        for key, vals in data[subset].items():
            data[subset][key] = np.array(vals)

    return data


def get_models_name_from_model_type(models_type):
    models_name = {
        'variational': 'VI',
        'laplace': 'LA',
        'hamiltonian': 'HMC',
        'deep': 'Deep Ensemble',
        'mcdropout': 'MC Dropout',
        'bootstrap': 'Bootstrap',
        'reference': 'Reference Distribution'
    }
    return [models_name[m] for m in models_type]
