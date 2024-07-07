from utils.data_generation import target_function
from utils.losses import negative_log_likelihood

base_arguments = {
    "input_dim": 1,
    "hidden_dim": 40,
    "output_dim": 1,
    "reg_fct": target_function,
    "n_hidden": 4,
    "n_samples": 200,
    "test_n_samples": 500,
    "wandb_active": False
}

model_arguments = {
    "ensemble": {
        **base_arguments,
        "bootstrap_size": 0.6,
        "loss_fct": negative_log_likelihood,
        "batch_size": 32
    },
    "mcdropout": {
        **base_arguments,
        "loss_fct": negative_log_likelihood,
        "batch_size": 32
    },
    'deep_ensemble': {
        **base_arguments,
        "loss_fct": negative_log_likelihood,
        "batch_size": 32
    },
    'hamiltonian': {
        **base_arguments,
        "num_samples": 1000,
    },
    'laplace': {
        **base_arguments,
        "loss_fct": negative_log_likelihood,
        "batch_size": 32
    },
    'variational': {
        **base_arguments,
        "loss_fct": None,
        "batch_size": 32
    }
}


def get_model_arguments(model_type: str):
    return model_arguments[model_type]
