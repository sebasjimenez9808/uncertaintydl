bootstrap_configuration = {
    "name": "Bootstrap",
    "method": "grid",
    "parameters": {
        "lr": {"values": [0.1, 0.01, 0.001]},
        "epochs": {"values": [50, 100, 200]},
        "models": {"values": [25, 50, 100]}
    },
    "metric": {"goal": "minimize", "name": "mse"},
}

mcdropout_configuration = {
    "name": "MC Dropout",
    "method": "grid",
    "parameters": {
        "lr": {"values": [0.1, 0.01, 0.001]},
        "epochs": {"values": [50, 100, 200]},
        "dropout_p": {"values": [0.1, 0.3, 0.5]}
    },
    "metric": {"goal": "minimize", "name": "mse"},
}

deep_ensemble_configuration = {
    "name": "Deep Ensemble",
    "method": "grid",
    "parameters": {
        "lr": {"values": [0.1, 0.01, 0.001]},
        "epochs": {"values": [50, 100, 200]},
        "n_models": {"values": [25, 50, 100]}
    },
    "metric": {"goal": "minimize", "name": "mse"},
}

hamiltonian_configuration = {
    "name": "Hamiltonian",
    "method": "grid",
    "parameters": {
        "step_size": {"values": [0.0005, 0.005, 0.01]},
        "num_steps_per_sample": {"values": [30, 40, 50]},
        "burn": {"values": [10, 15, 20]},
        "tau": {"values": [500, 1000, 2000]}
    },
    "metric": {"goal": "minimize", "name": "mse"},
}

laplace_configuration = {
    "name": "Laplace",
    "method": "grid",
    "parameters": {
        "lr": {"values": [0.03, 0.01, 0.001]},
        "epochs": {"values": [50, 100, 200]}
    },
    "metric": {"goal": "minimize", "name": "mse"},
}

variation_configuration = {
    "name": "Variational",
    "method": "grid",
    "parameters": {
        "lr": {"values": [0.1, 0.01, 0.001]},
        "epochs": {"values": [50, 100, 200]},
        "kl_weight": {"values": [0.3, 0.5, 0.7]}
    },
    "metric": {"goal": "minimize", "name": "mse"},
}

def get_config_file(model: str):
    if model == "ensemble":
        return bootstrap_configuration
    if model == "mcdropout":
        return mcdropout_configuration
    if model == "deep_ensemble":
        return deep_ensemble_configuration
    if model == "hamiltonian":
        return hamiltonian_configuration
    if model == "laplace":
        return laplace_configuration
    if model == "variational":
        return variation_configuration


def get_training_inputs(config, model: str):
    if model == "ensemble":
        return {
            "n_epochs": config.epochs,
            "lr": config.lr,
            "n_models": config.models
        }
    if model == "mcdropout":
        return {
            "n_epochs": config.epochs,
            "lr": config.lr,
            "dropout_p": config.dropout_p
        }
    if model == "deep_ensemble":
        return {
            "n_epochs": config.epochs,
            "lr": config.lr,
            "n_models": config.n_models
        }
    if model == "hamiltonian":
        return {
            "step_size": config.step_size,
            "num_steps_per_sample": config.num_steps_per_sample,
            "burn": config.burn,
            "tau": config.tau
        }
    if model == "laplace":
        return {
            "n_epochs": config.epochs,
            "lr": config.lr
        }
    if model == "variational":
        return {
            "n_epochs": config.epochs,
            "lr": config.lr,
            "kl_weight": config.kl_weight
        }
    return {}
