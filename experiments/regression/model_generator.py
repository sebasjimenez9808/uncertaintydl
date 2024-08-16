from models.Bootstrap import BootstrapEnsemble
from models.MCDropout import MCDropoutNet
from models.MCMCHamiltorch import MCMCReg
from models.deep_ensemble import DeepEnsemble
from models.laplace_scratch import LaplaceReg
from models.old_vi import VIModel
import time

def generate_model(model_type, input_dim, hidden_dim, output_dim, reg_fct, n_hidden, n_samples,
                   test_n_samples, wandb_active=False, heteroscedastic: bool = False, num_samples: int = 1000,
                   problem: str = 'regression', train_interval: tuple = (-3, 3), test_interval: tuple = (-5, 5),
                   loss_fct: callable = None, seed: int = 42, samples_prediction: int = 500, n_models: int = 100,
                   loss_fct_vector: callable = None, **kwargs):
    if model_type == "bootstrap":
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
                       test_interval=test_interval, loss_fct=loss_fct_vector, seed=seed,
                       **kwargs)
    elif model_type == 'laplace':
        return LaplaceReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                          reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                          test_n_samples=test_n_samples, heteroscedastic=heteroscedastic, train_interval=train_interval,
                          test_interval=test_interval, seed=seed, num_samples=samples_prediction)
    else:
        raise ValueError("Model type not recognized")


def train_model(model, **kwargs):
    model.train_model(**kwargs)


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
    time_start = time.time()
    train_model(model, loss_fct=loss_fct, n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    time_end = time.time()
    model.training_time = time_end - time_start
    model.make_predictions_on_test()

    return model
