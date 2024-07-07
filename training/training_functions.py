from models.Bootstrap import BootstrapEnsemble
from models.LaplaceApprox import LaplaceReg
from models.MCDropout import MCDropoutNet
from models.MCMCHamiltorch import MCMCReg
from models.deep_ensemble import DeepEnsemble
from models.variational_inference import BayesianNet

def generate_model(model_type, input_dim, hidden_dim, output_dim, reg_fct, n_hidden, n_samples,
                   test_n_samples, wandb_active=False,
                   **kwargs):
    if model_type == "ensemble":
        return BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                 reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                                 test_n_samples=test_n_samples, wandb_active=wandb_active,
                                 **kwargs)

    elif model_type == "mcdropout":
        return MCDropoutNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                            reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                            test_n_samples=test_n_samples, **kwargs)
    elif model_type == "variational":
        return BayesianNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                           reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                           test_n_samples=test_n_samples, **kwargs)
    elif model_type == "deep_ensemble":
        return DeepEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                            reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                            test_n_samples=test_n_samples, **kwargs)
    elif model_type == 'hamiltonian':
        return MCMCReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                       test_n_samples=test_n_samples)
    elif model_type == 'laplace':
        return LaplaceReg(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                          reg_fct=reg_fct, n_hidden=n_hidden, n_samples=n_samples,
                          test_n_samples=test_n_samples, )
    else:
        raise ValueError("Model type not recognized")


def train_model(model, **kwargs):
    if isinstance(model, BootstrapEnsemble):
        model.train_model(**kwargs)
    elif isinstance(model, MCDropoutNet):
        model.train_model(**kwargs)
    elif isinstance(model, BayesianNet):
        model.train_model(**kwargs)
    elif isinstance(model, DeepEnsemble):
        model.train_model(**kwargs)
    elif isinstance(model, MCMCReg):
        model.sample(**kwargs)
    elif isinstance(model, LaplaceReg):
        model.train_model(**kwargs)
    else:
        raise ValueError("Model type not recognized")


def train_and_test_model(model: str, input_dim: int, hidden_dim: int, output_dim: int,
                         reg_fct: callable, n_hidden: int, loss_fct: callable = None,
                         n_epochs: int = 100, lr: float = 0.01, batch_size: int = 32,
                         **kwargs):
    model = generate_model(model_type=model, input_dim=input_dim, hidden_dim=hidden_dim,
                           output_dim=output_dim, reg_fct=reg_fct, n_hidden=n_hidden, **kwargs)
    train_model(model, loss_fct=loss_fct, n_epochs=n_epochs, lr=lr, batch_size=batch_size)
    model.make_predictions_on_test()
    return model
