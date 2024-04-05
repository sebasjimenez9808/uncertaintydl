import torch.nn as nn
import matplotlib.pyplot as plt
from models.Bootstrap import BootstrapEnsemble
from models.MCDropout import MCDropoutNet
from models.deep_ensemble import DeepEnsemble
from models.variational_inference import BayesianNet
from utils.data_generation import target_function
from utils.losses import negative_log_likelihood


# %%

def generate_model(model_type, input_dim, hidden_dim, output_dim, reg_fct, n_hidden, **kwargs):
    if model_type == "ensemble":
        return BootstrapEnsemble(input_dim, hidden_dim, output_dim, reg_fct, n_hidden, **kwargs)
    elif model_type == "mcdropout":
        return MCDropoutNet(input_dim, hidden_dim, output_dim, reg_fct, n_hidden, **kwargs)
    elif model_type == "variational":
        return BayesianNet(input_dim, hidden_dim, output_dim, reg_fct, n_hidden, **kwargs)
    else:
        raise ValueError("Model type not recognized")


def train_model(model, loss_fct, n_epochs, lr, batch_size):
    if isinstance(model, BootstrapEnsemble):
        model.train_model(loss_fct, n_epochs, lr, batch_size)
    elif isinstance(model, MCDropoutNet):
        model.train_model(loss_fct, n_epochs, lr, batch_size)
    elif isinstance(model, BayesianNet):
        model.train_model(n_epochs, lr, batch_size)
    else:
        raise ValueError("Model type not recognized")


def plot_model_loss(model):
    if isinstance(model, BootstrapEnsemble):
        loss = model.model_loss[90]
    elif isinstance(model, MCDropoutNet):
        loss = model.model_loss
    elif isinstance(model, BayesianNet):
        loss = model.model_loss
    else:
        raise ValueError("Model type not recognized")
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter([i for i in range(100)], loss, s=2, alpha=0.7)
    plt.xlabel("Independent Variable (x)")
    plt.ylabel("Dependent Variable (y)")
    plt.title("Synthetic Data with Bimodal Distribution")
    # write code to save figure
    plt.savefig('data.png')
    plt.show()


# %%

# ensemble
training_model = generate_model("ensemble", input_dim=1, hidden_dim=40, output_dim=1,
                                reg_fct=target_function, n_hidden=4, n_models=100, bootstrap_size=0.6)
train_model(training_model, loss_fct=nn.functional.mse_loss, n_epochs=100, lr=0.01, batch_size=40)
plot_model_loss(training_model)

# dropout
training_model = generate_model("mcdropout", input_dim=1, hidden_dim=40, output_dim=1,
                                reg_fct=target_function, n_hidden=4, dropout_p=0.3)
train_model(training_model, loss_fct=nn.functional.mse_loss, n_epochs=100, lr=0.01, batch_size=40)
plot_model_loss(training_model)

# variational
training_model = generate_model("variational", input_dim=1, hidden_dim=40, output_dim=1,
                                reg_fct=target_function, n_hidden=4, kl_weight=0.7)
train_model(training_model, loss_fct=None, n_epochs=100, lr=0.01, batch_size=40)
plot_model_loss(training_model)

model = DeepEnsemble(input_dim=1, hidden_dim=40, output_dim=1, reg_fct=target_function, n_hidden=1, n_models=10)
model.train_model(negative_log_likelihood, n_epochs=500, lr=0.01, batch_size=40)
mean, var = model(model.data_set.x_test)

mean, var = model.models[0](torch.from_numpy(model.data_set.x_test).float().view(-1, 1))

plt.figure(figsize=(10, 6))
plt.scatter([i for i in range(500)], model.model_loss[1], s=2, alpha=0.7)
plt.xlabel("Independent Variable (x)")
plt.ylabel("Dependent Variable (y)")
plt.title("Synthetic Data with Bimodal Distribution")
# write code to save figure
plt.savefig('data.png')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(model.data_set.x_test, model.data_set.y_test, s=2, alpha=0.6)
plt.scatter(model.data_set.x_test, mean.detach().numpy(), s=2, alpha=0.9)
plt.xlabel("Independent Variable (x)")
plt.ylabel("Dependent Variable (y)")
plt.title("Synthetic Data with Bimodal Distribution")
# write code to save figure
plt.savefig('data.png')
plt.show()
plt.close()
