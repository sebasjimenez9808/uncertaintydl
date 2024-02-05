import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
from models.data_generation import generate_synthetic_data
import torch

torch.manual_seed(42)
np.random.seed(42)

x_train, y_train, y_bernoulli_train = generate_synthetic_data()
# make this arrays tensors

x_train = torch.Tensor(x_train).reshape(-1, 1)
y_train = torch.Tensor(y_train)


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Define variational parameters for weight and bias
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_rho = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # Sample weight and bias from their variational distributions
        weight = Normal(self.weight_mu, torch.log(1 + torch.exp(self.weight_rho))).rsample()
        bias = Normal(self.bias_mu, torch.log(1 + torch.exp(self.bias_rho))).rsample()

        return torch.nn.functional.linear(x, weight, bias)


class SineActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class BayesianNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.fc4 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc4(x))

        return x


def loss_function(model, x, y, kl_weight=0.01):
    # Compute model prediction
    prediction = model(x)

    # Compute negative log likelihood (NLL) loss
    nll_loss = nn.functional.mse_loss(prediction, y)

    # Compute KL divergence between variational posterior and prior
    kl_loss = 0.
    for layer in model.children():
        if isinstance(layer, BayesianLinear):
            kl_loss += torch.sum(
                layer.weight_rho - torch.log(1 + torch.exp(layer.weight_rho)) +
                (layer.weight_mu ** 2) / (1 + torch.exp(layer.weight_rho)) - 0.5
            )
            kl_loss += torch.sum(
                layer.bias_rho - torch.log(1 + torch.exp(layer.bias_rho)) +
                (layer.bias_mu ** 2) / (1 + torch.exp(layer.bias_rho)) - 0.5
            )

    # Combine NLL and KL losses
    loss = nll_loss + kl_weight * kl_loss

    return loss


# ... (Set up data, optimizer, etc.)

model = BayesianNet(1, 40, 1)  # Adjust input/hidden/output dimensions as needed

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adjust learning rate as needed

for epoch in range(100):
    optimizer.zero_grad()
    loss = loss_function(model, x_train, y_train)
    loss.backward()
    optimizer.step()


def generate_predictions(model, x, num_samples=100):
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            prediction = model(x)
        predictions.append(prediction.detach().numpy())  # Store as NumPy arrays
    return np.asarray(predictions)


predictions = generate_predictions(model, x_train)

mean_prediction = np.mean(predictions, axis=0)
std_prediction = np.std(predictions, axis=0)

mean_prediction = mean_prediction.reshape(-1, 1)
std_prediction = std_prediction.reshape(-1, 1)

x_plot = x_train.numpy().squeeze()  # Convert to NumPy array for plotting
mean_prediction = mean_prediction.squeeze()
std_prediction = std_prediction.squeeze()

# order x_plot mean_prediction and std_prediction based on x_plot
x_plot, mean_prediction, std_prediction = zip(*sorted(zip(x_plot, mean_prediction, std_prediction)))
x_plot = np.array(x_plot)
mean_prediction = np.array(mean_prediction)
std_prediction = np.array(std_prediction)

plt.plot(x_plot, mean_prediction, label='Mean Prediction',
         color='blue')
#plt.fill_between(x_plot, mean_prediction - std_prediction, mean_prediction + std_prediction, alpha=0.2,
#                 label='Uncertainty')
plt.scatter(x_train.numpy(), y_train.numpy(), label='True Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bayesian Neural Network Predictions with Uncertainty')
plt.legend()
plt.show()

plt.close()
