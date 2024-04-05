import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils.data_generation import generate_synthetic_data

torch.manual_seed(42)
np.random.seed(42)

x_train, y_train, y_bernoulli_train = generate_synthetic_data()

x_train = torch.Tensor(x_train).reshape(-1, 1)
y_train = torch.Tensor(y_train)


class BootstrapNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=40, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten after pooling
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for final layer
        return x


# generate bootstraping samples from the training data
def generate_bootstrap_samples(x, y, num_samples=100):
    """Generates bootstrap samples from the training data.

    Args:
        x (torch.Tensor): The independent variable data.
        y (torch.Tensor): The dependent variable data.
        num_samples (int, optional): The number of bootstrap samples to generate. Defaults to 100.

    Returns:
        tuple: The generated bootstrap samples for x and y.
    """
    x_samples = []
    y_samples = []
    for _ in range(num_samples):
        np.random.seed(_ + 1 * 10 + 42)
        indices = np.random.choice(len(x), int(len(x) * 0.6), replace=True)  # Generate random indices with replacement
        x_samples.append(x[indices])
        y_samples.append(y[indices])
    return torch.stack(x_samples), torch.stack(y_samples)


x_bootstrap, y_bootstrap = generate_bootstrap_samples(x_train, y_train)

# iterate over the bootstrap samples and train a model on each
models = []
for i in range(x_bootstrap.shape[0]):
    torch.manual_seed(i + 42)
    model = BootstrapNet(1, 40, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model(x_bootstrap[i].unsqueeze(1)) # unsqueeze this x for convolutional nn
        loss = F.mse_loss(predictions, y_bootstrap[i].unsqueeze(1))
        loss.backward()
        optimizer.step()
    models.append(model)


# generate predictions from each model
def generate_predictions(models, x):
    predictions = []
    for model in models:
        with torch.no_grad():
            prediction = model(x.unsqueeze(1))
        predictions.append(prediction.detach().numpy())
    return np.asarray(predictions)


predictions = generate_predictions(models, x_train)

# calculate mean and standard deviation of the predictions based on x_bootstrap values
mean_prediction = np.mean(predictions, axis=0)
std_prediction = np.std(predictions, axis=0)

#mean_prediction = np.array(predictions[50])

mean_prediction = mean_prediction.reshape(-1, 1)
std_prediction = std_prediction.reshape(-1, 1)

x_plot = x_train.numpy().squeeze()  # Convert to NumPy array for plotting
mean_prediction = mean_prediction.squeeze()
std_prediction = std_prediction.squeeze()

x_plot, mean_prediction, std_prediction = zip(*sorted(zip(x_plot, mean_prediction, std_prediction)))
x_plot = np.array(x_plot)
mean_prediction = np.array(mean_prediction)
std_prediction = np.array(std_prediction)

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, s=2, alpha=0.7)
plt.plot(x_plot, mean_prediction, label='Mean Prediction', color='blue')
plt.fill_between(x_plot, mean_prediction - std_prediction, mean_prediction + std_prediction, alpha=0.2,
                 label='Uncertainty')
plt.xlabel("Independent Variable (x)")
plt.ylabel("Dependent Variable (y)")
plt.title("Bootstrap CNN")
plt.legend()
plt.show()
plt.close()
