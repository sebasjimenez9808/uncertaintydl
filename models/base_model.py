import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class RegressionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, n_hidden: int = 1,
                 seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        #        x = torch.relu(self.fc3(x))
        pred = self.fc_mu(x)
        return pred

    def get_accuracy(self, y_values, predictions, stacked: bool = False):
        if not stacked:
            predictions = torch.stack(predictions, dim=0)
        # prediction has dimensions (n_models, n_samples, n_classes * 2)
        num_classes = int(predictions.size(2) / 2)  # there are two outputs per class: mean and variance

        mean_predictions = np.array(predictions[:, :, : num_classes])
        var_predictions = np.array(predictions[:, :, num_classes:])
        means = np.mean(mean_predictions, axis=0)
        vars = np.mean(var_predictions, axis=0)

        mean_tensor = torch.tensor(means, dtype=torch.float32)
        log_var_tensor = torch.tensor(vars, dtype=torch.float32)

        # Number of samples to draw
        num_samples_to_draw = 1000
        n_samples = mean_predictions.shape[1]

        # Step 1: Sample from the normal distribution
        # Here we need to sample for each mean and variance
        epsilon = torch.randn(num_samples_to_draw, n_samples,
                              2)  # Shape (num_samples_to_draw, n_samples_from_data, categories)
        std_tensor = torch.exp(log_var_tensor / 2)  # Standard deviation tensor (n_samples, categories)
        var_tensor = torch.exp(log_var_tensor)  # Variance tensor (n_samples, categories)

        # Broadcast mean and std to match epsilon shape for element-wise operation
        mean_broadcasted = mean_tensor.unsqueeze(0).expand(num_samples_to_draw, -1, -1)
        std_broadcasted = std_tensor.unsqueeze(0).expand(num_samples_to_draw, -1, -1)

        # Sample from the normal distribution
        sampled_tensor = mean_broadcasted + std_broadcasted * epsilon  # Shape (num_samples_to_draw, n_samples, categories)

        # Step 2: Apply softmax to obtain probabilities
        # Apply softmax along the last dimension
        probabilities_tensor = F.softmax(sampled_tensor, dim=-1)

        probabilities_mean = torch.mean(probabilities_tensor, dim=0)
        variance_mean = torch.var(probabilities_tensor, dim=0)

        self.mean_predictions = probabilities_mean[:, 1]
        self.epistemic_uncertainty = variance_mean[:, 1]
        self.aleatoric_uncertainty = var_tensor[:, 1]

        # use probabilities_mean to calculate entropy
        entropy = -torch.sum(probabilities_mean * torch.log(probabilities_mean), dim=-1)
        self.aleatoric_entropy = entropy

        # make a tensor of dim (500, 1) with the index of the greatest value for each sample
        predictions_cat = torch.argmax(probabilities_mean, dim=-1)

        true_labels = y_values.squeeze().numpy()
        predictions_labels = predictions_cat.numpy()

        accuracy = np.mean(true_labels == predictions_labels)
        return accuracy

    def get_information_theoretical_decomposition(self, predictions, stacked: bool = False):
        if not stacked:
            predictions = torch.stack(predictions, dim=0)
        prediction_class_zero = (1 - predictions)

        self.aleatoric_entropy = -torch.mean(predictions * torch.log(predictions) +
                                             prediction_class_zero * torch.log(prediction_class_zero), dim=0)
        predictions_mean = torch.mean(predictions, dim=0)
        predictions_mean_class_zero = torch.mean(prediction_class_zero, dim=0)

        self.total_entropy = -(predictions_mean * torch.log(predictions_mean) +
                               predictions_mean_class_zero * torch.log(predictions_mean_class_zero))

        self.epistemic_entropy = self.total_entropy - self.aleatoric_entropy

        self.predictions = predictions
        self.mean_predictions = predictions_mean
