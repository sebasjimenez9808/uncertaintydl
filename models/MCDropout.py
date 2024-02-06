import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.data_generation import generate_synthetic_data

torch.manual_seed(42)
np.random.seed(42)

x_train, y_train, y_bernoulli_train = generate_synthetic_data()
# make this arrays tensors

x_train = torch.Tensor(x_train).reshape(-1, 1)
y_train = torch.Tensor(y_train)


class MCDropoutNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_p)  # Apply dropout to all layers

    def forward(self, x, i=1):
        torch.manual_seed(i + 42)
        x = torch.relu(self.dropout(self.fc1(x)))
        x = torch.relu(self.dropout(self.fc2(x)))
        x = torch.relu(self.dropout(self.fc3(x))) # No activation for final layer
        return x


model = MCDropoutNet(1, 40, 1)
# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adjust learning rate as needed

for epoch in range(100):
    #model.train()
    optimizer.zero_grad()
    predictions = model(x_train, i=epoch+1)  # Pass the epoch number as the seed for dropout
    loss = F.mse_loss(predictions, y_train.unsqueeze(1))  # Calculate loss on mean prediction
    loss.backward()
    optimizer.step()

def generate_predictions(model, x, num_samples=100):
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            #model.eval()
            prediction = model(x, _+101, )
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


x_plot, mean_prediction, std_prediction = zip(*sorted(zip(x_plot, mean_prediction, std_prediction)))
x_plot = np.array(x_plot)
mean_prediction = np.array(mean_prediction)
std_prediction = np.array(std_prediction)

plt.plot(x_plot, mean_prediction, label='Mean Prediction',
         color='blue')
plt.fill_between(x_plot, mean_prediction - std_prediction, mean_prediction + std_prediction, alpha=0.2,
                 label='Uncertainty')
plt.scatter(x_train.numpy(), y_train.numpy(), label='True Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('MC Dropout')
plt.legend()
plt.show()

plt.close()
