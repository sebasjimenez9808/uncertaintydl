import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
# %%
probabilities = np.array([0.2, 0.5, 0.3])
logit_means = np.log(probabilities / (1 - probabilities))
labels = np.array([0, 1, 0])
n_samples = 100
cases = [
    np.array([0.1, 0.1, 0.1]),
    np.array([0.1, 0.2, 0.3]),
    np.array([0.1, 1, 3])
]
std_factors = [0.1, 0.5, 1, 3, 5, 10, 15]
loss_results = {}
for std_case in cases:
    loss_list = []
    for i in std_factors:
        logit_std_modified = std_case * i
        perturbed_logit = logit_means + logit_std_modified * np.random.randn(n_samples, 3)
        loss = 0
        for i in range(n_samples):
            log_sum_exp = np.log(np.sum(np.exp(perturbed_logit[i,])))
            true_class_logit = perturbed_logit[i, labels == 1][0]
            exp_subtract = np.exp(true_class_logit - log_sum_exp)
            loss += exp_subtract
        mean_loss = loss / n_samples
        loss_list.append(mean_loss)
    loss_results[str(std_case)] = loss_list
#%%
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

# Plot the lines
for i, key in enumerate(loss_results):
    ax.plot(std_factors, loss_results[key], label=rf'$\sigma_{i + 1}=[{cases[i][0]}, {cases[i][1]}, {cases[i][2]}]\times z$')

# Set labels and title using LaTeX
ax.set_xlabel(r'Logit Standard Deviation Factor z', fontsize=32)
ax.set_ylabel(r'Sampling Softmax Loss', fontsize=32)
ax.set_title(r'Varying Logit Standard Deviation - Fixed Logit Means', fontsize=25)

# Customize appearance
ax.grid(False)  # Add grid lines
ax.legend(loc='upper right', fontsize=18)  # Add legend
ax.tick_params(axis='both', which='major', labelsize=22)

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot
fig.savefig("plots/simulation/sampling_softmax_fix_mean.png")
# %%
probabilities = np.array([0.2, 0.35, 0.45])
logit_means = np.log(probabilities / (1 - probabilities))
labels = np.array([0, 1, 0])
n_samples = 100
cases = [
    np.array([0.1, 0.1, 0.1]),
    np.array([0.1, 0.2, 0.3]),
    np.array([0.1, 1, 3])
]
std_factors = [0.1, 0.5, 1, 3, 5, 10, 15]
loss_results = {}
for std_case in cases:
    loss_list = []
    for i in std_factors:
        logit_std_modified = std_case * i
        perturbed_logit = logit_means + logit_std_modified * np.random.randn(n_samples, 3)
        loss = 0
        for i in range(n_samples):
            log_sum_exp = np.log(np.sum(np.exp(perturbed_logit[i,])))
            true_class_logit = perturbed_logit[i, labels == 1][0]
            exp_subtract = np.exp(true_class_logit - log_sum_exp)
            loss += exp_subtract
        mean_loss = loss / n_samples
        loss_list.append(mean_loss)
    loss_results[str(std_case)] = loss_list
# %%
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

# Plot the lines
for i, key in enumerate(loss_results):
    ax.plot(std_factors, loss_results[key], label=rf'$\sigma_{i + 1}=[{cases[i][0]}, {cases[i][1]}, {cases[i][2]}]\times z$')

# Set labels and title using LaTeX
ax.set_xlabel(r'Logit Standard Deviation Factor z', fontsize=32)
ax.set_ylabel(r'Sampling Softmax Loss', fontsize=32)
ax.set_title(r'Varying Logit Standard Deviation - Fixed Logit Means', fontsize=25)

# Customize appearance
ax.grid(False)  # Add grid lines
ax.legend(loc='upper right', fontsize=18)  # Add legend
ax.tick_params(axis='both', which='major', labelsize=22)

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot
fig.savefig("plots/simulation/sampling_softmax_fix_mean_wrong_label.png")

# %%
labels = np.array([0, 0, 1])
n_samples = 100
cases = [
    np.array([0.2, 0.35, 0.45]),
    np.array([0.2, 0.6, 0.2]),
    np.array([0.2, 0.25, 0.55])
]
std_factors = [0.1, 0.5, 1, 3, 5, 10, 15]
logit_std = np.array([0.1, 0.1, 0.1])
loss_results = {}
for prob_case in cases:
    logit_means = np.log(prob_case / (1 - prob_case))
    loss_list = []
    for i in std_factors:
        logit_std_modified = logit_std * i
        perturbed_logit = logit_means + logit_std_modified * np.random.randn(n_samples, 3)
        loss = 0
        for i in range(n_samples):
            log_sum_exp = np.log(np.sum(np.exp(perturbed_logit[i,])))
            true_class_logit = perturbed_logit[i, labels == 1][0]
            exp_subtract = np.exp(true_class_logit - log_sum_exp)
            loss += exp_subtract
        mean_loss = loss / n_samples
        loss_list.append(mean_loss)
    loss_results[f'[{round(logit_means[0], 2)}, {round(logit_means[1], 2)}, {round(logit_means[2], 2)}]'] = loss_list
# %%
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

# Plot the lines
for i, key in enumerate(loss_results):
    ax.plot(std_factors, loss_results[key], label=rf'$\mu_{i + 1}={key}$')

# Set labels and title using LaTeX
ax.set_xlabel(r'Logit Standard Deviation Factor z', fontsize=32)
ax.set_ylabel(r'Sampling Softmax Loss', fontsize=32)
ax.set_title(r'Varying Logit Means - Fixed Logit Standard Deviation', fontsize=25)

# Customize appearance
ax.grid(False)  # Add grid lines
ax.legend(loc='upper right', fontsize=18)  # Add legend
ax.tick_params(axis='both', which='major', labelsize=22)

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot
fig.savefig("plots/simulation/sampling_softmax_fix_std.png")