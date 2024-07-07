import numpy as np
from scipy.stats import norm
import torch
from scipy.stats import wasserstein_distance


def accuracy(model):
    y_values_test = model.data_set.test_data.y.detach().numpy()
    predictions = model.mean_predictions
    return np.sqrt(np.sum((y_values_test - predictions) ** 2) / np.sum(y_values_test ** 2))


def predictive_capacity(model):
    y_values_test = model.data_set.test_data.y.detach().numpy()
    predictions = model.mean_predictions
    var_predictions = model.aleatoric_uncertainty + model.epistemic_uncertainty
    mpl = 0
    for i in range(len(predictions)):
        pdf_values = norm.pdf(y_values_test[i], loc=predictions[i], scale=np.sqrt(var_predictions[i]))
        mpl += pdf_values
    return mpl / len(y_values_test)


def rmsce(model):
    y_values_test = model.data_set.test_data.y.detach().numpy()
    predictions = model.mean_predictions
    predicted_variances = model.aleatoric_uncertainty + model.epistemic_uncertainty
    p_values = np.linspace(0, 1, 100)
    N_p = len(p_values)

    rmsce = 0

    for p in p_values:
        # Calculate inverse CDF (quantile function) for the predicted Gaussian distributions
        u_hat_p = norm.ppf(p, loc=predictions, scale=np.sqrt(predicted_variances))

        # Calculate the observed proportion
        observed_proportion = np.mean(y_values_test <= u_hat_p)

        # Update the RMSCE sum
        rmsce += (p - observed_proportion) ** 2

    rmsce = np.sqrt(rmsce / N_p)

    return rmsce


def nipg(model, model_ref):
    predicted_variances = model.aleatoric_uncertainty + model.epistemic_uncertainty
    predicted_variances_ref = model_ref.aleatoric_uncertainty + model_ref.epistemic_uncertainty
    dot_product_gp = np.dot(predicted_variances_ref, predicted_variances)
    dot_product_gg = np.dot(predicted_variances_ref, predicted_variances_ref)
    dot_product_pp = np.dot(predicted_variances, predicted_variances)

    # Compute NIP-G
    nip_g = dot_product_gp / np.sqrt(dot_product_gg * dot_product_pp)

    return nip_g


def get_wasserstein_distance(model_ref, model):
    x_values_test = model.data_set.test_data.x
    indices = np.argsort(x_values_test)
    indices = indices[x_values_test[indices] > -3]
    indices = indices[x_values_test[indices] < 3]

    total_wasserstein_distance = []
    aleatoric_wasserstein_distance = []
    epistemic_wasserstein_distance = []
    for i in indices:
        p = torch.distributions.Normal(model.mean_predictions[i],
                                       model.aleatoric_uncertainty[i] + model.epistemic_uncertainty[i])
        p_ref = torch.distributions.Normal(model_ref.mean_predictions[i],
                                           model_ref.aleatoric_uncertainty[i] + model_ref.epistemic_uncertainty[i])
        total_wasserstein_distance.append(wasserstein_distance(p_ref.sample((1000,)).numpy().flatten(),
                                                               p.sample((1000,)).numpy().flatten()))
        p_ale = torch.distributions.Normal(model.mean_predictions[i], model.aleatoric_uncertainty[i])
        p_ale_ref = torch.distributions.Normal(model_ref.mean_predictions[i], model_ref.aleatoric_uncertainty[i])
        aleatoric_wasserstein_distance.append(wasserstein_distance(p_ale_ref.sample((1000,)).numpy().flatten(),
                                                                   p_ale.sample((1000,)).numpy().flatten()))
        p_epi = torch.distributions.Normal(model.mean_predictions[i], model.epistemic_uncertainty[i])
        p_epi_ref = torch.distributions.Normal(model_ref.mean_predictions[i], model_ref.epistemic_uncertainty[i])
        epistemic_wasserstein_distance.append(wasserstein_distance(p_epi_ref.sample((1000,)).numpy().flatten(),
                                                                    p_epi.sample((1000,)).numpy().flatten()))
    wasserstein_dist = sum(total_wasserstein_distance) / len(total_wasserstein_distance)
    wasserstein_dist_ale = sum(aleatoric_wasserstein_distance) / len(aleatoric_wasserstein_distance)
    wasserstein_dist_epi = sum(epistemic_wasserstein_distance) / len(epistemic_wasserstein_distance)

    return wasserstein_dist, wasserstein_dist_ale, wasserstein_dist_epi


def get_kl_divergence(model_ref, model):
    x_values_test = model.data_set.test_data.x
    indices = np.argsort(x_values_test)
    indices = indices[x_values_test[indices] > -3]
    indices = indices[x_values_test[indices] < 3]

    total_kl_divergence = []
    total_kl_divergence_ale = []
    total_kl_divergence_epi = []
    for i in indices:
        p = torch.distributions.Normal(model.mean_predictions[i],
                                       np.abs(model.aleatoric_uncertainty[i] + model.epistemic_uncertainty[i]))
        p_ref = torch.distributions.Normal(model_ref.mean_predictions[i],
                                           model_ref.aleatoric_uncertainty[i] + model_ref.epistemic_uncertainty[i])
        total_kl_divergence.append(torch.distributions.kl.kl_divergence(p, p_ref).item())
        p_ale = torch.distributions.Normal(model.mean_predictions[i], np.abs(model.aleatoric_uncertainty[i]))
        p_ale_ref = torch.distributions.Normal(model_ref.mean_predictions[i], np.abs(model_ref.aleatoric_uncertainty[i]))
        total_kl_divergence_ale.append(torch.distributions.kl.kl_divergence(p_ale, p_ale_ref).item())
        p_epi = torch.distributions.Normal(model.mean_predictions[i], np.abs(model.epistemic_uncertainty[i]))
        p_epi_ref = torch.distributions.Normal(model_ref.mean_predictions[i], np.abs(model_ref.epistemic_uncertainty[i]))
        total_kl_divergence_epi.append(torch.distributions.kl.kl_divergence(p_epi, p_epi_ref).item())
    kl_div = sum(total_kl_divergence) / len(total_kl_divergence)
    kl_div_ale = sum(total_kl_divergence_ale) / len(total_kl_divergence_ale)
    kl_div_epi = sum(total_kl_divergence_epi) / len(total_kl_divergence_epi)
    return kl_div, kl_div_ale, kl_div_epi

def get_entropy_measures(model):
    """ For classification models """
    x_values_test = model.data_set.test_data.x
    indices_in_data = np.argsort(x_values_test)
    indices_in_data = indices_in_data[x_values_test[indices_in_data] > 0]
    indices_in_data = indices_in_data[x_values_test[indices_in_data] < 0.75]

    indices_out_data = np.argsort(x_values_test)
    indices_out_data = indices_out_data[x_values_test[indices_out_data] >= 0.75]

    aleatoric_entropy_in = model.aleatoric_entropy[indices_in_data].mean().item()
    aleatoric_entropy_out = model.aleatoric_entropy[indices_out_data].mean().item()
    aleatoric_entropy_all = model.aleatoric_entropy.mean().item()

    epistemic_uncertainty_in = model.epistemic_entropy[indices_in_data].mean().item()
    epistemic_uncertainty_out = model.epistemic_entropy[indices_out_data].mean().item()
    epistemic_uncertainty_all = model.epistemic_entropy.mean().item()

    return {
        'aleatoric_entropy_in': aleatoric_entropy_in,
        'aleatoric_entropy_out': aleatoric_entropy_out,
        'aleatoric_entropy_all': aleatoric_entropy_all,
        'epistemic_uncertainty_in': epistemic_uncertainty_in,
        'epistemic_uncertainty_out': epistemic_uncertainty_out,
        'epistemic_uncertainty_all': epistemic_uncertainty_all
    }