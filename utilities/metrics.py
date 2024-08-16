import numpy as np
from scipy.stats import norm
import torch
from scipy.stats import wasserstein_distance


def calculate_accuracy(model, indices=None, model_ref=None):
    y_values_test = model['test_data']['y'][indices]
    predictions = model['test_data']['predictions'][indices]
    return {"all": np.sqrt(np.sum((y_values_test - predictions) ** 2) / np.sum(y_values_test ** 2))}


def calculate_predictive_capacity(model, indices=None, model_ref=None):
    y_values_test = model['test_data']['y'][indices]
    predictions = model['test_data']['predictions'][indices]
    var_predictions = model['test_data']['aleatoric_uncertainty'][indices] + \
                      model['test_data']['epistemic_uncertainty'][indices]
    mpl = 0
    for i in range(len(predictions)):
        pdf_values = norm.pdf(y_values_test[i], loc=predictions[i], scale=np.sqrt(var_predictions[i]))
        mpl += pdf_values
    capacity = mpl / len(y_values_test)

    predictions = model['test_data']['predictions'][indices]
    var_predictions = model['test_data']['aleatoric_uncertainty'][indices]
    mpl_ale = 0
    for i in range(len(predictions)):
        pdf_values = norm.pdf(y_values_test[i], loc=predictions[i], scale=np.sqrt(var_predictions[i]))
        mpl_ale += pdf_values
    capacity_ale = mpl_ale / len(y_values_test)

    predictions = model['test_data']['predictions'][indices]
    var_predictions = model['test_data']['epistemic_uncertainty'][indices]
    mpl_epi = 0
    for i in range(len(predictions)):
        pdf_values = norm.pdf(y_values_test[i], loc=predictions[i], scale=np.sqrt(var_predictions[i]))
        mpl_epi += pdf_values
    capacity_epi = mpl_epi / len(y_values_test)

    return {
        'all': capacity,
        'aleatoric': capacity_ale,
        'epistemic': capacity_epi
    }


def calculate_rmsce(model, model_ref=None, indices=None):
    y_values_test = model['test_data']['y'][indices]
    predictions = model['test_data']['predictions'][indices]
    predicted_variances = model['test_data']['aleatoric_uncertainty'][indices] + \
                          model['test_data']['epistemic_uncertainty'][indices]
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

    predicted_variances = model['test_data']['aleatoric_uncertainty'][indices]
    p_values = np.linspace(0, 1, 100)
    N_p = len(p_values)

    rmsce_ale = 0

    for p in p_values:
        # Calculate inverse CDF (quantile function) for the predicted Gaussian distributions
        u_hat_p = norm.ppf(p, loc=predictions, scale=np.sqrt(predicted_variances))

        # Calculate the observed proportion
        observed_proportion = np.mean(y_values_test <= u_hat_p)

        # Update the RMSCE sum
        rmsce_ale += (p - observed_proportion) ** 2

    rmsce_ale = np.sqrt(rmsce_ale / N_p)

    predicted_variances = model['test_data']['epistemic_uncertainty'][indices]
    p_values = np.linspace(0, 1, 100)
    N_p = len(p_values)

    rmsce_epis = 0

    for p in p_values:
        # Calculate inverse CDF (quantile function) for the predicted Gaussian distributions
        u_hat_p = norm.ppf(p, loc=predictions, scale=np.sqrt(predicted_variances))

        # Calculate the observed proportion
        observed_proportion = np.mean(y_values_test <= u_hat_p)

        # Update the RMSCE sum
        rmsce_epis += (p - observed_proportion) ** 2

    rmsce_epis = np.sqrt(rmsce_epis / N_p)

    return {
        'all': rmsce,
        'aleatoric': rmsce_ale,
        'epistemic': rmsce_epis
    }


def calculate_nipg(model, model_ref, indices):
    predicted_variances = model['test_data']['aleatoric_uncertainty'][indices] + \
                          model['test_data']['epistemic_uncertainty'][indices]
    predicted_variances_ref = model_ref['test_data']['aleatoric_uncertainty'][indices] + model_ref['test_data'][
        'epistemic_uncertainty'][indices]
    dot_product_gp = np.dot(predicted_variances_ref, predicted_variances)
    dot_product_gg = np.dot(predicted_variances_ref, predicted_variances_ref)
    dot_product_pp = np.dot(predicted_variances, predicted_variances)

    # Compute NIP-G
    nip_g = dot_product_gp / np.sqrt(dot_product_gg * dot_product_pp)

    predicted_variances = model['test_data']['aleatoric_uncertainty'][indices]
    predicted_variances_ref = model_ref['test_data']['aleatoric_uncertainty'][indices]
    dot_product_gp = np.dot(predicted_variances_ref, predicted_variances)
    dot_product_gg = np.dot(predicted_variances_ref, predicted_variances_ref)
    dot_product_pp = np.dot(predicted_variances, predicted_variances)

    # Compute NIP-G
    nip_g_ale = dot_product_gp / np.sqrt(dot_product_gg * dot_product_pp)

    predicted_variances = model['test_data']['epistemic_uncertainty'][indices]
    predicted_variances_ref = model_ref['test_data']['epistemic_uncertainty'][indices]
    dot_product_gp = np.dot(predicted_variances_ref, predicted_variances)
    dot_product_gg = np.dot(predicted_variances_ref, predicted_variances_ref)
    dot_product_pp = np.dot(predicted_variances, predicted_variances)

    # Compute NIP-G
    nip_g_epis = dot_product_gp / np.sqrt(dot_product_gg * dot_product_pp)

    return {
        'all': nip_g,
        'aleatoric': nip_g_ale,
        'epistemic': nip_g_epis
    }


def get_wasserstein_distance(model_ref, model, indices):
    total_wasserstein_distance = []
    aleatoric_wasserstein_distance = []
    epistemic_wasserstein_distance = []
    for i in indices:
        p = torch.distributions.Normal(model['test_data']['predictions'][i],
                                       model['test_data']['aleatoric_uncertainty'][i] +
                                       model['test_data']['epistemic_uncertainty'][i])
        p_ref = torch.distributions.Normal(model_ref['test_data']['predictions'][i],
                                           model_ref['test_data']['aleatoric_uncertainty'][i] +
                                           model_ref['test_data']['epistemic_uncertainty'][i])
        total_wasserstein_distance.append(wasserstein_distance(p_ref.sample((1000,)).numpy().flatten(),
                                                               p.sample((1000,)).numpy().flatten()))
        p_ale = torch.distributions.Normal(model['test_data']['predictions'][i],
                                           model['test_data']['aleatoric_uncertainty'][i])
        p_ale_ref = torch.distributions.Normal(model_ref['test_data']['predictions'][i],
                                               model_ref['test_data']['aleatoric_uncertainty'][i])
        aleatoric_wasserstein_distance.append(wasserstein_distance(p_ale_ref.sample((1000,)).numpy().flatten(),
                                                                   p_ale.sample((1000,)).numpy().flatten()))
        p_epi = torch.distributions.Normal(model['test_data']['predictions'][i],
                                           model['test_data']['epistemic_uncertainty'][i])
        p_epi_ref = torch.distributions.Normal(model_ref['test_data']['predictions'][i],
                                               model_ref['test_data']['epistemic_uncertainty'][i])
        epistemic_wasserstein_distance.append(wasserstein_distance(p_epi_ref.sample((1000,)).numpy().flatten(),
                                                                   p_epi.sample((1000,)).numpy().flatten()))
    wasserstein_dist = sum(total_wasserstein_distance) / len(total_wasserstein_distance)
    wasserstein_dist_ale = sum(aleatoric_wasserstein_distance) / len(aleatoric_wasserstein_distance)
    wasserstein_dist_epi = sum(epistemic_wasserstein_distance) / len(epistemic_wasserstein_distance)

    return {
        'all': wasserstein_dist,
        'aleatoric': wasserstein_dist_ale,
        'epistemic': wasserstein_dist_epi
    }


def get_kl_divergence(model_ref, model, indices, return_array: bool = False):
    total_kl_divergence = []
    total_kl_divergence_ale = []
    total_kl_divergence_epi = []
    for i in indices:
        p = torch.distributions.Normal(model['test_data']['predictions'][i],
                                       np.abs(model['test_data']['aleatoric_uncertainty'][i] +
                                              model['test_data']['epistemic_uncertainty'][i]))
        p_ref = torch.distributions.Normal(model_ref['test_data']['predictions'][i],
                                           model_ref['test_data']['aleatoric_uncertainty'][i] +
                                           model_ref['test_data']['epistemic_uncertainty'][i])
        total_kl_divergence.append(torch.distributions.kl.kl_divergence(p, p_ref).item())
        p_ale = torch.distributions.Normal(model['test_data']['predictions'][i],
                                           np.abs(model['test_data']['aleatoric_uncertainty'][i]))
        p_ale_ref = torch.distributions.Normal(model_ref['test_data']['predictions'][i],
                                               np.abs(model_ref['test_data']['aleatoric_uncertainty'][i]))
        total_kl_divergence_ale.append(torch.distributions.kl.kl_divergence(p_ale, p_ale_ref).item())
        p_epi = torch.distributions.Normal(model['test_data']['predictions'][i],
                                           np.abs(model['test_data']['epistemic_uncertainty'][i]))
        p_epi_ref = torch.distributions.Normal(model_ref['test_data']['predictions'][i],
                                               np.abs(model_ref['test_data']['epistemic_uncertainty'][i]))
        total_kl_divergence_epi.append(torch.distributions.kl.kl_divergence(p_epi, p_epi_ref).item())

    total_kl_divergence = [x if not np.isnan(x) else 0 for x in total_kl_divergence]
    total_kl_divergence_ale = [x if not np.isnan(x) else 0 for x in total_kl_divergence_ale]
    total_kl_divergence_epi = [x if not np.isnan(x) else 0 for x in total_kl_divergence_epi]
    if not return_array:
        total_kl_divergence = sum(total_kl_divergence) / len(total_kl_divergence) if len(total_kl_divergence) > 0 else 0
        total_kl_divergence_ale = sum(total_kl_divergence_ale) / len(total_kl_divergence_ale) if len(total_kl_divergence_ale) > 0 else 0
        total_kl_divergence_epi = sum(total_kl_divergence_epi) / len(total_kl_divergence_epi) if len(total_kl_divergence_epi) > 0 else 0
    return {
        'all': total_kl_divergence,
        'aleatoric': total_kl_divergence_ale,
        'epistemic': total_kl_divergence_epi
    }


def get_entropy_measures(model):
    """ For classification models """
    x_values_test = model['test_data']['x']
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


def iterate_over_data(model, model_ref, metric: callable):
    training_interval = model['train_interval']
    x_values_test = model['test_data']['x']
    indices = np.argsort(x_values_test)
    indices_in_sample = indices[x_values_test[indices] > training_interval[0]]
    indices_in_sample = indices_in_sample[x_values_test[indices_in_sample] < training_interval[1]]
    indices_out_sample = indices[
        (x_values_test[indices] < training_interval[0]) | (x_values_test[indices] > training_interval[1])]
    return {
        'in_sample': metric(model=model, model_ref=model_ref, indices=indices_in_sample),
        'out_sample': metric(model=model, model_ref=model_ref, indices=indices_out_sample),
        'all': metric(model=model, model_ref=model_ref, indices=indices)
    }


def calculate_all_metrics_regression(model, model_ref):
    kl_divergence = iterate_over_data(model=model, model_ref=model_ref, metric=get_kl_divergence)
    wasserstein_distance = iterate_over_data(model=model, model_ref=model_ref, metric=get_wasserstein_distance)
    predictive_capacity = iterate_over_data(model=model, model_ref=model_ref, metric=calculate_predictive_capacity)
    predicitve_accuracy = iterate_over_data(model=model, model_ref=model_ref, metric=calculate_accuracy)
    root_mean_squared_error = iterate_over_data(model=model, model_ref=model_ref, metric=calculate_rmsce)
    uncertainty_compared_to_data = iterate_over_data(model=model, model_ref=model_ref, metric=calculate_nipg)

    return {
        'KL Divergence': kl_divergence,
        'Wasserstein-1': wasserstein_distance,
        'Predictive Capacity': predictive_capacity,
        'Accuracy': predicitve_accuracy,
        'RMSE': root_mean_squared_error,
        'NIP-G': uncertainty_compared_to_data
    }

def save_text_file(metrics, filename):
    with open(filename, 'w') as f:
        f.write(str(metrics))