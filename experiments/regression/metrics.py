import copy
from experiments.regression.utils import (model_params_template, get_model_filename,
                                          read_json_file, get_models_name_from_model_type)
from utilities.metrics import *

params = copy.deepcopy(model_params_template)


# %%
def get_all_metrics(params: dict):
    metrics_results = {}
    models = ['variational', 'laplace', 'hamiltonian', 'deep', 'mcdropout', 'bootstrap']
    models_names = get_models_name_from_model_type(models)
    model_ref = read_json_file(get_model_filename(model_type='reference', config=params))
    for model_type, model_name in zip(models, models_names):
        model = read_json_file(get_model_filename(model_type=model_type, config=params))
        metrics_results[model_name] = calculate_all_metrics_regression(model=model, model_ref=model_ref)
    return metrics_results


def get_all_metrics_from_models_files(models_files: list, reference_file: str, models_names: list):
    metrics_results = {}
    models = [read_json_file(f'results/regression/{model_file}') for model_file in models_files]
    model_ref = read_json_file(f'results/regression/{reference_file}')
    for model, model_name in zip(models, models_names):
        metrics_results[model_name] = calculate_all_metrics_regression(model=model, model_ref=model_ref)
    return metrics_results


# %&

def get_latex_table_from_metrics(metrics: dict, sample_type: str = 'all', measure_type: str = 'all',
                                 metrics_subset: list = None, models_subset: list = None):
    metrics_subset = metrics_subset or ['KL Divergence', 'Wasserstein-1', 'NIP-G', 'Predictive Capacity', 'Accuracy',
                                        'RMSE']
    columns = 'c' * len(metrics_subset) * (1 if isinstance(sample_type, str) else len(sample_type))
    models_subset = models_subset or ['VI', 'LA', 'HMC', 'Deep Ensemble', 'MC Dropout', 'Bootstrap']
    latex_code = "\\begin{table} \n \\centering \n \\begin{tabular}{l|" + f"{columns}" + "} \\toprule \n"
    if isinstance(sample_type, str):
        titles = metrics_subset
    else:
        titles = []
        for metric in metrics_subset:
            for sample in sample_type:
                titles.append(f"{metric} {sample}")
    latex_code += " & ".join(['model'] + titles) + " \\\\ \n \\midrule \n"
    for model in models_subset:
        if isinstance(sample_type, str):
            latex_code += " & ".join([model] + [f"{metrics[model][metric][sample_type][measure_type]:.2f}" for metric in
                                            metrics_subset]) + " \\\\ \n"
        else:
            values = []
            for metric in metrics_subset:
                for sample in sample_type:
                    values.append(f"{metrics[model][metric][sample][measure_type]:.2f}")
            latex_code += " & ".join([model] + values) + " \\\\ \n"

    latex_code += "\\bottomrule \n \end{tabular} \n \\caption{Quality Evaluation} \n \\label{tab:my_label} \n \\end{table}"
    return latex_code


def get_latex_table_from_metrics_vertical(metrics: dict, sample_type: list = ['all'], measure_type: str = 'all',
                                 metrics_subset: list = None, models_subset: list = None):
    samples_dictionaries = {'all': '', 'in_sample': ' In', 'out_sample': ' Out'}
    metrics_subset = metrics_subset or ['KL Divergence', 'Wasserstein-1', 'NIP-G', 'Predictive Capacity', 'Accuracy',
                                        'RMSE']
    models_subset = models_subset or ['VI', 'LA', 'HMC', 'Deep Ensemble', 'MC Dropout', 'Bootstrap']
    columns = 'c' * len(models_subset)
    latex_code = "\\begin{table}[H] \n \\centering \n \\begin{tabular}{l|" + f"{columns}" + "} \\toprule \n"

    latex_code += " & ".join(['Model'] + models_subset) + " \\\\ \n \\midrule \n"

    for metric in metrics_subset:
        for sample in sample_type:
            cols = [f"{metric} {samples_dictionaries[sample]}"]
            for model in models_subset:
                if metrics[model][metric][sample][measure_type] <= 500:
                    cols.append(f"{metrics[model][metric][sample][measure_type]:.2f}")
                else:
                    cols.append("$>500$")

            latex_code += " & ".join(cols) + " \\\\ \n"

    latex_code += "\\bottomrule \n \end{tabular} \n \\caption{Quality Evaluation} \n \\label{tab:my_label} \n \\end{table}"
    return latex_code


# %% Specify model specifics to evaluate
params['n_epochs'] = 400
metrics = get_all_metrics(params)
# %%
latex_code = get_latex_table_from_metrics(metrics, sample_type='all', measure_type='all')
save_text_file(filename='results/regression/all_metrics_test_set.txt', metrics=latex_code)

latex_code = get_latex_table_from_metrics(metrics, sample_type='in_sample', measure_type='all')
save_text_file(filename='results/regression/all_metrics_in_sample.txt', metrics=latex_code)

latex_code = get_latex_table_from_metrics(metrics, sample_type='out_sample', measure_type='all')
save_text_file(filename='results/regression/all_metrics_out_sample.txt', metrics=latex_code)

latex_code = get_latex_table_from_metrics(metrics, sample_type='all', measure_type='aleatoric',
                                          metrics_subset=['KL Divergence', 'Wasserstein-1', 'NIP-G'])
save_text_file(filename='results/regression/distance_metrics_test_set_aleatoric.txt', metrics=latex_code)

latex_code = get_latex_table_from_metrics(metrics, sample_type='all', measure_type='epistemic',
                                          metrics_subset=['KL Divergence', 'Wasserstein-1', 'NIP-G'])
save_text_file(filename='results/regression/distance_metrics_test_set_epistemic.txt', metrics=latex_code)

latex_code = get_latex_table_from_metrics(metrics, sample_type='in_sample', measure_type='aleatoric',
                                          metrics_subset=['KL Divergence', 'Wasserstein-1', 'NIP-G'])
save_text_file(filename='results/regression/distance_metrics_in_sample_aleatoric.txt', metrics=latex_code)

latex_code = get_latex_table_from_metrics(metrics, sample_type='in_sample', measure_type='epistemic',
                                          metrics_subset=['KL Divergence', 'Wasserstein-1', 'NIP-G'])
save_text_file(filename='results/regression/distance_metrics_in_sample_epistemic.txt', metrics=latex_code)

latex_code = get_latex_table_from_metrics(metrics, sample_type='out_sample', measure_type='aleatoric',
                                          metrics_subset=['KL Divergence', 'Wasserstein-1', 'NIP-G'])
save_text_file(filename='results/regression/distance_metrics_out_sample_aleatoric.txt', metrics=latex_code)

latex_code = get_latex_table_from_metrics(metrics, sample_type='out_sample', measure_type='epistemic',
                                          metrics_subset=['KL Divergence', 'Wasserstein-1', 'NIP-G'])
save_text_file(filename='results/regression/distance_metrics_out_sample_epistemic.txt', metrics=latex_code)

# %%
models_to_compare = [
    'deep_het_True_ts_100_ep_400_hd_40_nh_1_nm_50_lr_0.01.json',
    'mcdropout_het_True_ts_100_ep_1000_hd_40_nh_1_nm_50_lr_0.01.json',
    'bootstrap_het_True_ts_100_ep_1000_hd_40_nh_1_nm_50_lr_0.01.json',
    'variational_het_True_ts_100_ep_1000_hd_40_nh_1_sp_1000_lr_0.01.json',
    'laplace_het_True_ts_100_ep_1000_hd_40_nh_1_sp_1000_lr_0.01.json',
    'hamiltonian_het_True_ts_100_ep_1000_hd_40_nh_1_sp_1000_lr_0.01.json'
]

models_to_compare = [
    'deep_het_True_ts_30_ep_400_hd_40_nh_1_nm_50_lr_0.01.json',
    'mcdropout_het_True_ts_30_ep_1000_hd_40_nh_1_nm_50_lr_0.01.json',
    'bootstrap_het_True_ts_30_ep_400_hd_40_nh_1_nm_50_lr_0.01.json',
    'variational_het_True_ts_100_ep_400_hd_40_nh_1_sp_1000_lr_0.01.json',
    'laplace_het_True_ts_30_ep_1000_hd_40_nh_1_sp_250_lr_0.01.json',
    'hamiltonian_het_True_ts_30_ep_400_hd_40_nh_1_sp_1000_lr_0.01.json'
]
model_names = [
    'Deep Ensemble',
    'MC Dropout',
    'Bootstrap',
    'VI',
    'LA',
    'HMC'
]

reference = 'reference_het_True_ts_100_ep_400_hd_40_nh_1_nm_50_lr_0.01.json'

metrics = get_all_metrics_from_models_files(models_files=models_to_compare, reference_file=reference,
                                            models_names=model_names)

latex_code = get_latex_table_from_metrics_vertical(metrics, sample_type=['all', 'in_sample', 'out_sample'],
                                          measure_type='all',
                                          metrics_subset=['KL Divergence', 'Wasserstein-1', 'NIP-G'])
save_text_file(filename='results/tables/regression/distance_measures_30.txt', metrics=latex_code)

latex_code = get_latex_table_from_metrics_vertical(metrics, sample_type=['all', 'in_sample', 'out_sample'],
                                          measure_type='aleatoric',
                                          metrics_subset=['KL Divergence', 'Wasserstein-1', 'NIP-G'])
save_text_file(filename='results/tables/regression/distance_measures_aleatoric_30.txt', metrics=latex_code)

latex_code = get_latex_table_from_metrics_vertical(metrics, sample_type=['all', 'in_sample', 'out_sample'],
                                          measure_type='epistemic',
                                          metrics_subset=['KL Divergence', 'Wasserstein-1', 'NIP-G'])
save_text_file(filename='results/tables/regression/distance_measures_epistemic_30.txt', metrics=latex_code)

latex_code = get_latex_table_from_metrics_vertical(metrics, sample_type=['all'],
                                          measure_type='all',
                                          metrics_subset=['Predictive Capacity', 'Accuracy', 'RMSE'])
save_text_file(filename='results/tables/regression/evaluation_metrics_30.txt', metrics=latex_code)

# %% Get training times
models = [5, 10, 50]
samples_prediction = [100, 250, 1000]
samples = [15, 30, 100]
epochs = [200, 400, 1000]
models_type = ['variational', 'laplace', 'hamiltonian', 'deep', 'mcdropout', 'bootstrap']
models_names = get_models_name_from_model_type(models_type)

results_bayesian = {}
results_ensemble = {}
for n_epochs in epochs:
    for n_samples in samples:
        for n_samples_prediction, n_models in zip(samples_prediction, models):
            params = copy.deepcopy(model_params_template)
            params['n_models'] = n_models
            params['samples_prediction'] = n_samples_prediction
            params['n_samples'] = n_samples
            params['n_epochs'] = n_epochs
            for model_type, model_name in zip(models_type, models_names):
                model_result = read_json_file(get_model_filename(model_type=model_type, config=params))
                if not model_result:
                    continue
                if model_type in ['variational', 'laplace', 'hamiltonian']:
                    model_config = f'{n_samples}_{n_samples_prediction}_{n_epochs}'
                    if model_name not in results_bayesian:
                        results_bayesian[model_name] = {}
                    results_bayesian[model_name][model_config] = model_result["training_time"]
                else:
                    model_config = f'{n_samples}_{n_models}_{n_epochs}'
                    if model_name not in results_ensemble:
                        results_ensemble[model_name] = {}
                    results_ensemble[model_name][model_config] = model_result["training_time"]
