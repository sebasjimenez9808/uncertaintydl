import copy

from experiments.regression.plots_model import plot_predictions_with_uncertainty, plot_uncertainties, \
    plot_kl_divergence, grid_three_plots, grid_model_list_plots
from experiments.regression.utils import (model_params_template, get_model_filename,
                                          read_json_file, get_models_name_from_model_type)

params = copy.deepcopy(model_params_template)
params['n_models'] = 5
params['samples_prediction'] = 1000
params['n_epochs'] = 1000

# %%
models = ['variational', 'laplace', 'hamiltonian', 'deep', 'mcdropout', 'bootstrap', 'reference']
models = ['mcdropout']
models_names = get_models_name_from_model_type(models)
for model_type, model_name in zip(models, models_names):
    model = read_json_file(get_model_filename(model_type=model_type, config=params))
    model_ref = read_json_file(get_model_filename(model_type='reference', config=params))
    plot_filename = get_model_filename(model_type=model_type, config=params, folder='plots', file_format='png')
    uncertainties_file_name = get_model_filename(model_type=model_type, config=params, folder='plots',
                                                 file_format='png', extra_name_part='_uncertainties')
    kl_div_filename = get_model_filename(model_type=model_type, config=params, folder='plots',
                                         file_format='png', extra_name_part='_kl_div')
    filename_grid = get_model_filename(model_type=model_type, config=params, folder='plots',
                                       file_format='png', extra_name_part='_grid')

    plot_predictions_with_uncertainty(model=model, filename=plot_filename, title=model_name,
                                      y_lim=(-70, 70))
    plot_uncertainties(model=model, filename=uncertainties_file_name, title=model_name,
                       y_lim=(0, 70))
    plot_kl_divergence(model=model, model_ref=model_ref, filename=kl_div_filename, title=model_name,
                       y_lim=(0, 70))
    grid_three_plots(model=model, model_ref=model_ref, filename=plot_filename, title=model_name,
                     y_lim=(-70, 70))
# %%
models_to_compare = [
    'reference_het_True_ts_100_ep_400_hd_40_nh_1_nm_50_lr_0.01.json',
    'deep_het_True_ts_100_ep_400_hd_40_nh_1_nm_50_lr_0.01.json',
    'mcdropout_het_True_ts_100_ep_1000_hd_40_nh_1_nm_50_lr_0.01.json',
    'bootstrap_het_True_ts_100_ep_1000_hd_40_nh_1_nm_50_lr_0.01.json',
    'variational_het_True_ts_100_ep_1000_hd_40_nh_1_sp_1000_lr_0.01.json',
    'laplace_het_True_ts_100_ep_1000_hd_40_nh_1_sp_1000_lr_0.01.json',
    'hamiltonian_het_True_ts_100_ep_1000_hd_40_nh_1_sp_1000_lr_0.01.json'
]
model_names = [
    'Reference',
    'Deep Ensemble',
    'MC Dropout',
    'Bootstrap',
    'Variational',
    'Laplace',
    'Hamiltonian'
]

reference = 'reference_het_True_ts_100_ep_400_hd_40_nh_1_nm_50_lr_0.01.json'

models_jsons = [read_json_file(f'results/regression/{m}') for m in models_to_compare]
reference_json = read_json_file(f'results/regression/{reference}')

grid_model_list_plots(models_list=models_jsons, model_ref=reference_json, filename='plots/regression/grid_models.png',
                      title='Regression Models', y_lim=(-70, 70), fig_size=(30, 5 * len(models_jsons)),
                      model_names=model_names, title_plots_fontsize=25, title_rows_fontsize=25,
                      legends_fontsize=20)


models_to_compare = [
    'reference_het_True_ts_100_ep_400_hd_40_nh_1_nm_50_lr_0.01.json',
    'deep_het_True_ts_30_ep_400_hd_40_nh_1_nm_50_lr_0.01.json',
    'mcdropout_het_True_ts_30_ep_1000_hd_40_nh_1_nm_50_lr_0.01.json',
    'bootstrap_het_True_ts_30_ep_400_hd_40_nh_1_nm_50_lr_0.01.json',
    'variational_het_True_ts_100_ep_400_hd_40_nh_1_sp_1000_lr_0.01.json',
    'laplace_het_True_ts_30_ep_1000_hd_40_nh_1_sp_250_lr_0.01.json',
    'hamiltonian_het_True_ts_30_ep_400_hd_40_nh_1_sp_1000_lr_0.01.json'
]


reference = 'reference_het_True_ts_100_ep_400_hd_40_nh_1_nm_50_lr_0.01.json'

models_jsons = [read_json_file(f'results/regression/{m}') for m in models_to_compare]
reference_json = read_json_file(f'results/regression/{reference}')

grid_model_list_plots(models_list=models_jsons, model_ref=reference_json, filename='plots/regression/grid_models_30.png',
                      title='Regression Models', y_lim=(-70, 70), fig_size=(30, 5 * len(models_jsons)),
                      model_names=model_names, title_plots_fontsize=25, title_rows_fontsize=25,
                      legends_fontsize=20)