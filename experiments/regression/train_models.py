from experiments.regression.model_generator import *
import copy

from experiments.regression.utils import model_params_template, save_json_model

params = copy.deepcopy(model_params_template)

#% Modify params as needed

params['n_models'] = 5
params['samples_prediction'] = 1000
params['n_samples'] = 30


models = [5, 10, 50]
samples_prediction = [100, 250, 1000]
samples = [15, 30, 100]
epochs = [200, 400, 1000]

for n_epochs in epochs:
    for n_samples in samples:
        for n_samples_prediction, n_models in zip(samples_prediction, models):
            params = copy.deepcopy(model_params_template)
            params['n_models'] = n_models
            params['samples_prediction'] = n_samples_prediction
            params['n_samples'] = n_samples
            params['n_epochs'] = n_epochs

            try:
                model_vi = train_and_test_model("variational", **params)
                save_json_model(params, model_vi,  'variational')
            except:
                print("Error in variational model")

            try:
                model_la = train_and_test_model("laplace", **params)
                save_json_model(params, model_la,  'laplace')
            except:
                print("Error in laplace model")

            try:
                model_ham = train_and_test_model("hamiltonian", step_size=0.0005, num_steps_per_sample=100,
                                             burn=20, tau=1, **params)
                save_json_model(params, model_ham,  'hamiltonian')
            except:
                print("Error in hamiltonian model")

            # model_deep = train_and_test_model("deep_ensemble", **params)
            # save_json_model(params, model_deep,  'deep')
            #
            # model_mc = train_and_test_model("mcdropout", dropout_p=0.3, **params)
            # save_json_model(params, model_mc,  'mcdropout')
            #
            # model_boots = train_and_test_model("bootstrap", bootstrap_size=0.6, **params)
            # save_json_model(params, model_boots,  'bootstrap')

            # model_ref = train_and_test_model("deep_ensemble", resample=True, **params)
            # save_json_model(params, model_ref,  'reference')
