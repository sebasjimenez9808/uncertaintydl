import sys
import os
conf_path = os.getcwd()
print(conf_path)
sys.path.append(conf_path)
from experiments.classification.model_generator import *
import copy
import argparse

from experiments.classification.plots_model import plot_predictions_with_uncertainty
from experiments.classification.utils import model_params_template_information
from experiments.regression.utils import save_json_model, get_model_filename, read_json_file

params = copy.deepcopy(model_params_template_information)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a classification model.")

    # model_vi = train_and_test_model("variational", **params)
    # save_json_model(params, model_vi,  'variational', problem='classification')

    #model = read_json_file(get_model_filename(model_type='variational', config=params, problem='classification'))
    #plot_predictions_with_uncertainty(model=model, filename='variational.png', title='Variational')



    model_vi = train_and_test_model("bootstrap", **params)
    save_json_model(params, model_vi,  'bootstrap', problem='classification')

    #model = read_json_file(get_model_filename(model_type='bootstrap', config=params, problem='classification'))
    #plot_predictions_with_uncertainty(model=model, filename='bootstrap.png', title='Bootstrap')

if __name__ == '__main__':
    main()