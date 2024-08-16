from assets.model_params import get_model_arguments
from assets.tunning_config import get_config_file, get_training_inputs
from training.training_functions import train_and_test_model

#import wandb

model = 'variational'

model_arguments = get_model_arguments(model)


def train(config):
    """ Config comes from wandb"""
    model_name = 'variational'
    training_inputs = get_training_inputs(config, model_name)
    model = train_and_test_model(model_name, **model_arguments, **training_inputs)
    return model.test_mse


config_file = get_config_file(model)


def main():
    wandb.init(project="Thesis", name=config_file['name'])
    mse = train(wandb.config)
    wandb.log({"mse": mse})


sweep_id = wandb.sweep(sweep=config_file, project="Thesis")

wandb.agent(sweep_id, function=main, count=27)

# wandb.teardown()
