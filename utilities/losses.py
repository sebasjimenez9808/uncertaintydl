import torch
import torch.nn as nn


# def negative_log_likelihood(predictions, y):
#     """ Loss function when no uncertainty is present """
#     return 0.5 * (predictions - y) ** 2

def negative_log_likelihood(predictions, y, beta: float = 0, return_vector: bool = False):
    """
    :param y: labels
    :param predictions: ([mean1, mean2, ...], [std1, std2, ...])
    :return:
    """
    neg_log_likelihood = 0.5 * torch.exp(-predictions[:, 1]) * (
                predictions[:, 0] - y.squeeze(1)) ** 2 + 0.5 * predictions[:, 1]

    if beta > 0:
        neg_log_likelihood = neg_log_likelihood * predictions[:, 1].detach() ** beta
    if return_vector:
        return neg_log_likelihood
    return neg_log_likelihood.mean()


def negative_log_likelihood_vector(predictions, y, beta: float = 0):
    """
    :param y: labels
    :param predictions: ([mean1, mean2, ...], [std1, std2, ...])
    :return:
    """
    neg_log_likelihood = 0.5 * torch.exp(-predictions[:, 1]) * (
                predictions[:, 0] - y.squeeze(1)) ** 2 + 0.5 * predictions[:, 1]

    if beta > 0:
        neg_log_likelihood = neg_log_likelihood * predictions[:, 1].detach() ** beta
    return neg_log_likelihood


def classification_loss(predictions, y, num_samples=100, return_vector: bool = False):
    """
    loss function for binary classification
    :param predictions: ([mean1, mean2, ...], [std1, std2, ...])
    :param y: one-hot encoded labels
    :param num_samples:
    :return:
    """
    epsilon_value = 1e-10
    num_classes = int(predictions.size(1) / 2)  # there are two outputs per class: mean and variance

    mean_predictions = predictions[:, : num_classes]
    var_predictions = predictions[:, num_classes:]

    # gets size of the first dimension of the tensor, dim is (batch_size, num_classes)
    batch_size = mean_predictions.size(0)

    # mean_predictions.unsqueeze(0) adds another dimensino at the begninnig of the tensor
    # so instead of being (batch_size, num_classes) it becomes (1, batch_size, num_classes)
    # this is done so pytorch broadcasts the tensor to the correct shape
    epsilon = torch.randn(num_samples, batch_size, num_classes)
    perturbed_logits = mean_predictions.unsqueeze(0) + torch.exp(var_predictions.unsqueeze(0) / 2) * epsilon

    exp_logits = torch.exp(perturbed_logits)  # Exponentiate the perturbed logits
    sum_exp_logits = torch.sum(exp_logits, dim=-1)
    log_sum_exp_logits = torch.log(sum_exp_logits + epsilon_value)  # Take the logarithm

    # to int so it can be used as a tensor
    y_label = y.to(torch.int64)
    # use repetition to get a tensor of same dimensions
    labels_expanded = y_label.unsqueeze(0).expand(num_samples, -1, -1)
    # Gather the perturbed logits for the true class - index 2 because we want to gather along the last dimension which
    # is the label
    true_class_logits = torch.gather(perturbed_logits, 2, labels_expanded)
    # remove last singleton dimension
    true_class_logits = true_class_logits.squeeze(-1)
    # Subtract the log-sum-exp from the perturbed logits for the true class
    logits_subtraction = true_class_logits - log_sum_exp_logits

    exp_logits_subtraction = torch.exp(logits_subtraction)  # Exponentiate the differences

    # Average these exponentials over the samples
    mean_exp_logits_subtraction = torch.mean(exp_logits_subtraction, dim=0)

    # Take the negative logarithm
    loss = torch.log(mean_exp_logits_subtraction + epsilon_value)
    # sum over all data points
    # check if loss is nan
    if return_vector:
        return -loss

    return -loss.sum()


def classification_loss_vector(predictions, y, num_samples=100):
    """
    loss function for binary classification
    :param predictions: ([mean1, mean2, ...], [std1, std2, ...])
    :param y: one-hot encoded labels
    :param num_samples:
    :return:
    """
    epsilon_value = 1e-10
    num_classes = int(predictions.size(1) / 2)  # there are two outputs per class: mean and variance

    mean_predictions = predictions[:, : num_classes]
    var_predictions = predictions[:, num_classes:]

    # gets size of the first dimension of the tensor, dim is (batch_size, num_classes)
    batch_size = mean_predictions.size(0)

    # mean_predictions.unsqueeze(0) adds another dimensino at the begninnig of the tensor
    # so instead of being (batch_size, num_classes) it becomes (1, batch_size, num_classes)
    # this is done so pytorch broadcasts the tensor to the correct shape
    epsilon = torch.randn(num_samples, batch_size, num_classes)
    perturbed_logits = mean_predictions.unsqueeze(0) + torch.exp(var_predictions.unsqueeze(0) / 2) * epsilon

    exp_logits = torch.exp(perturbed_logits)  # Exponentiate the perturbed logits
    sum_exp_logits = torch.sum(exp_logits, dim=-1)
    log_sum_exp_logits = torch.log(sum_exp_logits + epsilon_value)  # Take the logarithm

    # to int so it can be used as a tensor
    y_label = y.to(torch.int64)
    # use repetition to get a tensor of same dimensions
    labels_expanded = y_label.unsqueeze(0).expand(num_samples, -1, -1)
    # Gather the perturbed logits for the true class - index 2 because we want to gather along the last dimension which
    # is the label
    true_class_logits = torch.gather(perturbed_logits, 2, labels_expanded)
    # remove last singleton dimension
    true_class_logits = true_class_logits.squeeze(-1)
    # Subtract the log-sum-exp from the perturbed logits for the true class
    logits_subtraction = true_class_logits - log_sum_exp_logits

    exp_logits_subtraction = torch.exp(logits_subtraction)  # Exponentiate the differences

    # Average these exponentials over the samples
    mean_exp_logits_subtraction = torch.mean(exp_logits_subtraction, dim=0)

    # Take the negative logarithm
    loss = torch.log(mean_exp_logits_subtraction + epsilon_value)
    # sum over all data points
    # check if loss is nan

    return -loss


def classification_error_information(predictions, y, return_vector: bool = False):
    if return_vector:
        return nn.BCELoss(reduce=None)(predictions, y)
    return nn.BCELoss()(predictions, y)

def classification_error_information_vector(predictions, y):
    return nn.BCELoss(reduce=None)(predictions, y)
