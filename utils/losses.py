import torch


def negative_log_likelihood(predictions, y):
    """
    :param y: labels
    :param predictions: ([mean1, mean2, ...], [std1, std2, ...])
    :return:
    """
    neg_log_likelihood = torch.log(predictions[1] ** 2) / 2 + (predictions[0] - y) ** 2 / (2 * predictions[1] ** 2)
    return neg_log_likelihood.mean()

