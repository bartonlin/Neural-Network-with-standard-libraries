import numpy as np


def cross_entropy_sigmoid(y, y_hat):
    m = y.shape[1]
    # cost = np.sum(y*np.log(y_hat) + (1-y)*np.log(1 - y_hat)) / (-1*m)
    cost = (1. / m) * (-np.dot(y, np.log(y_hat).T) - np.dot(1 - y, np.log(1 - y_hat).T))

    # So that we have a real number at the end instead of a singleton; e.g. [[3]] => 3
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost


def cross_entropy_sigmoid_derivative(y, y_hat):
    m = y.shape[1]
    return (-(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat)))


cost_functions = {"cross_entropy_sigmoid": (cross_entropy_sigmoid, cross_entropy_sigmoid_derivative)}