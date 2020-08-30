import numpy as np


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))

    return s


def d_sigmoid(x):
    # sigmoid函數微分
    s = 1 / (1 + np.exp(-x))

    return s * (1 - s)

activation_functions = {"sigmoid": (sigmoid, d_sigmoid)}
