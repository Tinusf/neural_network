import numpy as np


def relu(value, derivate=False):
    if derivate:
        # return np.sign(value)
        value[value <= 0] = 0
        value[value > 0] = 1
        return value
        # return 1 if value > 0 else 0
    else:
        return np.maximum(0, value)


def tanh(x, derivate=False):
    if derivate:
        return (np.cosh(x) ** 2 - np.sinh(x) ** 2) / np.cosh(x) ** 2
    return np.tanh(x)
