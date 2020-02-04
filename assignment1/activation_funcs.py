import numpy as np
from scipy.special import softmax as sm


def relu(value, derivate=False):
    new = np.copy(value)
    if derivate:
        # return np.sign(value)
        new[new <= 0] = 0
        new[new > 0] = 1
        return new
        # return 1 if new > 0 else 0
    else:
        return np.maximum(0, new)


def tanh(x, derivate=False):
    if derivate:
        return (np.cosh(x) ** 2 - np.sinh(x) ** 2) / np.cosh(x) ** 2
    return np.tanh(x)


def softmax(value, derivate=False):
    if derivate:
        # If softmax is used while cross_entropy is used then we don't need this.
        pass
    else:
        return sm(value)


def linear(value, derivate=False):
    new = np.copy(value)
    if derivate:
        new.fill(1)
        return new
    return new
