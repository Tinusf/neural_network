import numpy as np


def relu(value, derivate=False):
    if derivate:
        return np.sign(value)
        # return 1 if value > 0 else 0
    else:
        return np.maximum(0, value)

