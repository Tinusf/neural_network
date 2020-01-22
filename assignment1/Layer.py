import numpy as np


class Layer:
    def __init__(self, weights, activations):
        self.weights = weights
        self.activations = activations

    def get_output(self):
        return np.transpose(self.weights).dot(activations)


if __name__ == '__main__':
    weights = np.array([[1, 2], [2, 3]])
    activations = np.array([0.2, -0.2])
    layer = Layer(weights, activations)
    print(layer.get_output())
