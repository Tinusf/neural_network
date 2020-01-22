import numpy as np


class Layer:
    def __init__(self, weights, activations, bias):
        self.weights = weights
        self.activations = activations
        self.bias = bias

    def get_output(self):
        # TODO: Remember to change it for the first layer without activations.
        return np.transpose(self.weights).dot(activations)

    def activation(self):
        pass


if __name__ == '__main__':
    weights = np.array([[1, 2], [2, 3]])
    activations = np.array([0.2, -0.2])
    layer = Layer(weights, activations)
    print(layer.get_output())
