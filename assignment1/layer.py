import numpy as np
import activations


class Layer:
    def __init__(self, w, x, b, loss, activation_func):
        self.w = w
        self.x = x
        self.b = b
        self.loss = loss
        self.activation_func = activation_func

    def forward(self, x):
        # TODO: Remember to change it for the first layer without activations.
        value = np.transpose(self.w).dot(x) + self.b
        return self.activation(value)

    def get_z(self, x):
        return np.transpose(self.w).dot(x) + self.b

    def activation(self, value):
        if self.activation_func == "relu":
            return activations.relu(value)





if __name__ == '__main__':
    weights = np.array([[1, 2], [2, 3]])
    activations = np.array([0.2, -0.2])
    layer = Layer(weights, activations)
    print(layer.get_output())
