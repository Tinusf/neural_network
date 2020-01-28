import numpy as np
import activation_funcs
from scipy.special import softmax

class Layer:
    def __init__(self, w, x, b, loss, activation_func):
        # This is the weights in to this layer.
        self.w = w
        # This is the activation_funcs in to this layer.
        self.x = x
        # Biases for this layer.
        self.b = b
        self.loss = loss
        self.activation_func = activation_func

    def forward(self, x):
        # TODO: Remember to change it for the first layer without activations.
        z = self.get_z(x)
        return self.activation(z)

    def get_z(self, x):
        return np.transpose(self.w).dot(x) + self.b

    def activation(self, value):
        if self.activation_func == "relu":
            return activation_funcs.relu(value)
        if self.activation_func == "softmax":
            return softmax(value)





if __name__ == '__main__':
    weights = np.array([[1, 2], [2, 3]])
    activation_funcs = np.array([0.2, -0.2])
    layer = Layer(weights, activation_funcs)
    print(layer.get_output())
