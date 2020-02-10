import numpy as np
import activation_funcs


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
        z = self.get_z(x)
        return self.activation(z), z

    def get_z(self, x):
        # Don't need to transpose the weights because that was done when the weights were created.
        return self.w.dot(x) + self.b

    def activation(self, value):
        if self.activation_func == "relu":
            return activation_funcs.relu(value)
        if self.activation_func == "tanh":
            return activation_funcs.tanh(value)
        if self.activation_func == "softmax":
            return activation_funcs.softmax(value)
        if self.activation_func == "linear":
            return activation_funcs.linear(value)
