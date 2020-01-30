from layer import Layer
import numpy as np
import activation_funcs

np.random.seed(42)


class Network:
    def __init__(self, X_data, y_data, number_of_nodes, loss, activation_functions):
        """
        :param X_data:
        :param y_data:
        :param number_of_nodes: Can for example be: [2, 1]
        Then the input layer consists of 2 nodes and there is 1 output node.
        This would be modelled by using one layer in this program.
        """
        self.X_data = X_data
        self.y_data = y_data
        self.loss = loss
        self.layers = []

        for i in range(len(number_of_nodes) - 1):
            weights = np.random.normal(size=number_of_nodes[i:i + 2]).T
            # TODO: figure out which is correct.
            biases = np.random.normal(size=(number_of_nodes[i + 1], 1))
            layer = Layer(weights, X_data, biases, loss, activation_functions[i])
            self.layers.append(layer)
            print(weights.shape, biases.shape)
        print("lol")

    def feed_forward(self, x):
        activations = [x]
        zs = []
        for layer in self.layers:
            x, z = layer.forward(x)
            zs.append(z)
            activations.append(x)
        return activations, zs

    def get_loss(self, layer, target_y, estimate_y, derivate=False):
        if layer.loss == "L2":
            loss = (target_y - estimate_y) ** 2
            print("loss", loss)
            if derivate:
                return estimate_y - target_y
            return loss
        if layer.loss == "cross_entropy":
            import math
            # loss = -np.sum(target_y * math.log(estimate_y))
            loss = -np.sum([target_y[x] * math.log(estimate_y[x]) for x in range(len(target_y))])
            print("loss", loss)
            if derivate:
                print("lol")
                return estimate_y - target_y
            return loss

    def get_activations_func(self, layer, z, derivate=False):
        if layer.activation_func == "relu":
            return activation_funcs.relu(z, derivate=derivate)
        elif layer.activation_func == "tanh":
            return activation_funcs.tanh(z, derivate=derivate)
        elif layer.activation_func == "softmax":
            return None

    def back_propagation(self, activations, target_y, zs, learning_rate=0.001):
        last_error = None
        for layer_i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_i]
            z = zs[layer_i]
            if layer_i == len(self.layers) - 1:
                # This is the last layer
                activation_func_derivate =self.get_activations_func(layer, z, derivate=True)
                if activation_func_derivate is None:
                    last_error = np.array((self.get_loss(layer, target_y, activations[-1],
                                                         derivate=True)))
                else:
                    last_error = np.array((self.get_loss(layer, target_y, activations[-1],
                                                         derivate=True))) \
                                 * np.array((activation_func_derivate))

            else:
                next_layer = self.layers[layer_i + 1]
                activation_func_derivate = self.get_activations_func(layer, z, derivate=True)
                if activation_func_derivate is None:
                    last_error = np.transpose(next_layer.w).dot(last_error)
                else:
                    last_error = np.transpose(next_layer.w).dot(last_error) * (activation_func_derivate)
                    print(last_error.shape)

            layer.b = layer.b - (learning_rate * last_error)
            layer.w = layer.w - (learning_rate * np.array(last_error).dot(np.transpose(activations[
                                                                                        layer_i])))

    def train(self):
        for epoch in range(10000):
            for i in range(len(self.X_data)):
                # Needs to be shape for example: (2,1) instead of (2,)
                x = self.X_data[i].reshape(self.X_data[i].shape[0], 1)
                y = self.y_data[i]
                if y.shape: # if the y is an array and not just a single number.
                    y = y.reshape(y.shape[0], 1)
                activations, zs = self.feed_forward(x)
                self.back_propagation(activations, y, zs)

    # def predict(self, input):
    # return np.max(0, input.dot(self.weights))
