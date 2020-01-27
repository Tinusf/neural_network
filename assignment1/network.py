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
            weights = np.random.normal(size=number_of_nodes[i:i + 1])
            # TODO: figure out which is correct.
            biases = np.random.normal(size=number_of_nodes[i + 1])

            layer = Layer(weights, X_data, biases, loss, activation_functions[i])
            self.layers.append(layer)

    def feed_forward(self, x):
        activations = [x]
        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)
        return activations

    def back_propagation(self, activations, target_y, learning_rate=0.01):
        if self.loss == "L2":
            # last_layer = self.layers[-1]
            # z_array = last_layer.get_z(x)
            # last_jacobian = np.array([activation_funcs.relu(z, True) for z in z_array])
            last_jacobian = None
            for layer_i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_i]
                z = layer.get_z(activations[layer_i])
                if layer_i == len(self.layers) - 1:
                    # This is the last layer
                    last_jacobian = (activations[layer_i + 1] - target_y).dot(
                        activation_funcs.relu(z, derivate=True))

                else:
                    next_layer = self.layers[layer_i + 1]
                    # z = layer.get_z(x)
                    last_jacobian = np.transpose(next_layer.w).dot(last_jacobian).dot(
                        (activation_funcs.relu(z, derivate=True)))

                loss = (target_y - activations[-1]) ** 2
                print("loss", loss)
                layer.b -= learning_rate * last_jacobian
                layer.w -= learning_rate * activations[layer_i].dot(last_jacobian)

        elif self.loss == "cross_entropy":
            for layer in reversed(self.layers):
                pass

    def train(self):
        for epoch in range(10000):
            for i in range(len(self.X_data)):
                x = self.X_data[i]
                y = self.y_data[i]
                activations = self.feed_forward(x)
                self.back_propagation(activations, y)

    # def predict(self, input):
    # return np.max(0, input.dot(self.weights))
