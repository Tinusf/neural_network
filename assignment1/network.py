from layer import Layer
import numpy as np
import activations

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
            biases = np.random.normal(size=number_of_nodes[i + 1])

            layer = Layer(weights, X_data, biases, loss, activation_functions[i])
            self.layers.append(layer)

    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def back_propagation(self, activation, x, target_y, learning_rate=0.05):
        if self.loss == "L2":
            for layer in reversed(self.layers):
                loss = (target_y - activation) ** 2
                print("loss", loss)
                print(target_y)
                z = layer.get_z(x)
                gradient = (activation - target_y) * learning_rate * x * activations.relu(z,
                                                                                          derivate=True)

                layer.w -= gradient
                layer.b -= (activation - target_y) * learning_rate * activations.relu(z, True)

        elif self.loss == "cross_entropy":
            for layer in reversed(self.layers):
                pass

    def train(self):
        for epoch in range(10000):
            for i in range(len(self.X_data)):
                x = self.X_data[i]
                y = self.y_data[i]
                activations = self.feed_forward(x)
                self.back_propagation(activations, x, y)

    # def predict(self, input):
    # return np.max(0, input.dot(self.weights))
