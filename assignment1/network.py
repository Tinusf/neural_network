from layer import Layer
import numpy as np
import activations

np.random.seed(42)


class Network:
    def __init__(self, X_data, y_data, number_of_nodes, loss, activation_func):
        """
        :param X_data:
        :param y_data:
        :param number_of_nodes: Can for example be: [2, 1]
        Then the input layer consists of 2 nodes and there is 1 output node.
        """
        self.X_data = X_data
        self.y_data = y_data
        self.loss = loss
        if number_of_nodes[-1] == 1:
            # Only one output node.
            self.weights = np.random.normal(size=number_of_nodes[0])
        else:
            self.weights = np.random.normal(size=number_of_nodes)
        print(self.weights)
        print(""
              "")

        # print(weights)
        bias = [0.2, 0.2]
        input_layer = Layer(self.weights, X_data, bias, loss, activation_func)

        self.layers = [input_layer]

    def feed_forward(self, x):
        return self.layers[0].forward(x)

    def back_propagation(self, activation, x, target_y, learning_rate=0.05):
        if self.loss == "L2":
            # Need to be the sum of all
            loss = (target_y - activation) ** 2
            print("loss", loss)
            z = self.layers[0].get_z(x)
            gradient = (activation - target_y) * learning_rate * x * activations.relu(z, True)
            self.layers[0].w -= gradient
            self.layers[0].b -= (activation - target_y) * learning_rate * activations.relu(z, True)
        elif self.loss == "cross_entropy":
            pass

    def train(self):
        for epoch in range(10000):
            for i in range(len(self.X_data)):
                x = self.X_data[i]
                y = self.y_data[i]

                activations = self.feed_forward(x)
                print(activations)
                self.back_propagation(activations, x, y)

        pass

    def predict(self, input):
        return np.max(0, input.dot(self.weights))
