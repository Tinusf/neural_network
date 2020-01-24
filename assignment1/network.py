from layer import Layer
import numpy as np
import activations

np.random.seed(42)


class Network:
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        self.weights = np.array(np.random.normal(size=2))

        # print(weights)
        bias = 0.2
        input_layer = Layer(self.weights, X_data, bias, "L2", "relu")

        self.layers = [input_layer]

    def feed_forward(self, x):
        return self.layers[0].forward(x)

    def back_propagation(self, activation, x, target_y, learning_rate=0.05):
        # Need to be the sum of all
        loss = (target_y - activation) ** 2
        print("loss", loss)
        z = self.layers[0].get_z(x)
        gradient = (activation - target_y) * x * learning_rate * activations.relu(z, True)
        self.layers[0].w -= gradient
        self.layers[0].b -= (activation - target_y) * learning_rate * activations.relu(z, True)

    def train(self):
        for epoch in range(10000):
            for i in range(len(self.X_data)):
                x = self.X_data[i]
                y = self.y_data[i]

                activations = self.feed_forward(x)
                # print(activations)
                self.back_propagation(activations, x, y)
        print("siste")
        print("her:")
        print(self.layers[0].forward(np.array([0, 0])))
        print(self.layers[0].forward(np.array([1, 0])))
        print(self.layers[0].forward(np.array([0, 1])))
        print(self.layers[0].forward(np.array([1, 1])))
        print("done")
        print(self.weights)
        print(self.layers[0].b)
        pass

    def predict(self, input):
        return np.max(0, input.dot(self.weights))
