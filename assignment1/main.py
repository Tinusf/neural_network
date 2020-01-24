from config import Config
from layer import Layer
import numpy as np
from network import Network
np.random.seed(42)

# It was suggested to use this.
# from configparser import ConfigParser



def main():
    config = Config("config.txt")

    # Input layer

    X_data = np.array([[1, 1, 1],
                       [0, 1, 1],
                       [1, 0, 0],
                       [0, 0, 1]])
    y_data = np.array([1, 1, 3, 2])
    network = Network(X_data, y_data, [3, 2], "L2", "softmax")

    network.train()
    # print(network.predict(np.array([1,1])))

    # activations = np.array([1, 1])
    # weights = np.random.normal(size=2)
    # print(weights)
    # bias = 0.2
    # input_layer = Layer(weights, activations, bias, "L2", "relu")
    #
    # print(input_layer.forward())
    # # Output layer
    # activations = np.array([])

    # Data. Simple AND example:



if __name__ == '__main__':
    main()
