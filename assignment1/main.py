from config import Config
from layer import Layer
import numpy as np
from network import Network

np.random.seed(42)


# It was suggested to use this.
# from configparser import ConfigParser


def main():
    config = Config("config.txt")

    X_data = np.array([[1, 1, 1],
                       [0, 1, 1],
                       [1, 0, 1],
                       [0, 0, 0]])
    y_data = np.array([1, 0, 0, 0])
    activation_functions = ["relu", "relu", "relu"]
    network = Network(X_data, y_data, [3, 2, 1], "L2", activation_functions)

    network.train()


if __name__ == '__main__':
    main()
