from config import Config
import numpy as np
from network import Network

np.random.seed(42)


# It was suggested to use this.
# from configparser import ConfigParser


def main():
    config = Config("config.txt")

    # X_data = np.array([[1, 1],
    #                    [1, 0],
    #                    [0, 1],
    #                    [0, 0]])
    # y_data = np.array([1, 1, 1, 0])
    # activation_functions = ["tanh", "tanh", "tanh"]
    # network = Network(X_data, y_data, [2, 1], "L2", activation_functions)

    X_data = np.array([[1, 1, 1, 1],
                       [0, 1, 1, 0],
                       [1, 0, 1, 0],
                       [0, 0, 0, 0]])
    y_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    activation_functions = ["softmax", "softmax"]
    network = Network(X_data, y_data, [4, 4, 4], "cross_entropy", activation_functions)

    network.train()


if __name__ == '__main__':
    main()
