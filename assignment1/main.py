from config import Config
import numpy as np
from read_data import read_file, one_hot
from network import Network

np.random.seed(42)


def main():
    config = Config("config.txt")
    lr = config.config["learning_rate"]
    X_data, y_data = read_file(config.config["training"])
    activation_functions = config.config["activations"]
    loss_type = config.config["loss_type"]

    layers = config.config["layers"]
    layers.insert(0, X_data.shape[1])

    if loss_type == "cross_entropy":
        activation_functions.append("softmax")
        layers.append(10)
        y_data = one_hot(y_data)
        print("lo")
    else:
        activation_functions.append("relu") # TODO:  Typisk linear.
        layers.append(1)
    X_data = np.nan_to_num(X_data)
    y_data = np.nan_to_num(y_data)
    network = Network(X_data, y_data, layers, loss_type, activation_functions,
                      lr)

    # X_data = np.array([[1, 1],
    #                    [1, 0],
    #                    [0, 1],
    #                    [0, 0]])
    # y_data = np.array([1, 1, 1, 0])
    # activation_functions = ["tanh", "tanh", "tanh"]
    # network = Network(X_data, y_data, [2, 1], "L2", activation_functions)

    # X_data = np.array([[1, 1, 1, 1],
    #                    [0, 1, 1, 0],
    #                    [1, 0, 1, 0],
    #                    [0, 0, 0, 0]])
    # y_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # activation_functions = ["softmax", "softmax"]
    # network = Network(X_data, y_data, [4, 4, 4], "cross_entropy", activation_functions, lr)


    network.train()


if __name__ == '__main__':
    main()
