from config import Config
import numpy as np
from read_data import read_file, one_hot, get_num_of_classes
from network import Network

np.random.seed(42)


def main():
    config = Config("config.txt")
    lr = config.config["learning_rate"]
    X_train, y_train = read_file(config.config["training"])
    X_val, y_val = read_file(config.config["validation"])
    activation_functions = config.config["activations"]
    loss_type = config.config["loss_type"]
    l2_regularization_factor = config.config["L2_regularization"]

    layers = config.config["layers"]
    layers.insert(0, X_train.shape[1])

    if loss_type == "cross_entropy":
        n_classes = get_num_of_classes(y_train)
        activation_functions.append("softmax")
        layers.append(n_classes)
        y_train = one_hot(y_train, classes=n_classes)
        y_val = one_hot(y_val, classes=n_classes)
    else:
        activation_functions.append("relu")  # TODO:  Typisk linear.
        layers.append(1)
    network = Network(X_train, y_train, layers, loss_type, activation_functions,
                      lr, X_val=X_val, y_val=y_val, regularization_factor=l2_regularization_factor)

    # X_data = np.array([[1, 1],
    #                    [1, 0],
    #                    [0, 1],
    #                    [0, 0]])
    # y_data = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])
    # activation_functions = ["softmax"]
    # layers = [2, 2]
    # network = Network(X_data, y_data, layers, "cross_entropy", activation_functions,
    # regularization_factor=0.0001)

    # X_data = np.array([[0.8, 0.7, 0.2, 1],
    #                    [0, 1, 1, 0],
    #                    [1, 0, 1, 0],
    #                    [0, 0, 0, 0]])
    # y_data = one_hot(np.array([0, 1, 2, 3]), 4)
    # y_train = one_hot(y_train)
    # activation_functions = ["relu", "softmax"]
    # layers = [784, 400, 10]
    # # TODO: hvorfor funker x_data og y_data men ikke x_train og y_train???? er det fordi
    # # den ikke har l2 generalization så vekter blir større og større for ever?
    # network = Network(X_train, y_train, layers, "cross_entropy", activation_functions, lr)

    assert len(layers) == len(activation_functions) + 1
    network.train()


if __name__ == '__main__':
    main()
