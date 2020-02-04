from config import Config
import numpy as np
from read_data import read_file, one_hot, get_num_of_classes
from network import Network

np.random.seed(42)


def main():
    config = Config("5_any_structure.txt")
    lr = config.config["learning_rate"]
    no_epochs = config.config["no_epochs"]
    X_train, y_train = read_file(config.config["training"])
    X_val, y_val = None, None
    if "validation" in config.config:
        X_val, y_val = read_file(config.config["validation"])
    activation_functions = []
    if "activations" in config.config:
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
        if y_val is not None:
            y_val = one_hot(y_val, classes=n_classes)
    else:
        activation_functions.append("linear")  # TODO:  Typisk linear, kan være relu og.
        layers.append(1)
    network = Network(X_train, y_train, layers, loss_type, activation_functions,
                      lr, X_val=X_val, y_val=y_val,
                      regularization_factor=l2_regularization_factor, no_epochs=no_epochs)

    assert len(layers) == len(activation_functions) + 1
    network.train()


if __name__ == '__main__':
    main()
