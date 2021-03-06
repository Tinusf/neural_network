from config import Config
import numpy as np
from read_data import read_file, one_hot, get_num_of_classes
from network import Network

np.random.seed(42)


def main():
    demo_data=True
    config = Config("6_demo.txt")
    lr = config.config["learning_rate"]
    no_epochs = config.config["no_epochs"]
    X_train, y_train = read_file(config.config["training"])
    if demo_data:
        # Subtract 3 if the demo data is to be used
        y_train = np.array([y - 3 for y in y_train])
    X_val, y_val = None, None
    if "validation" in config.config:
        # Read the validation data.
        X_val, y_val = read_file(config.config["validation"])
        if demo_data:
            # Subtract 3 if the demo data is to be used
            y_val = np.array([y - 3 for y in y_val])
    activation_functions = []
    if "activations" in config.config:
        activation_functions = config.config["activations"]
    loss_type = config.config["loss_type"]
    l2_regularization_factor = config.config["L2_regularization"]

    layers = config.config["layers"]
    # Insert the number of features as the number of nodes in the first layer.
    layers.insert(0, X_train.shape[1])

    if loss_type == "cross_entropy":
        # If cross_entropy is used then we need to use softmax for the last layer.
        n_classes = get_num_of_classes(y_train)
        activation_functions.append("softmax")
        # Append n_classes as the number of nodes in the last layer.
        layers.append(n_classes)
        y_train = one_hot(y_train, classes=n_classes)
        if y_val is not None:
            y_val = one_hot(y_val, classes=n_classes)
    else: # L2
        activation_functions.append("linear")  # TODO:  Typisk linear, kan være relu og.
        # Here we append 1 node at the last layer.
        layers.append(1)
    network = Network(X_train, y_train, layers, loss_type, activation_functions,
                      lr, X_val=X_val, y_val=y_val,
                      regularization_factor=l2_regularization_factor, no_epochs=no_epochs)

    assert len(layers) == len(activation_functions) + 1
    network.train()


if __name__ == '__main__':
    main()
