from Config import Config
from Layer import Layer
import numpy as np


def main():
    config = Config("config.txt")

    # Input layer
    activations = None
    weights = np.array([0.2, 0.2])
    bias = 0.2
    input_layer = Layer(weights, activations, bias)

    # Output layer
    activations = np.array([])

    # Data. Simple AND example:
    data_X = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    data_y = np.array([1, 0, 0, 0])


if __name__ == '__main__':
    main()
