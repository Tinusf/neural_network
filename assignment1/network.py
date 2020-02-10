from layer import Layer
import numpy as np
import activation_funcs
import matplotlib.pyplot as plt

np.random.seed(42)


class Network:
    def __init__(self, X_trian, y_train, number_of_nodes, loss, activation_functions, lr=0.0001,
                 X_val=None, y_val=None, regularization_factor=0, no_epochs=10000):
        """
        :param X_trian:
        :param y_train:
        :param number_of_nodes: Can for example be: [2, 1]
        Then the input layer consists of 2 nodes and there is 1 output node.
        This would be modelled by using one layer in this program.
        """
        self.X_train = X_trian
        self.y_train = y_train
        self.loss = loss
        self.layers = []
        self.lr = lr
        self.no_epochs = no_epochs
        self.regularization_factor = regularization_factor
        if X_val is not None:
            self.X_val = X_val
            self.y_val = y_val
            self.use_validation = True
        else:
            self.use_validation = False

        for i in range(len(number_of_nodes) - 1):
            # Transpose the weights here so I don't have to do it every time you run feedforward.
            weights = np.random.normal(size=number_of_nodes[i:i + 2]).T / np.sqrt(
                number_of_nodes[i])
            biases = np.random.normal(size=(number_of_nodes[i + 1], 1))
            # Create this layer.
            layer = Layer(weights, X_trian, biases, loss, activation_functions[i])
            self.layers.append(layer)

    def feed_forward(self, x):
        activations = [x]
        zs = []
        for layer in self.layers:
            x, z = layer.forward(x)
            zs.append(z)
            activations.append(x)
        return np.array(activations), np.array(zs)

    def get_l2_regularization_loss(self):
        all_weights_squared = np.sum(np.sum(layer.w ** 2) for layer in self.layers)
        all_biases_squared = np.sum(np.sum(layer.b ** 2) for layer in self.layers)
        return self.regularization_factor * (all_weights_squared + all_biases_squared)

    def get_loss(self, layer, target_y, estimate_y, derivate=False):
        if layer.loss == "L2":
            loss = (target_y - estimate_y) ** 2
            if derivate:
                return estimate_y - target_y
            return loss
        if layer.loss == "cross_entropy":
            # add 1e-9 just to avoid taking the log of 0.
            loss = -np.sum(target_y * np.log(estimate_y + 1e-9))

            if derivate:
                derivate = estimate_y - target_y
                return derivate
            return loss

    def get_activations_func(self, layer, z, derivate=False):
        if layer.activation_func == "relu":
            return activation_funcs.relu(z, derivate=derivate)
        elif layer.activation_func == "tanh":
            return activation_funcs.tanh(z, derivate=derivate)
        elif layer.activation_func == "softmax":
            # I assume that if we use softmax then we use cross_entropy and therefore we don't
            # need the derivate of the activation funciton.
            return None
        elif layer.activation_func == "linear":
            return activation_funcs.linear(z, derivate=derivate)

    def back_propagation(self, activations, target_y, zs, learning_rate=0.0001):
        # Keep track of the previous error.
        last_error = None
        # Go through every layer backwards.
        for layer_i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_i]
            z = zs[layer_i]
            if layer_i == len(self.layers) - 1:
                # This is the last layer
                activation_func_derivate = self.get_activations_func(layer, z, derivate=True)
                loss = self.get_loss(layer, target_y, activations[-1])
                if self.regularization_factor > 0:
                    # Add regularization loss if it's used.
                    l2_loss = self.get_l2_regularization_loss()
                    loss += l2_loss
                if activation_func_derivate is None:
                    # If it's none it means we are in a softmax/ cross_entropy layer and we
                    # don't need it.
                    last_error = np.array((self.get_loss(layer, target_y, activations[-1],
                                                         derivate=True)))
                else:
                    last_error = np.array((self.get_loss(layer, target_y, activations[-1],
                                                         derivate=True))) \
                                 * np.array((activation_func_derivate))

            else:
                # Not the last layer.
                # Next layer.
                next_layer = self.layers[layer_i + 1]
                activation_func_derivate = self.get_activations_func(layer, z, derivate=True)
                if activation_func_derivate is None:
                    last_error = np.transpose(next_layer.w).dot(last_error)
                else:
                    last_error = np.transpose(next_layer.w).dot(last_error) * (
                        activation_func_derivate)

            if self.regularization_factor > 0:
                layer.w = layer.w - learning_rate * (np.array(last_error).dot(np.transpose(
                    activations[layer_i])) + self.regularization_factor * layer.w)
                layer.b = layer.b - learning_rate * (last_error + self.regularization_factor *
                                                     layer.b)
            else:
                layer.w = layer.w - (learning_rate * np.array(last_error).dot(np.transpose(
                    activations[layer_i])))
                layer.b = layer.b - (learning_rate * last_error)

        return loss

    def train(self):
        # These lists are used for plotting the graph.
        epoches = []
        train_losses = []
        val_losses = []
        for epoch in range(1, self.no_epochs):
            print("Epoch", epoch)
            total_loss = 0.0
            for i in range(len(self.X_train)):
                # Needs to be shape for example: (2,1) instead of (2,)
                x = self.X_train[i].reshape(self.X_train[i].shape[0], 1)
                y = self.y_train[i]
                if y.shape:  # if the y is an array and not just a single number.
                    y = y.reshape(y.shape[0], 1)
                activations, zs = self.feed_forward(x)
                loss = self.back_propagation(activations, y, zs, learning_rate=self.lr).squeeze()
                total_loss += loss

            training_loss = total_loss / len(self.X_train)
            print("Training loss", training_loss)
            if self.use_validation:
                # Check validation loss
                total_val_loss = 0.0
                for i in range(len(self.X_val)):
                    x = self.X_val[i].reshape(self.X_val[i].shape[0], 1)
                    y = self.y_val[i]
                    if y.shape:  # if the y is an array and not just a single number.
                        y = y.reshape(y.shape[0], 1)
                    activations, zs = self.feed_forward(x)
                    loss = self.get_loss(self.layers[-1], y, activations[-1]).squeeze()
                    total_val_loss += loss
                val_loss = total_val_loss / len(self.X_val)
                val_losses.append(val_loss)
                print("Validation loss", val_loss)
            epoches.append(epoch)
            train_losses.append(training_loss)
            if epoch % 5 == 1:
                plt.plot(epoches, train_losses, "b", label="Training loss")
                if self.use_validation:
                    plt.plot(epoches, val_losses, "r", label="Validation loss")
                if epoch == 1:
                    # Only plot the legend on the first epoch.
                    plt.legend()
                plt.pause(0.0001)

            if epoch == self.no_epochs - 1:
                # write out the learned weights.
                f = open("weights_learned.txt", "w")
                weights = [layers.w for layers in self.layers]
                biases = [layers.b for layers in self.layers]
                output = "weights: " + str(weights) + "\nbiases: " + str(biases)
                f.write(output)
                f.close()
                print(output)
        plt.show()
