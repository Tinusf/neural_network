from layer import Layer
import numpy as np
import activation_funcs
import matplotlib.pyplot as plt

np.random.seed(42)


class Network:
    def __init__(self, X_trian, y_train, number_of_nodes, loss, activation_functions, lr=0.0001,
                 X_val=None, y_val=None, regularization_factor=0):
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
        self.regularization_factor = regularization_factor
        if X_val is not None:
            self.X_val = X_val
            self.y_val = y_val
            self.use_validation = True
        else:
            self.use_validation = False

        for i in range(len(number_of_nodes) - 1):
            weights = np.random.normal(size=number_of_nodes[i:i + 2]).T / np.sqrt(
                number_of_nodes[i])
            # TODO: figure out which is correct.
            biases = np.random.normal(size=(number_of_nodes[i + 1], 1))
            layer = Layer(weights, X_trian, biases, loss, activation_functions[i])
            self.layers.append(layer)

    def feed_forward(self, x):
        activations = [x]
        zs = []
        for layer in self.layers:
            x, z = layer.forward(x)
            zs.append(z)
            # owo = [str(i) for i in np.reshape(z, -1)]
            # if "nan" in owo:
            #     print("kurwa")
            activations.append(x)
        return np.array(activations), np.array(zs)

    def get_l2_regularization(self, derivate=False, weights=False):
        # TODO: skal denne inkludere biases?
        if derivate:
            l2_derivate_matrix = np.zeros_like(weights)
            l2_derivate_matrix.fill(self.regularization_factor)
            return l2_derivate_matrix
        else:
            all_weights_squared = np.sum(np.sum(layer.w ** 2) for layer in self.layers)
            # all_biases_squared = np.sum(np.sum(layer.b ** 2) for layer in self.layers)
            return self.regularization_factor * all_weights_squared

    def get_loss(self, layer, target_y, estimate_y, derivate=False):
        if layer.loss == "L2":
            loss = (target_y - estimate_y) ** 2
            if derivate:
                return estimate_y - target_y
            return loss
        if layer.loss == "cross_entropy":
            # loss = -np.sum(target_y * np.log(estimate_y))
            loss = -np.sum(target_y * np.log(estimate_y + 1e-9))
            # loss = -np.sum([target_y[x] * np.log(estimate_y[x]+ 1e-9) for x in range(len(target_y))])

            if str(loss) == "nan":
                print("Lol")
            if derivate:
                derivate = estimate_y - target_y
                # derivate[(derivate >= -0.000001) & (derivate <= 0.000001)] = 0
                # derivate[(derivate >= 0.999)] = 1
                # derivate[(derivate <= -0.999)] = -1
                return derivate
            return loss

    def get_activations_func(self, layer, z, derivate=False):
        if layer.activation_func == "relu":
            return activation_funcs.relu(z, derivate=derivate)
        elif layer.activation_func == "tanh":
            return activation_funcs.tanh(z, derivate=derivate)
        elif layer.activation_func == "softmax":
            return None
        elif layer.activation_func == "linear":
            return activation_funcs.linear(z, derivate=derivate)

    def back_propagation(self, activations, target_y, zs, learning_rate=0.0001):
        last_error = None
        for layer_i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_i]
            # if self.regularization_factor:
            #     layer.w = layer.w * self.get_l2_regularization(derivate=True, weights=layer.w)
            z = zs[layer_i]
            if layer_i == len(self.layers) - 1:
                # This is the last layer
                activation_func_derivate = self.get_activations_func(layer, z, derivate=True)
                loss = self.get_loss(layer, target_y, activations[-1])
                if self.regularization_factor > 0:
                    l2_loss = self.get_l2_regularization()
                    loss += l2_loss
                if activation_func_derivate is None:
                    last_error = np.array((self.get_loss(layer, target_y, activations[-1],
                                                         derivate=True)))
                else:
                    last_error = np.array((self.get_loss(layer, target_y, activations[-1],
                                                         derivate=True))) \
                                 * np.array((activation_func_derivate))

            else:
                next_layer = self.layers[layer_i + 1]
                activation_func_derivate = self.get_activations_func(layer, z, derivate=True)
                if activation_func_derivate is None:
                    last_error = np.transpose(next_layer.w).dot(last_error)
                else:
                    last_error = np.transpose(next_layer.w).dot(last_error) * (
                        activation_func_derivate)
            # print(last_error)
            # hender layer.b blir infinite.

            if self.regularization_factor > 0:
                layer.w = layer.w - (learning_rate * np.array(last_error).dot(np.transpose(
                    activations[layer_i])) + self.regularization_factor * layer.w)

                layer.b = layer.b - (learning_rate * last_error)
            else:
                layer.b = layer.b - (learning_rate * last_error)
                layer.w = layer.w - (learning_rate * np.array(last_error).dot(np.transpose(
                    activations[layer_i])))


        return loss

    def train(self):
        epoches = []
        train_losses = []
        val_losses = []
        for epoch in range(1, 10000):
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
            plt.plot(epoches, train_losses, "b", label="Training loss")
            if self.use_validation:
                plt.plot(epoches, val_losses, "r", label="Validation loss")
            plt.legend()
            # plt.show()
            plt.pause(0.000001)

    # def predict(self, input):
    # return np.max(0, input.dot(self.weights))
