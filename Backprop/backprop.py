import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up loging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', handlers=[
    logging.FileHandler("backprop_logs/back.log"),
    logging.StreamHandler()
])

EPOCHS=500

class SigmoidMethod:
    """
    Sigmoid forward: f(x) = 1 / (1 + e^(-x))

    Derivacia forward: f'(x) = f(x) * (1 - f(x))
    Sigmoid backward: gradient * Derivacia...
    """

    def forward(self, x):
        self.x = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, gradient):
        sigmoid_deriv = self.output * (1 - self.output)
        return gradient * sigmoid_deriv

class TanhMethod:
    """
       Tanh forward: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

       Tanh backward: gradient = 1 - tanh(x)^2
    """

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, d_output):
        return d_output * (1 - self.output ** 2)

class ReLUMethod:
    """
    ReLU aktivated forward: f(x) = >0 - x

    Derivacia forward: f'(x) = 1 ,pre x>0 |  0 ,pre x<=0
    backward: gradient * Derivacia...
    """

    def forward(self, x):
        self.x = x
        return np.maximum(0, x) # return 0 or x

    def backward(self, gradient):
        return gradient * (self.x > 0)

class LossMethod:
    """
    Mean Squared Error (MSE) forward: loss= 1/n*sum((y_pred - y_true)^2)

    backward: gradient = 2 * (y_pred - y_true) / n
    """

    def forward(self, predicted, target):
        self.predicted = predicted
        self.target = target
        return np.mean((predicted - target) ** 2)

    def backward(self):
        return (2 * (self.predicted - self.target))/self.predicted.size

class LinearLayer:
    """
    LinearLayer:
    - Váhy: random hodnoty z normalneho rozdelenia * 0.01
    - Bias: null vektors

    forward: y= Wx + b

    backward:
    - Gradient weight(weights_gradient): dL/dW= X^T * gradient
    - Gradient bias(biases_gradient): dL/db= sum(gradient)
    - Gradient vstupu(input_gradient): dL/dX= gradient * W^T

    if momentum > 0:
    - weight_momentum: momentum * weight_momentum - learning_rate * weights_gradient
    else:
    - weights -= learning_rate * weights_gradient
    """

    def __init__(self, input_size, output_size, learning_rate=0.1, momentum=0.0):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)

        self.last_input = None
        self.last_output = None


    def forward(self, x):
        self.last_input = x
        self.last_output = np.dot(x, self.weights) + self.biases
        return self.last_output

    def backward(self, gradient):
        input_gradient = np.dot(gradient, self.weights.T)
        weights_gradient = np.dot(self.last_input.T, gradient)
        biases_gradient = np.sum(gradient, axis=0, keepdims=True)

        self.weight_momentum = self.momentum * self.weight_momentum - self.learning_rate * weights_gradient
        self.bias_momentum = self.momentum * self.bias_momentum - self.learning_rate * biases_gradient

        self.weights += self.weight_momentum
        self.biases += self.bias_momentum

        return input_gradient

class LayerContainer:
    """
    forward: Prechadzame všetky vrstvy modelu v poradi a postupne volame ich forward
    Vystup jednej vrstvy služia ako vstup pre next vrstvu

    backward: Prechadza všetky vrstvy modelu v opačnom poradi (od last to first) a postupne volame ich backward
    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


import matplotlib.pyplot as plt


def print_model_visual(results, title):
    problems = ["XOR","xor", "AND", "ORR"]
    for problem in problems:
        plt.figure(figsize=(10, 6))
        for name, losses in results.items():
            if problem in name:
                plt.plot(losses, label=name)
        plt.title(f"{title} - {problem}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

def train_model(X, y, model, epochs, verbose=True):
    loss_fn = LossMethod()
    losses = []
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(X)):
            # forward pass
            prediction = model.forward(X[i].reshape(1, -1))

            # loss calculation
            loss = loss_fn.forward(prediction, y[i].reshape(1, -1))
            total_loss += loss

            # backpropagation
            gradient = loss_fn.backward()
            model.backward(gradient)

        avg_loss = total_loss / len(X)
        losses.append(avg_loss)

        # early stoping
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter > epochs // 10:
            logging.info(f"Early stopping at epoch {epoch}")
            break

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss}")
            logging.info(f"Epoch {epoch}, Loss: {avg_loss}")

    return losses

def main():
    # XOR
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])

    # AND
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([[0], [0], [0], [1]])

    # OR
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([[0], [1], [1], [1]])

    # XOR tests
    tests = {
        "xor (1 hidden layer) ReLu": {
            "X": X_xor, "y": y_xor,
            "layers": [
                LinearLayer(2, 4, learning_rate=0.01, momentum=0.8987060343120898),
                ReLUMethod(),
                LinearLayer(4, 1, learning_rate=0.01, momentum=0.8987060343120898),
                ReLUMethod(),
            ]
        },
        "xor (1 hidden layer) Tahn": {
            "X": X_xor, "y": y_xor,
            "layers": [
                LinearLayer(2, 4, learning_rate=0.01, momentum=0.8987060343120898),
                TanhMethod(),
                LinearLayer(4, 1, learning_rate=0.01, momentum=0.8987060343120898),
                TanhMethod(),
            ]
        },
        "XOR (2 hidden layers) Tanh": {
            "X": X_xor, "y": y_xor,
            "layers": [
                LinearLayer(2, 8, learning_rate=0.01, momentum=0.8987060343120898),
                TanhMethod(),
                LinearLayer(8, 4, learning_rate=0.01, momentum=0.8987060343120898),
                TanhMethod(),
                LinearLayer(4, 1, learning_rate=0.01, momentum=0.8987060343120898),
                TanhMethod(),
            ]
        },
        "XOR (2 hidden layers) ReLu": {
            "X": X_xor, "y": y_xor,
            "layers": [
                LinearLayer(2, 8, learning_rate=0.01, momentum=0),
                ReLUMethod(),
                LinearLayer(8, 4, learning_rate=0.01, momentum=0),
                ReLUMethod(),
                LinearLayer(4, 1, learning_rate=0.01, momentum=0),
                ReLUMethod(),
            ]
        },

        "xor (1 hidden layer) Sigmoid": {
            "X": X_xor, "y": y_xor,
            "layers": [
                LinearLayer(2, 4, learning_rate=0.01, momentum=0.8987060343120898),
                SigmoidMethod(),
                LinearLayer(4, 1, learning_rate=0.01, momentum=0.8987060343120898),
                SigmoidMethod(),
            ]
        },
        "XOR (2 hidden layers) Sigmoid": {
            "X": X_xor, "y": y_xor,
            "layers": [
                LinearLayer(2, 8, learning_rate=0.01, momentum=0.8987060343120898),
                SigmoidMethod(),
                LinearLayer(8, 4, learning_rate=0.01, momentum=0.8987060343120898),
                SigmoidMethod(),
                LinearLayer(4, 1, learning_rate=0.01, momentum=0.8987060343120898),
                SigmoidMethod(),
            ]
        },

        "xor (1 hidden layer) Mix": {
            "X": X_xor, "y": y_xor,
            "layers": [
                LinearLayer(2, 4, learning_rate=0.01, momentum=0.8987060343120898),
                ReLUMethod(),
                LinearLayer(4, 1, learning_rate=0.01, momentum=0.8987060343120898),
                SigmoidMethod(),
            ]
        },
        "XOR Custom": {
            "X": X_xor, "y": y_xor,
            "layers": [
                LinearLayer(2, 8, learning_rate=7.85101358317508e-05, momentum=0.8987060343120898),
                ReLUMethod(),
                LinearLayer(8, 4, learning_rate=7.85101358317508e-05, momentum=0.8987060343120898),
                TanhMethod(),
                LinearLayer(4, 1, learning_rate=7.85101358317508e-05, momentum=0.8987060343120898),
                SigmoidMethod(),
            ]
        },
    }

    # Add AND and OR tests
    for problem, config in [("AND", (X_and, y_and)), ("ORR", (X_or, y_or))]:
        for layers_count in [1, 2]:
            if layers_count == 1:
                layers = [
                    LinearLayer(2, 1, learning_rate=0.01, momentum=0.8987060343120898),
                    SigmoidMethod()
                ]
            else:
                layers = [
                    LinearLayer(2, 4, learning_rate=0.01, momentum=0.8987060343120898),
                    SigmoidMethod(),
                    LinearLayer(4, 1, learning_rate=0.01, momentum=0.8987060343120898),
                    SigmoidMethod()
                ]

            tests[f"{problem} ({layers_count} layer, momentum)"] = {
                "X": config[0], "y": config[1], "layers": layers
            }


    results = {}
    for name, config in tests.items():
        print(f"\nTrain: {name}")
        logging.info(f"Train: {name}")

        model = LayerContainer(config["layers"])
        losses = train_model(config["X"], config["y"], model, epochs=EPOCHS)
        results[name] = losses

    print_model_visual(results, "Training XOR, AND, OR")
