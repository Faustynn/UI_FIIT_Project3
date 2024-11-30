import numpy as np
import matplotlib.pyplot as plt


class Sigmoid:
    def forward(self, x):
        """
        Sigmoid aktivačná funkcia: f(x) = 1 / (1 + e^(-x))
        Derivácia: f'(x) = f(x) * (1 - f(x))
        """
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self, gradient):
        sigmoid_derivative = self.forward(self.x) * (1 - self.forward(self.x))
        return gradient * sigmoid_derivative


class ReLU:
    def forward(self, x):
        """
        ReLU aktivačná funkcia: f(x) = max(0, x)
        Derivácia: f'(x) = 1 ak x > 0, inak 0
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, gradient):
        return gradient * (self.x > 0)


class LinearLayer:
    def __init__(self, input_size, output_size, learning_rate=0.1, momentum=0):
        """
        Lineárna vrstva s inicializáciou váh:
        - Váhy: náhodné hodnoty z normálneho rozdelenia * 0.01
        - Bias: nulové vektory

        Dopredný smer: y = Wx + b
        Spätný smer: 
        - Gradient váh: dL/dW = X^T * gradient
        - Gradient bias: dL/db = sum(gradient)
        - Gradient vstupu: dL/dX = gradient * W^T
        """
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Momentum terms pre váhy a bias
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)

        # Uloženie vstupov a výstupov pre backpropagation
        self.last_input = None
        self.last_output = None

    def forward(self, x):
        self.last_input = x
        self.last_output = np.dot(x, self.weights) + self.biases
        return self.last_output

    def backward(self, gradient):
        # Výpočet gradientov
        input_gradient = np.dot(gradient, self.weights.T)
        weights_gradient = np.dot(self.last_input.T, gradient)
        biases_gradient = np.sum(gradient, axis=0, keepdims=True)

        # Aktualizácia váh s alebo bez momentu
        if self.momentum > 0:
            self.weight_momentum = self.momentum * self.weight_momentum - self.learning_rate * weights_gradient
            self.bias_momentum = self.momentum * self.bias_momentum - self.learning_rate * biases_gradient

            self.weights += self.weight_momentum
            self.biases += self.bias_momentum
        else:
            self.weights -= self.learning_rate * weights_gradient
            self.biases -= self.learning_rate * biases_gradient

        return input_gradient


class MSELoss:
    def forward(self, predicted, target):
        """
        Mean Squared Error (MSE) chybová funkcia:
        Loss = 1/n * sum((y_pred - y_true)^2)
        Derivácia: gradient = 2 * (y_pred - y_true) / n
        """
        self.predicted = predicted
        self.target = target
        return np.mean((predicted - target) ** 2)

    def backward(self):
        return 2 * (self.predicted - self.target) / self.predicted.size


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


def train_network(X, y, model, epochs=500, verbose=True):
    """
    Všeobecná trénovacia funkcia pre rôzne problémy
    """
    loss_fn = MSELoss()
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(X)):
            # Dopredný smer
            prediction = model.forward(X[i].reshape(1, -1))

            # Výpočet straty
            loss = loss_fn.forward(prediction, y[i].reshape(1, -1))
            total_loss += loss

            # Spätný smer
            gradient = loss_fn.backward()
            model.backward(gradient)

        avg_loss = total_loss / len(X)
        losses.append(avg_loss)

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss}")

    return losses


def plot_losses(results, title):
    """
    Vizualizácia priebehu trénovania
    """
    plt.figure(figsize=(10, 5))
    for label, loss in results.items():
        plt.plot(loss, label=label)
    plt.title(title)
    plt.xlabel('Epocha')
    plt.ylabel('Strata')
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_experiments():
    # Trénovacie dáta pre logické operácie
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])

    X_and = X_xor
    y_and = np.array([[0], [0], [0], [1]])

    X_or = X_xor
    y_or = np.array([[0], [1], [1], [1]])

    # Experimenty s rôznymi konfiguráciami
    experiments = {
        "XOR (bez momentu, lr=0.1)": {
            "X": X_xor, "y": y_xor,
            "layers": [
                LinearLayer(2, 4, learning_rate=0.1, momentum=0),
                Sigmoid(),
                LinearLayer(4, 1, learning_rate=0.1, momentum=0),
                Sigmoid()
            ]
        },
        "XOR (s momentom, lr=0.1)": {
            "X": X_xor, "y": y_xor,
            "layers": [
                LinearLayer(2, 4, learning_rate=0.1, momentum=0.9),
                Sigmoid(),
                LinearLayer(4, 1, learning_rate=0.1, momentum=0.9),
                Sigmoid()
            ]
        },
        "XOR (s momentom, lr=0.01)": {
            "X": X_xor, "y": y_xor,
            "layers": [
                LinearLayer(2, 4, learning_rate=0.01, momentum=0.9),
                Sigmoid(),
                LinearLayer(4, 1, learning_rate=0.01, momentum=0.9),
                Sigmoid()
            ]
        }
    }

    # Pridať ďalšie experimenty AND, OR
    for problem, config in [("AND", (X_and, y_and)), ("OR", (X_or, y_or))]:
        for layers_count in [1, 2]:
            if layers_count == 1:
                layers = [
                    LinearLayer(2, 1, learning_rate=0.1, momentum=0.9),
                    Sigmoid()
                ]
            else:
                layers = [
                    LinearLayer(2, 4, learning_rate=0.1, momentum=0.9),
                    Sigmoid(),
                    LinearLayer(4, 1, learning_rate=0.1, momentum=0.9),
                    Sigmoid()
                ]

            experiments[f"{problem} ({layers_count} vrstva, momentum)"] = {
                "X": config[0], "y": config[1], "layers": layers
            }

    # Trénovanie a ukladanie výsledkov
    results = {}
    for name, config in experiments.items():
        print(f"\nTrénovanie: {name}")
        model = NeuralNetwork(config["layers"])
        losses = train_network(config["X"], config["y"], model)
        results[name] = losses

    # Vizualizácia výsledkov
    plot_losses(results, "Priebeh trénovania pre rôzne konfigurácie")