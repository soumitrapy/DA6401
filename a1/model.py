import numpy as np
import argparse
import wandb

from utils import softmax, Sigmoid, ReLU, Tanh, crossentropyloss
from preprocessing import load_data


class NeuralNetwork:
    def __init__(self,args):# input_size, hidden_layers, output_size, optimizer='sgd', learning_rate=0.01):
        """
        Initialize the neural network with given parameters.
        :param input_size: Number of input neurons (784 for Fashion-MNIST)
        :param hidden_layers: List containing number of neurons in each hidden layer
        :param output_size: Number of output neurons (10 for Fashion-MNIST classes)
        :param optimizer: Optimization algorithm
        :param learning_rate: Learning rate for training
        """
        self.args = args
        self.layers = [args.input_size] + args.hidden_layers + [args.output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.01 for i in range(len(self.layers)-1)]
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)]
        self.learning_rate = args.learning_rate
        self.optimizer = args.optimizer
        self.initialize_optimizer(args)

        ## Activation function ###
        if args.activation_function=='sigmoid':
            self.af = Sigmoid()
        elif args.activation_function=='relu':
            self.af = ReLU()
        elif args.activation_function=='tanh':
            self.af = Tanh()
    
    def initialize_optimizer(self,args):
        """Initialize optimizer parameters."""
        self.velocities = [np.zeros_like(w) for w in self.weights]
        self.squares = [np.zeros_like(w) for w in self.weights]
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.t = 0

        self.gamma = args.gamma
        self.beta = args.beta
        self.beta1 = args.beta1
        self.beta2 = args.beta2

    
    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights) - 1):
            X = self.af(np.dot(X, self.weights[i]) + self.biases[i])
            activations.append(X)
        output = softmax(np.dot(X, self.weights[-1]) + self.biases[-1])
        activations.append(output)
        return activations

    def __call__(self, X):
        return self.forward(X)[-1]
        
    
    def backward(self, X, y, activations):
        """Perform backpropagation and update weights based on selected optimizer."""
        self.t += 1
        y_one_hot = np.eye(10)[y]
        deltas = [activations[-1] - y_one_hot]
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.af.derivative(activations[i])
            deltas.insert(0, delta)
        
        gradients_w = [np.dot(activations[i].T, deltas[i]) for i in range(len(self.weights))]
        gradients_b = [np.sum(deltas[i], axis=0, keepdims=True) for i in range(len(self.biases))]
        
        self.update_weights(gradients_w, gradients_b)
    
    def update_weights(self, gradients_w, gradients_b):
        """Update weights using selected optimization algorithm."""
        if self.optimizer == 'sgd':
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gradients_w[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]
        elif self.optimizer == 'momentum':
            gamma = self.gamma
            for i in range(len(self.weights)):
                self.velocities[i] = gamma * self.velocities[i] + self.learning_rate * gradients_w[i]
                self.weights[i] -= self.velocities[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]
        elif self.optimizer == 'nesterov':
            gamma = self.gamma
            for i in range(len(self.weights)):
                prev_velocity = self.velocities[i]
                self.velocities[i] = gamma * self.velocities[i] + self.learning_rate * gradients_w[i]
                self.weights[i] -= gamma * prev_velocity + (1 + gamma) * self.velocities[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]
        elif self.optimizer == 'rmsprop':
            beta =self.beta
            epsilon = 1e-8
            for i in range(len(self.weights)):
                self.squares[i] = beta * self.squares[i] + (1 - beta) * gradients_w[i]**2
                self.weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.squares[i]) + epsilon)
                self.biases[i] -= self.learning_rate * gradients_b[i]
        elif self.optimizer == 'adam':
            beta1, beta2 = self.beta1, self.beta2
            epsilon = 1e-8
            for i in range(len(self.weights)):
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * gradients_w[i]
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * gradients_w[i]**2
                m_hat = self.m[i] / (1 - beta1**self.t)
                v_hat = self.v[i] / (1 - beta2**self.t)
                self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                self.biases[i] -= self.learning_rate * gradients_b[i]
        elif self.optimizer == 'nadam':
            beta1, beta2 = self.beta1, self.beta2
            epsilon = 1e-8
            for i in range(len(self.weights)):
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * gradients_w[i]
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * gradients_w[i]**2
                m_hat = self.m[i] / (1 - beta1**self.t)
                v_hat = self.v[i] / (1 - beta2**self.t)
                self.weights[i] -= self.learning_rate * (beta1 * m_hat + (1 - beta1) * gradients_w[i] / (1 - beta1**self.t)) / (np.sqrt(v_hat) + epsilon)
                self.biases[i] -= self.learning_rate * gradients_b[i]

# Command-line argument parsing
def get_args():
    parser = argparse.ArgumentParser(description="Train a Feedforward Neural Network on Fashion-MNIST")
    parser.add_argument('--input_size','-in', type=int, default=28*28, help='input size of the model')
    parser.add_argument('--output_size','-out', type=int, default=10, help='output size of the model')
    parser.add_argument('--activation_function','-act', type=str, default='sigmoid', choices=['sigmoid','relu','tanh'], help='activation function')
    parser.add_argument('--hidden_layers','-hid', nargs='+', type=int, default=[128, 64], help='List of hidden layer sizes')
    parser.add_argument('--epochs','-e', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size','-b', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate','-lr', type=float, default=0.01, help='Learning rate for optimization')
    parser.add_argument('--optimizer','-opt', type=str, default='sgd', choices=['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam'], help='Optimization algorithm')
    parser.add_argument('--gamma', type=float, default=0.9, help='optimization hyper-parameter')
    parser.add_argument('--beta', type=float, default=0.99, help='optimization hyper-parameter')
    parser.add_argument('--beta1', type=float, default=0.9, help='optimization hyper-parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='optimization hyper-parameter')
    return parser.parse_args()

def train_model():
    args = get_args()
    nn = NeuralNetwork(args)
    X_train, y_train, X_test, y_test, class_names = load_data()
    X_train, X_test = X_train.reshape(-1, 784) / 255.0, X_test.reshape(-1, 784) / 255.0
    train_size = int(X_train.shape[0]*0.9)
    X_train, X_valid = X_train[:train_size], X_train[train_size:]
    y_train, y_valid = y_train[:train_size], y_train[train_size:]
    
    def train():
        wandb.init()
        config = wandb.config
        nn = NeuralNetwork(args)
        
        for epoch in range(config.epochs):
            activations = nn.forward(X_train)
            nn.backward(X_train, y_train, activations)
            loss = crossentropyloss(y_valid, nn(X_valid))
            wandb.log({"epoch": epoch + 1, "loss": loss})
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "epochs": {"values": [5, 10]},
            "hidden_layers": {"values": [3, 4, 5]},
            "hidden_layer_size": {"values": [32, 64, 128]},
            "weight_decay": {"values": [0, 0.0005, 0.5]},
            "learning_rate": {"values": [1e-3, 1e-4]},
            "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
            "batch_size": {"values": [16, 32, 64]},
            "weight_initialization": {"values": ["random", "Xavier"]},
            "activation_function": {"values": ["sigmoid", "tanh", "ReLU"]}
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="A1_DA6401")
    wandb.agent(sweep_id, train, count=20)




if __name__=="__main__":
    train_model()
