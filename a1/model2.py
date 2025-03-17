import numpy as np
import argparse

from utils import softmax, Sigmoid, ReLU, Tanh
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
        self.layers = [args.input_size] + args.hidden_layers + [args.output_size]
        #self.params = [(np.random.randn(self.layers[i], self.layers[i+1])*0.01, np.zeros((1, self.layers[i+1]))) for i in range(len(self.layers)-1)]
        self.W = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.01 for i in range(len(self.layers)-1)]
        self.b = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)]
        
        self.dw = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]
        
        ## Activation function ###
        if args.activation_function=='sigmoid':
            self.af = Sigmoid()
        elif args.activation_function=='relu':
            self.af = ReLU()
        elif args.activation_function=='tanh':
            self.af = Tanh()
        
    
    def forward(self, X):
        self.Z = []
        self.A = [X]
        a = X
        for i in range(len(self.W) - 1):
            z = np.dot(a, self.W[i]) + self.b[i]
            a = self.af(z)
            self.Z.append(z)
            self.A.append(a)

        z = np.dot(a, self.W[-1]) + self.b[-1]
        a = softmax(z)
        self.Z.append(z)
        self.A.append(a)
        return a


        
    
    def backward(self, X, y, A=None):
        if A is None:
            A = self.A
        y_one_hot = np.eye(10)[y]
        deltas = [self.A[-1] - y_one_hot]
        for i in range(len(self.W) - 1, 0, -1):
            delta = np.dot(deltas[0], self.W[i].T) * self.af.derivative(self.A[i])
            deltas.insert(0, delta)
        
        gradients_w = [np.dot(self.A[i].T, deltas[i]) for i in range(len(self.W))]    
        gradients_b = [np.sum(deltas[i], axis=0, keepdims=True) for i in range(len(self.b))]
        self.update_weights(gradients_w, gradients_b)
    
    def update_weights(self, gradients_w, gradients_b):
        if self.optimizer == 'sgd':
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gradients_w[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]
        elif self.optimizer == 'momentum':
            gamma = 0.9
            for i in range(len(self.weights)):
                self.velocities[i] = gamma * self.velocities[i] + self.learning_rate * gradients_w[i]
                self.weights[i] -= self.velocities[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]
        elif self.optimizer == 'nesterov':
            gamma = 0.9
            for i in range(len(self.weights)):
                prev_velocity = self.velocities[i]
                self.velocities[i] = gamma * self.velocities[i] + self.learning_rate * gradients_w[i]
                self.weights[i] -= gamma * prev_velocity + (1 + gamma) * self.velocities[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]
        elif self.optimizer == 'rmsprop':
            beta = 0.99
            epsilon = 1e-8
            for i in range(len(self.weights)):
                self.squares[i] = beta * self.squares[i] + (1 - beta) * gradients_w[i]**2
                self.weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.squares[i]) + epsilon)
                self.biases[i] -= self.learning_rate * gradients_b[i]
        elif self.optimizer == 'adam':
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
            for i in range(len(self.weights)):
                self.m[i] = beta1 * self.m[i] + (1 - beta1) * gradients_w[i]
                self.v[i] = beta2 * self.v[i] + (1 - beta2) * gradients_w[i]**2
                m_hat = self.m[i] / (1 - beta1**self.t)
                v_hat = self.v[i] / (1 - beta2**self.t)
                self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                self.biases[i] -= self.learning_rate * gradients_b[i]
        elif self.optimizer == 'nadam':
            beta1, beta2 = 0.9, 0.999
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
    return parser.parse_args()

        
if __name__=="__main__":
    args = get_args()
    nn = NeuralNetwork(args)
    X_train, y_train, X_test, y_test, class_names = load_data()
    X_train, X_test = X_train.reshape(-1, 784) / 255.0, X_test.reshape(-1, 784) / 255.0
    X_batch = X_train[:args.batch_size]
    y_batch = y_train[:args.batch_size]
    y_pred = nn.forward(X_batch)
    nn.backward(X_batch,y_batch)