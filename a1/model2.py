import numpy as np
from utils import *

class NeuralNetwork:
    def __init__(self,
                 input_size=28*28,
                 output_size=10,
                 hidden_layers=[64],
                 activation = 'relu',
                 ):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.01 for i in range(len(self.layers)-1)]
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)]
        self.relu = ReLU()
        self.softmax = Softmax()

    # def softmax(self, z):
    #     exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    #     return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # def relu(self, z):
    #     return np.maximum(0, z)
    
    # def relu_derivative(self, z):
    #     return (z > 0).astype(float)
    
    def forward(self, X):
        activations = [X]
        zs = []
        
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(activations[-1], w) + b
            zs.append(z)
            activations.append(self.relu(z))
        
        final_z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(final_z)
        activations.append(self.softmax(final_z))
        
        return activations, zs
    
    def backward(self, X, Y, activations, zs, learning_rate):
        m = X.shape[0]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        
        dL_dz = activations[-1] - Y  # Cross-entropy loss gradient
        
        for i in reversed(range(len(self.weights))):
            grads_w[i] = np.dot(activations[i].T, dL_dz) / m
            grads_b[i] = np.sum(dL_dz, axis=0, keepdims=True) / m 
            
            if i > 0:
                dL_dz = np.dot(dL_dz, self.weights[i].T) * self.relu_derivative(zs[i-1])
        
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]
    
    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            activations, zs = self.forward(X)
            self.backward(X, Y, activations, zs, learning_rate)
            
            if epoch % 10 == 0:
                loss = -np.sum(Y * np.log(activations[-1] + 1e-9)) / X.shape[0]
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        print("Training complete.")


if __name__=="__main__":
    # x_train, y_train, x_test, y_test, class_names = load_data()
    # print(x_train.shape)
    m = NeuralNetwork(28*28, 10,[3,14])
    x = np.random.randn(2, 28*28)
    out = m(x)
    print(out.shape)
