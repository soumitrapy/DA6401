from utils import *
from preprocessing import *
from model import *


# Command-line argument parsing
def get_args():
    parser = argparse.ArgumentParser(description="Train a Feedforward Neural Network on Fashion-MNIST")
    parser.add_argument('--input_size','-in', type=int, default=28*28, help='input size of the model')
    parser.add_argument('--output_size','-out', type=int, default=10, help='output size of the model')
    parser.add_argument('--hidden_layers','-hid', nargs='+', type=int, default=[128, 64], help='List of hidden layer sizes')
    parser.add_argument('--epochs','-e', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size','-b', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate','-lr', type=float, default=0.01, help='Learning rate for optimization')
    parser.add_argument('--optimizer','-opt', type=str, default='sgd', choices=['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam'], help='Optimization algorithm')
    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    nn = NeuralNetwork(args)