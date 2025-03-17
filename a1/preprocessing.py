import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import numpy as np

def load_data():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return x_train, y_train, x_test, y_test, class_names

def plot_samples(x_train, y_train, class_names):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        indices = (y_train == i).nonzero()[0]
        idx = np.random.choice(indices)
        ax = axes[i // 5, i % 5]
        ax.imshow(x_train[idx], cmap='gray')
        ax.set_title(class_names[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    x_train, y_train, x_test, y_test, class_names = load_data()
    plot_samples(x_train, y_train, class_names)
