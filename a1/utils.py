import numpy as np
class Sigmoid:
    def __call__(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def derivative(self, z):
        return self.__call__(z)*(1-self.__call__(z))
    
class ReLU:
    def __call__(self, z):
        return z*(z>0)
    
    def derivative(self, z):
        return (z>0).astype(float)

class Tanh:
    def __call__(self, z):
        return np.tanh(z)
    
    def derivative(self, z):
        return 1-np.tanh(z)**2

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def crossentropyloss(y, yhat):
    '''
    y: (n_samples, k)
    '''
    y_one_hot = np.eye(10)[y]
    eps = 1e-10
    p = np.log(yhat+eps)
    

    return np.sum(y_one_hot*p,-1)

if __name__=="__main__":
    y = np.eye(3)
    y = np.vstack([y, [0.0,0.0,1.0]])
    yhat = np.array([[0.5,0.25,0.25],
                     [0.5,0.5,0.0],
                     [0.0,0.25,0.75],
                     [0.0,0.20,0.80]])
    #print(crossentropyloss(y, yhat))
    
    