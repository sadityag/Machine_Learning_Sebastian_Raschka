# ADAptive LInear NEural classifier - ADALINE
import numpy as np

class ADALINE:
    def __init__(self, learning_rate: float=0.01, iterations: int=100, random_seed=42):
        self.eta = learning_rate
        self.epochs = iterations
        self.seed = random_seed
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_ 

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, 0)
    
    def activation(self, x):
        return x
    
    def train(self, X_train: np.array, y_train: np.array, learning_rate: float=0.0, iterations: int=0):
        random_generator = np.random.RandomState(self.seed)
        self.w_ = random_generator.normal(loc=0.0, scale=0.01, size=X_train.shape[1])
        self.b_ = 0.
        self.losses = []
        
        if iterations:
            self.epochs = iterations
        
        if learning_rate > 0.0:
            self.eta = learning_rate

        for _ in range(self.epochs):
            outputs = self.activation(self.net_input(X_train))
            err = y_train - outputs
            self.w_ += self.eta * 2.0 * X_train.T.dot(err) / X_train.shape[0]
            self.b_ += self.eta * 2.0 * err.mean()
            self.losses.append(np.mean(err ** 2))
        
        return self
            