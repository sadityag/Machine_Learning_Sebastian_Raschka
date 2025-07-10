import numpy as np

class Perceptron:
    def __init__(self, learning_rate: float=0.01, iterations: int=20, random_seed=42):
        self.eta = learning_rate
        self.epochs = iterations
        self.seed = random_seed
    
    def predict(self, X):
        return np.where(np.dot(X, self.w_) + self.b_ >= 0, 1, 0)
    
    def train(self, X_train: np.array, y_train: np.array, iterations: int=0):
        random_generator = np.random.RandomState(self.seed)
        self.w_ = random_generator.normal(loc=0.0, scale=0.01, size=X_train.shape[1])
        self.b_ = 0.
        self.errors_ = []
        
        if iterations:
            self.epochs = iterations

        for _ in range(self.epochs):
            predictions = self.predict(X_train)
            err = y_train - predictions
            self.w_ += self.eta * X_train.T.dot(err)
            self.b_ += self.eta * np.sum(err)
            self.errors_.append(np.sum(err ** 2))
        
        return self
            