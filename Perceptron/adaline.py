# ADAptive LInear NEural classifier - ADALINE
import numpy as np

class ADALINE:
    def __init__(self, learning_rate: float=0.01, iterations: int=100, annealing:bool=True, annealing_rate=0.96, random_seed=42) -> None:
        """
        Initialize ADALINE classifier.

        Parameters:
        learning_rate (float): Initial learning rate (eta).
        iterations (int): Number of training epochs.
        annealing (bool): Whether to use learning rate annealing.
        annealing_rate (float): Annealing decay rate per epoch.
        random_seed (int): Seed for random number generator.
        """
        self.eta = learning_rate
        self.epochs = iterations
        self.seed = random_seed
        self.w_initialized = False
        self.b_initialized = False
        self.rgen = np.random.RandomState(self.seed)
        self.annealing = annealing
        self.annealing_rate = annealing_rate
    
    def initialize_weights(self, dimension: int, reinitialize: bool=False) -> None:
        """
        Initialize weights and bias.

        Parameters:
        dimension (int): Number of features.
        reinitialize (bool): Force reinitialization if True.
        """
        if reinitialize or not self.w_initialized:
            self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=dimension)
            self.w_initialized = True

        if reinitialize or not self.b_initialized:
            self.b_ = 0.0
            self.b_initialized = True

    def _shuffle(self, X_train, y_train):
        """
        Shuffle training data.

        Parameters:
        X_train: Feature matrix (numpy array or pandas DataFrame).
        y_train: Target vector (numpy array or pandas Series).

        Returns:
        Shuffled X_train and y_train as numpy arrays.
        """
        # Convert pandas to numpy if needed
        if hasattr(X_train, 'values'):  # pandas DataFrame
            X_train = X_train.values
        if hasattr(y_train, 'values'):  # pandas Series
            y_train = y_train.values
    
        perm = self.rgen.permutation(len(y_train))
        return X_train[perm], y_train[perm]
        
    def net_input(self, X: np.array) -> np.array:
        """
        Calculate net input (weighted sum plus bias).

        Parameters:
        X (np.array): Feature matrix.

        Returns:
        Net input as numpy array.
        """
        return np.dot(X, self.w_) + self.b_ 

    def predict(self, X: np.array) -> np.array:
        """
        Predict class labels for samples in X.

        Parameters:
        X (np.array): Feature matrix.

        Returns:
        Predicted class labels (0 or 1).
        """
        return np.where(self.net_input(X) >= 0.5, 1, 0)
    
    def activation(self, x):
        """
        Linear activation function (identity).

        Parameters:
        x: Input value.

        Returns:
        Output value (same as input).
        """
        return x
    
    def _update_weights(self, X_train, target):
        """
        Update weights and bias using gradient descent.

        Parameters:
        X_train: Feature matrix (batch).
        target: Target values (batch).

        Returns:
        Mean squared error for the batch.
        """
        X_train = np.atleast_2d(X_train)
        target = np.atleast_1d(target)
        
        output = self.activation(self.net_input(X_train))
        error = target - output
        
        # Consistent gradient calculation regardless of batch size
        gradient = X_train.T.dot(error) / X_train.shape[0]  # Average over batch
        
        self.w_ += self.eta * 2.0 * gradient
        self.b_ += self.eta * 2.0 * np.mean(error)
        
        return np.mean(error**2)
    
    def run_batch_epoch(self, X_train: np.array, y_train: np.array) -> None:
        """
        Run one epoch of batch gradient descent.

        Parameters:
        X_train (np.array): Feature matrix.
        y_train (np.array): Target vector.

        Returns:
        self
        """
        self.losses.append(self._update_weights(X_train, y_train))
        return self

    def run_sgd_epoch(self, X_train: np.array, y_train: np.array, batch_size: int=1):
        """
        Run one epoch of stochastic gradient descent.

        Parameters:
        X_train (np.array): Feature matrix.
        y_train (np.array): Target vector.
        batch_size (int): Size of mini-batches.

        Returns:
        self
        """
        X, y = self._shuffle(X_train, y_train)
        
        epoch_losses = []
        for row in range(0, len(y), batch_size):
            # Fixed: slicing syntax
            batch_X = X[row:row+batch_size]
            batch_y = y[row:row+batch_size]
            epoch_losses.append(self._update_weights(batch_X, batch_y))
        self.losses.append(np.mean(epoch_losses))
        return self

    def sgd_train(self, X_train: np.array, y_train: np.array, learning_rate: float=0.0, iterations: int=0, batch_size: int=1):
        """
        Train model using stochastic gradient descent.

        Parameters:
        X_train (np.array): Feature matrix.
        y_train (np.array): Target vector.
        learning_rate (float): Learning rate (optional).
        iterations (int): Number of epochs (optional).
        batch_size (int): Size of mini-batches.

        Returns:
        self
        """
        self.initialize_weights(X_train.shape[1])
        self.losses = []
        
        if iterations:
            self.epochs = iterations
        
        if learning_rate > 0.0:
            self.eta = learning_rate
        
        lr = self.eta
        
        for e in range(self.epochs):
            if self.annealing:
                self.eta *= (self.annealing_rate ** e)
            self.run_sgd_epoch(X_train, y_train, batch_size)
        
        self.eta = lr

        return self
    
    def train(self, X_train: np.array, y_train: np.array, learning_rate: float=0.0, iterations: int=0):
        """
        Train model using batch gradient descent.

        Parameters:
        X_train (np.array): Feature matrix.
        y_train (np.array): Target vector.
        learning_rate (float): Learning rate (optional).
        iterations (int): Number of epochs (optional).

        Returns:
        self
        """
        self.initialize_weights(X_train.shape[1])
        self.losses = []
        
        if iterations:
            self.epochs = iterations
        
        if learning_rate > 0.0:
            self.eta = learning_rate
        
        for _ in range(self.epochs):
            self.run_batch_epoch(X_train, y_train)
        
        return self