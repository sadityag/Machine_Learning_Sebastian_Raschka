import numpy as np

class Perceptron:
    """
    Perceptron classifier.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size for weight updates.
    iterations : int, default=20
        Number of passes over the training dataset.
    random_seed : int, default=42
        Seed for random number generator for reproducibility.

    Attributes
    ----------
    w_ : ndarray
        Weights after fitting.
    b_ : float
        Bias unit after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    """

    def __init__(self, learning_rate: float=0.01, iterations: int=20, random_seed=42):
        self.eta = learning_rate  # Learning rate
        self.epochs = iterations  # Number of training epochs
        self.seed = random_seed   # Random seed for reproducibility
    
    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray, shape = [n_samples, n_features]
            Input samples.

        Returns
        -------
        labels : ndarray
            Predicted class labels.
        """
        # Compute linear combination and apply unit step function
        return np.where(np.dot(X, self.w_) + self.b_ >= 0, 1, 0)
    
    def train(self, X_train: np.array, y_train: np.array, iterations: int=0):
        """
        Fit training data.

        Parameters
        ----------
        X_train : ndarray, shape = [n_samples, n_features]
            Training vectors.
        y_train : ndarray, shape = [n_samples]
            Target values.
        iterations : int, optional
            Number of epochs to train. Overrides self.epochs if provided.

        Returns
        -------
        self : object
        """
        # Initialize random generator and weights
        random_generator = np.random.RandomState(self.seed)
        self.w_ = random_generator.normal(loc=0.0, scale=0.01, size=X_train.shape[1])  # Small random weights
        self.b_ = 0.  # Bias term
        self.errors_ = []  # Track errors in each epoch
        
        if iterations:
            self.epochs = iterations  # Override epochs if specified

        for _ in range(self.epochs):
            predictions = self.predict(X_train)  # Predict output
            err = y_train - predictions          # Calculate error
            self.w_ += self.eta * X_train.T.dot(err)  # Update weights
            self.b_ += self.eta * np.sum(err)         # Update bias
            self.errors_.append(np.sum(err ** 2))     # Store squared error
        
        return self