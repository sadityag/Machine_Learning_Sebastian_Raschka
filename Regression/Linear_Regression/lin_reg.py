import numpy as np
from typing import Self

class LinearRegressionGD:
    """
    Linear Regression using batch gradient descent or stochastic gradient descent (SGD).
    Supports learning rate annealing and random weight initialization.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        iterations: int = 50,
        annealing: bool = True,
        annealing_rate: float = 0.96,
        seed: int = 42
    ) -> None:
        """
        Initialize the LinearRegressionGD model.

        Parameters:
            learning_rate (float): Initial learning rate for gradient descent.
            iterations (int): Number of training epochs.
            annealing (bool): Whether to use learning rate annealing.
            annealing_rate (float): Annealing rate for learning rate decay.
            seed (int): Random seed for reproducibility.
        """
        self.rgen = np.random.RandomState(seed)  # Random generator for reproducibility
        self.w_initialized = False  # Flag to check if weights are initialized
        self.losses_ = []  # List to store loss values per epoch

        self.eta = learning_rate  # Learning rate
        self.annealing = annealing  # Flag for annealing
        self.annealing_rate = annealing_rate  # Annealing rate
        self.epochs = iterations  # Number of epochs

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the linear net input (hypothesis function).

        Parameters:
            X (np.ndarray): Input features (with bias column).

        Returns:
            np.ndarray: Linear predictions (X dot w).
        """
        return np.dot(X, self.w_)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mean squared error for predictions.

        Parameters:
            X (np.ndarray): Input features.
            y (np.ndarray): True target values.

        Returns:
            float: Mean squared error value.
        """
        predictions = self.net_input(X)
        return np.mean((y - predictions) ** 2)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the loss function (MSE).

        Parameters:
            X (np.ndarray): Input features.
            y (np.ndarray): True target values.

        Returns:
            float: Loss value.
        """
        return self.mse(X, y)

    def _concat_ones_col(self, X_train: np.ndarray) -> np.ndarray:
        """
        Concatenate a column of ones to input features for bias term.

        Parameters:
            X_train (np.ndarray): Input features.

        Returns:
            np.ndarray: Features with bias column prepended.
        """
        return np.concatenate((np.ones(shape=(X_train.shape[0], 1)), X_train), axis=1)

    def _initialize_weights(self, dimension: int, reinitialize: bool = False) -> None:
        """
        Initialize weights randomly using normal distribution.

        Parameters:
            dimension (int): Number of weights (features + bias).
            reinitialize (bool): Whether to reinitialize weights if already initialized.
        """
        if reinitialize or not self.w_initialized:
            self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=dimension)
            self.w_[0] = 0
            self.w_initialized = True

    def _shuffle(self, X_train, y_train) -> tuple[np.ndarray, np.ndarray]:
        """
        Shuffle training data for stochastic gradient descent.

        Parameters:
            X_train: Input features (numpy array or pandas DataFrame).
            y_train: Target values (numpy array or pandas Series).

        Returns:
            tuple: Shuffled features and target values as numpy arrays.
        """
        # Convert pandas to numpy if needed
        if hasattr(X_train, 'values'):  # pandas DataFrame
            X_train = X_train.values
        if hasattr(y_train, 'values'):  # pandas Series
            y_train = y_train.values

        perm = self.rgen.permutation(len(y_train))
        return X_train[perm], y_train[perm]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.

        Parameters:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted continuous values.
        """
        if not self.w_initialized:
            raise Exception("Weights have not been initialized. Call fit() first.")

        X = self._concat_ones_col(X_test)
        return self.net_input(X)

    def _update_weights(self, X: np.ndarray, target: np.ndarray) -> float:
        """
        Update weights using gradient descent for a batch.

        Parameters:
            X (np.ndarray): Batch features (with bias column).
            target (np.ndarray): Batch target values.

        Returns:
            float: Mean squared error for the batch.
        """
        X = np.atleast_2d(X)
        target = np.atleast_1d(target)

        predictions = self.net_input(X)  # Linear predictions
        error = target - predictions  # Error vector

        # Gradient calculation: -2/n * X^T * (y - y_hat)
        # We use positive gradient because we're doing gradient ascent on the negative loss
        gradient = X.T.dot(error) / X.shape[0]

        # Update weights
        self.w_ += self.eta * gradient

        return np.mean(error ** 2)  # Return MSE for this batch

    def _run_batch_epoch(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Run one epoch of batch gradient descent.

        Parameters:
            X (np.ndarray): Features (with bias column).
            y (np.ndarray): Target values.

        Returns:
            Self: The model instance for method chaining.
        """
        loss = self._update_weights(X, y)
        self.losses_.append(loss)
        return self

    def _run_sgd_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 1) -> Self:
        """
        Run one epoch of stochastic gradient descent.

        Parameters:
            X (np.ndarray): Features (with bias column).
            y (np.ndarray): Target values.
            batch_size (int): Size of mini-batches.

        Returns:
            Self: The model instance for method chaining.
        """
        X_shuffle, y_shuffle = self._shuffle(X, y)

        epoch_losses = []
        for row in range(0, len(y), batch_size):
            batch_X = X_shuffle[row:row + batch_size]
            batch_y = y_shuffle[row:row + batch_size]
            batch_loss = self._update_weights(batch_X, batch_y)
            epoch_losses.append(batch_loss)
        
        self.losses_.append(np.mean(epoch_losses))
        return self

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        iterations: int = 0,
        learning_rate: float = 0.0,
        sgd: bool = False,
        batch_size: int = 1,
        reinitialize_losses: bool = True
    ) -> Self:
        """
        Fit the linear regression model to training data.

        Parameters:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target values.
            iterations (int): Number of epochs (overrides constructor value if > 0).
            learning_rate (float): Learning rate (overrides constructor value if > 0).
            sgd (bool): Use SGD if True, else batch gradient descent.
            batch_size (int): Mini-batch size for SGD.
            reinitialize_losses (bool): Reset loss history before training.

        Returns:
            Self: The fitted model instance for method chaining.
        """
        X = self._concat_ones_col(X_train)
        self._initialize_weights(X.shape[1])
        
        if reinitialize_losses:
            self.losses_ = []

        if iterations > 0:
            self.epochs = iterations

        if learning_rate > 0.0:
            self.eta = learning_rate

        if not sgd:
            # Batch gradient descent
            for _ in range(self.epochs):
                self._run_batch_epoch(X, y_train)
        else:
            # Stochastic gradient descent
            initial_lr = self.eta  # Store initial learning rate

            for epoch in range(self.epochs):
                if self.annealing:
                    self.eta = initial_lr * (self.annealing_rate ** epoch)
                self._run_sgd_epoch(X, y_train, batch_size)

            self.eta = initial_lr  # Restore initial learning rate

        return self