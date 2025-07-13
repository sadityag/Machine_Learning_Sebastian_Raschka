import numpy as np
from typing import Self

class LogisticRegression:
    """
    Logistic Regression classifier using batch gradient descent or stochastic gradient descent (SGD).
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
        Initialize the LogisticRegression model.

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

    def activation(self, z: np.ndarray | float):
        """
        Sigmoid activation function.

        Parameters:
            z (np.ndarray | float): Input value(s).

        Returns:
            np.ndarray | float: Activated output.
        """
        # Clip input to avoid overflow in exp
        return 1 / (1 + np.exp(- np.clip(z, -250, 250)))

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the linear net input.

        Parameters:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Net input (X dot w).
        """
        # Uses the convention of w_0 = bias
        return np.dot(X, self.w_)

    def compute_posterior(self, activated, truth_label):
        """
        Compute posterior probability for a given label.

        Parameters:
            activated: Output of activation function.
            truth_label: True label (0 or 1).

        Returns:
            float: Posterior probability.
        """
        return activated if truth_label else 1 - activated

    def _log_loss(self, X: np.ndarray, y: np.ndarray):
        """
        Compute log loss for predictions.

        Parameters:
            X (np.ndarray): Input features.
            y (np.ndarray): True labels.

        Returns:
            float: Log loss value.
        """
        hypotheses = self.activation(self.net_input(X))
        return - (y.dot(np.log(hypotheses)) + (1 - y).dot(np.log(1 - hypotheses)))

    def _output_log_loss(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute log loss given output and true labels.

        Parameters:
            output (np.ndarray): Model output.
            y (np.ndarray): True labels.

        Returns:
            float: Log loss value.
        """
        return - (y.dot(np.log(output)) + (1 - y).dot(np.log(1 - output)))

    def _loss(self, X: np.ndarray, y: np.ndarray):
        """
        Compute exponential of log loss (not standard).

        Parameters:
            X (np.ndarray): Input features.
            y (np.ndarray): True labels.

        Returns:
            float: Exponential log loss.
        """
        return np.exp(self._log_loss(X, y))

    def _concat_ones_col(self, X_train: np.ndarray) -> np.ndarray:
        """
        Concatenate a column of ones to input features for bias term.

        Parameters:
            X_train (np.ndarray): Input features.

        Returns:
            np.ndarray: Features with bias column.
        """
        return np.concatenate((np.ones(shape=(X_train.shape[0], 1)), X_train), axis=1)

    def _initialize_weights(self, dimension: int, reinitialize: bool = False) -> None:
        """
        Initialize weights randomly.

        Parameters:
            dimension (int): Number of weights (features + bias).
            reinitialize (bool): Whether to reinitialize weights.
        """
        if reinitialize or not self.w_initialized:
            self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=dimension)  # Small random weights
            self.w_initialized = True

    def _shuffle(self, X_train, y_train) -> tuple[np.ndarray, np.ndarray]:
        """
        Shuffle training data.

        Parameters:
            X_train: Input features (numpy or pandas).
            y_train: Labels (numpy or pandas).

        Returns:
            tuple: Shuffled features and labels.
        """
        # Convert pandas to numpy if needed
        if hasattr(X_train, 'values'):  # pandas DataFrame
            X_train = X_train.values
        if hasattr(y_train, 'values'):  # pandas Series
            y_train = y_train.values

        perm = self.rgen.permutation(len(y_train))  # Generate random permutation
        return X_train[perm], y_train[perm]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for test data.

        Parameters:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted labels (0 or 1).
        """
        if not self.w_initialized:
            raise Exception("Your weights have not been initialized.")

        X = self._concat_ones_col(X_test)
        # Predict 1 if net input >= 0, else 0
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def _update_weights(self, X: np.ndarray, target: np.ndarray) -> float:
        """
        Update weights using gradient descent for a batch.

        Parameters:
            X (np.ndarray): Batch features.
            target (np.ndarray): Batch labels.

        Returns:
            float: Log loss for the batch.
        """
        X = np.atleast_2d(X)
        target = np.atleast_1d(target)

        output = self.activation(self.net_input(X))  # Linear output
        error = target - output  # Error vector

        # Gradient calculation (average over batch)
        gradient = X.T.dot(error) / X.shape[0]

        # Update weights and bias
        self.w_ += self.eta * 2.0 * gradient

        return float(self._output_log_loss(output, target))

    def _run_batch_epoch(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Run one epoch of batch gradient descent.

        Parameters:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.

        Returns:
            Self: The model instance.
        """
        self.losses_.append(self._update_weights(X, y))
        return self

    def _run_sgd_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 1):
        """
        Run one epoch of stochastic gradient descent.

        Parameters:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
            batch_size (int): Size of mini-batches.

        Returns:
            Self: The model instance.
        """
        X_shuffle, y_shuffle = self._shuffle(X, y)  # Shuffle data

        epoch_losses = []
        for row in range(0, len(y), batch_size):
            batch_X = X_shuffle[row:row + batch_size]
            batch_y = y_shuffle[row:row + batch_size]
            epoch_losses.append(self._update_weights(batch_X, batch_y))
        self.losses_.append(np.mean(epoch_losses))  # Average loss for epoch
        return self

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        iterations: int = 0,
        learning_rate: float = 0.0,
        sgd: bool = False,
        batch_size: int = 1,
        reinitialize_losses=True
    ) -> Self:
        """
        Fit the logistic regression model to training data.

        Parameters:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            iterations (int): Number of epochs (overrides constructor).
            learning_rate (float): Learning rate (overrides constructor).
            sgd (bool): Use SGD if True, else batch gradient descent.
            batch_size (int): Mini-batch size for SGD.
            reinitialize_losses (bool): Reset loss history.

        Returns:
            Self: The fitted model instance.
        """
        X = self._concat_ones_col(X_train)
        self._initialize_weights(X.shape[1])
        if reinitialize_losses:
            self.losses_ = []

        if iterations:
            self.epochs = iterations

        if learning_rate > 0.0:
            self.eta = learning_rate

        if not sgd:
            # Batch gradient descent
            for _ in range(self.epochs):
                self._run_batch_epoch(X, y_train)
        else:
            # Stochastic gradient descent
            lr = self.eta  # Store initial learning rate

            for e in range(self.epochs):
                if self.annealing:
                    self.eta *= (self.annealing_rate ** e)  # Anneal learning rate
                self._run_sgd_epoch(X, y_train, batch_size)

            self.eta = lr  # Restore initial learning rate

        return self