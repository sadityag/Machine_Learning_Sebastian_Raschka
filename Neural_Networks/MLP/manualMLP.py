import numpy as np
from typing import Tuple, Union
import numpy as np

class NeuralMLP:
    def __init__(
        self,
        num_features: int,
        num_hidden: int,
        num_classes: int,
        is_onehot: bool = False,
        rnd_seed: int = 42
    ) -> None:
        """
        Initialize the neural network parameters.

        Args:
            num_features: Number of input features.
            num_hidden: Number of hidden units.
            num_classes: Number of output classes.
            is_onehot: Whether the labels are one-hot encoded.
            rnd_seed: Random seed for reproducibility.
        """
        self.num_classes = num_classes

        rng = np.random.RandomState(rnd_seed)

        # Initialize hidden layer weights and biases
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # Initialize output layer weights and biases
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

        self.is_onehot = is_onehot

    def _sigmoid(self, z: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Sigmoid activation function.

        Args:
            z: Input value or array.

        Returns:
            Activated value(s).
        """
        return 1. / (1. + np.exp(-z))

    def int_to_onehot(self, y: np.ndarray) -> np.ndarray:
        """
        Convert integer labels to one-hot encoded format.

        Args:
            y: Array of integer labels.
            num_labels: Total number of classes.

        Returns:
            One-hot encoded array.
        """
        arr = np.zeros((y.shape[0], self.num_classes))
        for i, val in enumerate(y):
            arr[i, val] = 1
        return arr

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the network.

        Args:
            x: Input data.

        Returns:
            Tuple of hidden activations and output activations.
        """
        z_h = np.dot(x, self.weight_h.T) + self.bias_h  # Hidden layer linear combination
        a_h = self._sigmoid(z_h)                        # Hidden layer activation

        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out  # Output layer linear combination
        a_out = self._sigmoid(z_out)                            # Output layer activation

        return a_h, a_out

    def backward(
        self,
        x: np.ndarray,
        a_h: np.ndarray,
        a_out: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for computing gradients.

        Args:
            x: Input data.
            a_h: Hidden layer activations.
            a_out: Output layer activations.
            y: True labels.

        Returns:
            Gradients for output weights, output biases, hidden weights, hidden biases.
        """
        # Convert labels to one-hot if necessary
        if not self.is_onehot:
            y_onehot = self.int_to_onehot(y, self.num_classes)
        else:
            y_onehot = y

        # Compute gradient of loss w.r.t. output activation
        d_loss__d_a_out = 2. * (a_out - y_onehot) / y.shape[0]
        # Derivative of sigmoid for output layer
        d_a_out__d_z_out = a_out * (1 - a_out)
        # Delta for output layer
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        # Gradients for output weights and biases
        d_z_out__dw_out = a_h
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # Backpropagate to hidden layer
        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        # Derivative of sigmoid for hidden layer
        d_a_h__d_z_h = a_h * (1. - a_h)
        d_z_h__d_w_h = x

        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h)
    
    def mse(self, y_true: np.ndarray, probs: np.ndarray) -> np.float:
        tgt = y_true if self.is_onehot else self.int_to_onehot(y_true)
        return np.mean((tgt - probs) ** 2)
    
    def accuracy(self, y_true: np.ndarray, predicted: np.ndarray) -> np.float:
        return np.mean(predicted == y_true)