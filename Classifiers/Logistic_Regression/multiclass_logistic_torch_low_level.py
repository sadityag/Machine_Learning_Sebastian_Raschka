import torch
from typing import Tuple, Self, List

class MulticlassSoftmax:
    """
    Multiclass logistic regression (softmax) classifier implemented using PyTorch tensors.

    Args:
        num_features (int): Number of input features.
        num_classes (int): Number of output classes.
        is_onehot (bool): Whether input labels are already one-hot encoded.
        DEVICE (torch.device): Device to run computations on (CPU or CUDA).
    """
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        is_onehot: bool = False,
        DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ) -> None:
        self.DEVICE = DEVICE
        self.is_onehot = is_onehot
        self.num_features = num_features
        self.num_classes = num_classes
        # Initialize weights and bias to zeros
        self.weights = torch.zeros(num_classes, num_features, dtype=torch.float32, device=self.DEVICE)
        self.bias = torch.zeros(num_classes, dtype=torch.float32, device=self.DEVICE)

    def to_onehot(self, y, num_classes: int) -> torch.FloatTensor:
        """
        Converts integer class labels to one-hot encoded format.

        Args:
            y (torch.Tensor): Class labels (shape: [batch_size]).
            num_classes (int): Number of classes.

        Returns:
            torch.FloatTensor: One-hot encoded labels (shape: [batch_size, num_classes]).
        """
        y_onehot = torch.zeros(y.size(0), num_classes, device=y.device)
        y_onehot.scatter_(1, y.view(-1, 1).long(), 1)
        return y_onehot.float()

    def softmax(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies softmax activation to logits.

        Args:
            z (torch.Tensor): Logits (shape: [batch_size, num_classes]).

        Returns:
            torch.Tensor: Softmax probabilities (shape: [batch_size, num_classes]).
        """
        exp_z = torch.exp(z)
        return exp_z / torch.sum(exp_z, dim=1, keepdim=True)

    def cross_entropy(self, softmax: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        Computes cross-entropy loss.

        Args:
            softmax (torch.Tensor): Predicted probabilities (shape: [batch_size, num_classes]).
            y_target (torch.Tensor): True labels (one-hot, shape: [batch_size, num_classes]).

        Returns:
            torch.Tensor: Cross-entropy loss for each sample (shape: [batch_size]).
        """
        return -torch.sum(torch.log(softmax) * (y_target), dim=1)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes logits and softmax probabilities for input features.

        Args:
            x (torch.Tensor): Input features (shape: [batch_size, num_features]).

        Returns:
            torch.Tensor: Softmax probabilities (shape: [batch_size, num_classes]).
        """
        logits = torch.mm(x, self.weights.t()) + self.bias  # Linear transformation
        probs = self.softmax(logits)  # Softmax activation
        return probs

    def _backprop(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        probs: torch.Tensor,
        learning_rate: float = 0.01
    ) -> Self:
        """
        Performs one step of gradient descent to update weights and bias.

        Args:
            x (torch.Tensor): Input features (shape: [batch_size, num_features]).
            y (torch.Tensor): True labels (one-hot, shape: [batch_size, num_classes]).
            probs (torch.Tensor): Predicted probabilities (shape: [batch_size, num_classes]).
            learning_rate (float): Learning rate for gradient descent.

        Returns:
            Self: Updated model.
        """
        # Gradient of loss w.r.t. weights
        dLoss_dw = -torch.mm(x.t(), y - probs).t()
        # Gradient of loss w.r.t. bias
        dLoss_dbias = -torch.sum(y - probs, dim=0)
        # Update weights and bias
        self.weights -= learning_rate * dLoss_dw / y.size(0)
        self.bias -= learning_rate * dLoss_dbias / y.size(0)
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts class labels for input features.

        Args:
            x (torch.Tensor): Input features (shape: [batch_size, num_features]).

        Returns:
            torch.Tensor: Predicted class labels (shape: [batch_size]).
        """
        probs = self._forward(x)
        labels = torch.argmax(probs, dim=1)
        return labels

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes accuracy of the model.

        Args:
            x (torch.Tensor): Input features (shape: [batch_size, num_features]).
            y (torch.Tensor): True class labels (shape: [batch_size]).

        Returns:
            torch.Tensor: Accuracy (float).
        """
        labels = self.predict(x).float()
        accuracy = torch.sum(labels.view(-1) == y.float()).item() / y.size(0)
        return accuracy

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_epochs: int = 100,
        learning_rate: float = 0.01
    ) -> List[float]:
        """
        Trains the model using gradient descent.

        Args:
            x (torch.Tensor): Input features (shape: [batch_size, num_features]).
            y (torch.Tensor): True class labels (shape: [batch_size]) or one-hot labels.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for gradient descent.

        Returns:
            List[float]: Log loss for each epoch.
        """
        epoch_log_loss = []

        # Convert labels to one-hot if needed
        if not self.is_onehot:
            y_onehot = self.to_onehot(y, num_classes=self.num_classes)
        else:
            y_onehot = y

        for e in range(num_epochs):
            # Forward pass
            probs = self._forward(x)
            # Backpropagation and update
            updated_probs = self._backprop(x, y_onehot, probs, learning_rate)._forward(x)
            # Compute log loss
            log_loss = torch.mean(self.cross_entropy(updated_probs, y_onehot))
            epoch_log_loss.append(log_loss.item())

            # Print training progress
            print('Epoch: %03d' % (e + 1), end="")
            print(' | Train ACC: %.3f' % self.evaluate(x, y), end="")
            print(' | Cost: %.3f' % log_loss)

        return epoch_log_loss