import torch
import torch.nn.functional as F
from typing import Self, List

class MulticlassSoftmaxTorch(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int, DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        super(MulticlassSoftmaxTorch, self).__init__()
        self.DEVICE = DEVICE
        self.linear = torch.nn.Linear(num_features, num_classes)
        self.num_classes = num_classes  # Missing
        self.is_onehot = False  # Missing, or make it a parameter
        self.reset_weights()
        
    def reset_weights(self) -> Self:
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()
        return self
    
    def to_onehot(self, y: torch.Tensor, num_classes: int) -> torch.Tensor:
        y_onehot = torch.zeros(y.size(0), num_classes, device=y.device)
        y_onehot.scatter_(1, y.view(-1, 1).long(), 1)
        return y_onehot.float()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.linear(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
    
    def evaluate(self, predictions: torch.Tensor, truth_labels: torch.Tensor) -> float:
        accuracy = torch.sum(truth_labels.view(-1) == predictions.float()).item() / truth_labels.size(0)
        return accuracy
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits, probs = self.forward(x)
            return torch.argmax(probs, dim=1)
    
    def train(self, x: torch.Tensor, y: torch.Tensor, num_epochs: int=100, learning_rate: float=0.01, is_onehot: bool=False) -> List[float]:
        epoch_log_loss = []
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        y.to(self.DEVICE)
        x.to(self.DEVICE)

        if not is_onehot:
            y_onehot = self.to_onehot(y, num_classes=self.num_classes)
        else:
            y_onehot = y
        
        for e in range(num_epochs):
            optimizer.zero_grad()

            logits, probs = self.forward(x)

            loss = F.cross_entropy(logits, torch.argmax(y_onehot, dim=1))
            
            loss.backward()

            optimizer.step()
            
            epoch_log_loss.append(loss.item())

            if (e + 1) % 10 == 0 or e == 0:
                predictions = self.predict(x)  # Get predictions first
                acc = self.evaluate(predictions, torch.argmax(y_onehot, dim=1))
                print(f'Epoch: {e+1:03d} | Train ACC: {acc:.3f} | Loss: {loss.item():.3f}')
        
        return epoch_log_loss