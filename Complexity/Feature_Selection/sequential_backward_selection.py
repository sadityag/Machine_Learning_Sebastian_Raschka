from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from typing import Self, List

class SequentialBackwardSelection:
    def __init__(self, model, k_features: int=3, scoring=accuracy_score, test_size=0.3, random_state=42) -> None:
        self.k = k_features
        self.mod = clone(model)
        self.scoring = scoring
        self.test_size = test_size
        self.seed = random_state
    
    def transform(self, X: np.ndarray, indices: List[int]=None) -> np.ndarray:
        if indices is None:
            try:
                indices = self.best_indices_
            except AttributeError:
                print("You need to call fit before transforming.")
                return
        return X[:, indices]
    
    def _calculate_score(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, indices: List[int]) -> float:
        self.mod.fit(X_train[:, indices], y_train)
        pred = self.mod.predict(X_test[:, indices])
        return self.scoring(y_test, pred)

    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.seed, test_size=self.test_size)

        num_features = X.shape[1]
        self.best_indices_ = list(range(num_features))
        score = self._calculate_score(X_train, X_test, y_train, y_test, self.best_indices_)

        self.subsets_ = [self.best_indices_]
        self.scores_ = [score]

        while num_features > self.k:
            scores = []
            index_subsets = []

            for feature_indices in combinations(self.best_indices_, r=len(self.best_indices_)-1):
                scores.append(self._calculate_score(X_train, X_test, y_train, y_test, feature_indices))
                index_subsets.append(feature_indices)
            
            best_combo_idx= np.argmax(scores)

            self.best_indices_ = list(index_subsets[best_combo_idx])
            self.subsets_.append(self.best_indices_)
            self.scores_.append(scores[best_combo_idx])

            num_features -= 1
        
        self.k_score_ = self.scores_[-1]

        return self
    
    def fit_transform(self, X, y) -> np.ndarray:
        return self.fit(X=X, y=y).transform(X)