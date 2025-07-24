from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators

import numpy as np

import operator

from typing import Self, Any

class ClassifierMajority(BaseEstimator, ClassifierMixin):
    """
    An ensemble classifier that combines multiple classifiers using majority voting or probability averaging.

    Parameters
    ----------
    classifiers : list
        List of classifiers to be ensembled.
    vote : str, default='classlabel'
        Voting strategy: 'classlabel' for majority vote, 'probability' for averaging predicted probabilities.
    weights : list or None, optional
        List of weights for classifiers. If None, uniform weights are used.
    """
    def __init__(self, classifiers, vote: str='classlabel', weights=None) -> Self:
        self.classifiers = classifiers
        # Create a mapping of classifier names to classifier objects
        self.named_classifiers = {
            key: value for key,
            value in _name_estimators(classifiers)
        }
        # Validate voting strategy
        if vote not in ('probability', 'classlabel'):
            raise ValueError(f"vote must be 'probability' or 'classlabel'; got (vote={vote})")
        self.vote = vote

        # Validate weights
        if weights and len(weights) != len(classifiers):
            raise ValueError(f"The number of weights given do not match the number of classifiers: got {len(classifiers)} classifiers and {len(weights)} weights.")
        self.weights = weights
    
    def fit(self, X, y) -> Self:
        """
        Fit the ensemble of classifiers.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        # Encode class labels
        self.lablenc_ = LabelEncoder().fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        y_transformed = self.lablenc_.transform(y)
        # Clone and fit each classifier
        for clf in self.classifiers:
            self.classifiers_.append(clone(clf).fit(X, y_transformed))
        
        return self
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        # Collect probabilities from each classifier
        probs = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        # Average probabilities, weighted if specified
        return np.average(probs, axis=0, weights=self.weights)
    
    def class_predictions(self, X) -> np.ndarray:
        """
        Get class predictions from each classifier.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        predictions : array-like, shape = [n_samples, n_classifiers]
        """
        # Each row: predictions for a sample from all classifiers
        return np.asarray([clf.predict(X) for clf in self.classifiers_]).T

    
    def predict(self, X) -> Any:
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        labels : array-like, shape = [n_samples]
            Predicted class labels.
        """
        if self.vote == 'probability':
            # Use probability averaging
            majority_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            # Use class label voting
            predictions = self.class_predictions(X)
            # One-hot encode predictions for voting
            majority_vote = np.argmax(np.average(np.eye(len(self.classes_))[predictions], weights=self.weights, axis=1), axis=1)
        
        # Decode labels back to original encoding
        return self.lablenc_.inverse_transform(majority_vote)
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        if deep:
            # Get name and copy of classifiers
            out = self.named_classifiers.copy()
            # Iterate over them
            for name, step in self.named_classifiers.items():
                # For each classifier, grab each parameter and value, and associate with the name
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value
            return out
        else:
            return super().get_params(deep=False)
        
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__() if hasattr(super(), '__sklearn_tags__') else type('Tags', (), {})()
        tags.estimator_type = "classifier"
        return tags