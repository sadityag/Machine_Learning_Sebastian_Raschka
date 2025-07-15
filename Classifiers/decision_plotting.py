import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
from typing import Optional, List, Tuple, Union, Any
from numpy.typing import ArrayLike

class DecisionRegionPlotter:
    """
    A seaborn-based decision region plotter for machine learning classifiers.
    
    This class provides a clean interface for visualizing decision boundaries
    and regions for any classifier that implements a predict() method.
    """
    
    def __init__(self, style: str = 'whitegrid', palette: str = 'Set2', context: str = 'notebook') -> None:
        """
        Initialize the plotter with seaborn styling options.
        
        Parameters:
        -----------
        style : str, default='whitegrid'
            Seaborn style for the plot
        palette : str, default='Set2'
            Color palette for the classes
        context : str, default='notebook'
            Seaborn context for sizing
        """
        sns.set_style(style)
        sns.set_context(context)
        self.palette = palette
        
    def plot_decision_regions(self, X_train: ArrayLike, y_train: ArrayLike, classifier: Any,
                            X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                            feature_indices: Optional[Tuple[int, int]] = None,
                            resolution: float = 0.02, figsize: Tuple[int, int] = (10, 8), 
                            alpha_regions: float = 0.3, alpha_points: float = 0.8,
                            title: Optional[str] = None, feature_names: Optional[List[str]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot decision regions for a 2D projection of classification data.
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training feature matrix (can have any number of features)
        y_train : array-like, shape (n_samples,)
            Training target vector
        classifier : object
            Trained classifier with predict() method
        X_test : array-like, optional, shape (n_test_samples, n_features)
            Test feature matrix to highlight as test points
        y_test : array-like, optional, shape (n_test_samples,)
            Test target vector (used for validation but not plotting)
        feature_indices : tuple of int, optional, default=(0, 1)
            Which two features to plot (feature_index_x, feature_index_y)
        resolution : float, default=0.02
            Resolution for the decision region mesh
        figsize : tuple, default=(10, 8)
            Figure size
        alpha_regions : float, default=0.3
            Transparency for decision regions
        alpha_points : float, default=0.8
            Transparency for data points
        title : str, optional
            Plot title
        feature_names : list, optional
            Names for all features (will select the two being plotted)
        
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        
        # Convert to numpy arrays for easier indexing
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        if X_test is not None:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        
        # Validate input
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array")
        
        # Set default feature indices
        if feature_indices is None:
            feature_indices = (0, 1)
        
        # Validate feature indices
        if len(feature_indices) != 2:
            raise ValueError("feature_indices must contain exactly 2 indices")
        
        feat_x, feat_y = feature_indices
        n_features = X_train.shape[1]
        
        if feat_x >= n_features or feat_y >= n_features:
            raise ValueError(f"Feature indices {feature_indices} are out of bounds for data with {n_features} features")
        
        if feat_x < 0 or feat_y < 0:
            raise ValueError("Feature indices must be non-negative")
        
        # Extract the two features for plotting
        X_train_2d = X_train[:, [feat_x, feat_y]]
        
        if X_test is not None:
            X_test_2d = X_test[:, [feat_x, feat_y]]
            X_combined_2d = np.vstack((X_train_2d, X_test_2d))
            y_combined = np.hstack((y_train, y_test))
        else:
            X_combined_2d = X_train_2d
            y_combined = y_train
            X_test_2d = None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique classes and create color palette
        unique_classes = np.unique(y_combined)
        n_classes = len(unique_classes)
        colors = sns.color_palette(self.palette, n_classes)
        
        # Create colormap for regions
        cmap = ListedColormap(colors)
        
        # Create mesh for decision regions using combined 2D data range
        # Add padding based on data range for better visualization
        x1_range = X_combined_2d[:, 0].max() - X_combined_2d[:, 0].min()
        x2_range = X_combined_2d[:, 1].max() - X_combined_2d[:, 1].min()
        padding = max(x1_range, x2_range) * 0.1  # 10% padding
        
        x1_min, x1_max = X_combined_2d[:, 0].min() - padding, X_combined_2d[:, 0].max() + padding
        x2_min, x2_max = X_combined_2d[:, 1].min() - padding, X_combined_2d[:, 1].max() + padding
        
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
        )
        
        # Create mesh points with all original features
        mesh_2d = np.array([xx1.ravel(), xx2.ravel()]).T
        
        # For multi-dimensional data, we need to create full feature vectors for prediction
        # We'll use the mean values of the non-plotted features from the training data
        if n_features > 2:
            # Calculate mean values for non-plotted features
            other_features = [i for i in range(n_features) if i not in feature_indices]
            mean_values = np.mean(X_train[:, other_features], axis=0)
            
            # Create full feature vectors for mesh points
            mesh_full = np.zeros((mesh_2d.shape[0], n_features))
            mesh_full[:, feat_x] = mesh_2d[:, 0]
            mesh_full[:, feat_y] = mesh_2d[:, 1]
            mesh_full[:, other_features] = mean_values
        else:
            mesh_full = mesh_2d
        
        # Handle potential prediction errors
        try:
            predictions = classifier.predict(mesh_full)
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")
            predictions = np.zeros(mesh_full.shape[0])
        
        predictions = predictions.reshape(xx1.shape)
        
        # Plot decision regions with more levels for smoother boundaries
        contour_levels = max(n_classes, 10)  # Use at least 10 levels for smooth boundaries
        ax.contourf(xx1, xx2, predictions, alpha=alpha_regions, cmap=cmap, levels=contour_levels)
        
        # Plot data points for each class
        markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h']
        
        # Plot training points
        for idx, class_value in enumerate(unique_classes):
            train_mask = (y_train == class_value)
            if np.any(train_mask):
                ax.scatter(
                    X_train_2d[train_mask, 0], 
                    X_train_2d[train_mask, 1],
                    c=[colors[idx]], 
                    marker=markers[idx % len(markers)],
                    s=60,
                    alpha=alpha_points,
                    edgecolor='black',
                    linewidth=0.5,
                    label=f'Train Class {class_value}'
                )
        
        # Plot test points if provided
        if X_test_2d is not None and y_test is not None:
            for idx, class_value in enumerate(unique_classes):
                test_mask = (y_test == class_value)
                if np.any(test_mask):
                    ax.scatter(
                        X_test_2d[test_mask, 0], 
                        X_test_2d[test_mask, 1],
                        c='none',
                        edgecolor='black',
                        linewidth=2,
                        marker=markers[idx % len(markers)],
                        s=100,
                        label=f'Test Class {class_value}'
                    )
        
        # Customize plot
        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())
        
        # Set labels
        if feature_names and len(feature_names) > max(feat_x, feat_y):
            ax.set_xlabel(feature_names[feat_x])
            ax.set_ylabel(feature_names[feat_y])
        else:
            ax.set_xlabel(f'Feature {feat_x}')
            ax.set_ylabel(f'Feature {feat_y}')
        
        if title:
            ax.set_title(title)
        elif n_features > 2:
            # Add informative title about feature projection
            other_features = [i for i in range(n_features) if i not in feature_indices]
            ax.set_title(f'Decision Regions: Features {feat_x} vs {feat_y}\n'
                        f'(Other features fixed at mean values)')
        
        # Add legend
        ax.legend()
        
        # Apply seaborn styling
        sns.despine()
        
        return fig, ax
    
    def plot_multiple_classifiers(self, X_train: ArrayLike, y_train: ArrayLike, classifiers: List[Any], 
                                classifier_names: Optional[List[str]] = None,
                                X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                                feature_indices: Optional[Tuple[int, int]] = None,
                                resolution: float = 0.02, figsize: Tuple[int, int] = (15, 10),
                                feature_names: Optional[List[str]] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot decision regions for multiple classifiers in subplots.
        
        Parameters:
        -----------
        X_train, y_train : array-like
            Training feature matrix and target vector
        classifiers : list
            List of trained classifiers
        classifier_names : list, optional
            Names for each classifier
        X_test, y_test : array-like, optional
            Test feature matrix and target vector
        feature_indices : tuple of int, optional, default=(0, 1)
            Which two features to plot (feature_index_x, feature_index_y)
        resolution : float, default=0.02
            Resolution for the decision region mesh
        figsize : tuple, default=(15, 10)
            Figure size
        feature_names : list, optional
            Names for all features (will select the two being plotted)
        
        Returns:
        --------
        fig, axes : matplotlib figure and axis objects
        """
        
        n_classifiers = len(classifiers)
        n_cols = min(3, n_classifiers)
        n_rows = (n_classifiers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single subplot case
        if n_classifiers == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes array for easier iteration
        axes_flat = axes.flatten()
        
        for idx, (classifier, ax) in enumerate(zip(classifiers, axes_flat)):
            # Set current axis
            plt.sca(ax)
            
            # Get classifier name
            if classifier_names:
                title = classifier_names[idx]
            else:
                title = f'Classifier {idx + 1}'
            
            # Plot decision regions
            self._plot_single_region(X_train, y_train, classifier, ax, X_test, y_test,
                                   feature_indices, resolution, title, feature_names)
        
        # Hide unused subplots
        for idx in range(n_classifiers, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        return fig, axes
    
    def _plot_single_region(self, X_train: ArrayLike, y_train: ArrayLike, classifier: Any, ax: plt.Axes, 
                          X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                          feature_indices: Optional[Tuple[int, int]] = None,
                          resolution: float = 0.02, title: Optional[str] = None, 
                          feature_names: Optional[List[str]] = None) -> None:
        """Helper method for plotting a single decision region."""
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        if X_test is not None:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        
        # Set default feature indices
        if feature_indices is None:
            feature_indices = (0, 1)
        
        feat_x, feat_y = feature_indices
        n_features = X_train.shape[1]
        
        # Extract the two features for plotting
        X_train_2d = X_train[:, [feat_x, feat_y]]
        
        # Combine train and test data for plotting range
        if X_test is not None:
            X_test_2d = X_test[:, [feat_x, feat_y]]
            X_combined_2d = np.vstack((X_train_2d, X_test_2d))
            y_combined = np.hstack((y_train, y_test))
        else:
            X_combined_2d = X_train_2d
            y_combined = y_train
            X_test_2d = None
        
        # Get unique classes and create color palette
        unique_classes = np.unique(y_combined)
        n_classes = len(unique_classes)
        colors = sns.color_palette(self.palette, n_classes)
        cmap = ListedColormap(colors)
        
        # Create mesh with adaptive padding
        x1_range = X_combined_2d[:, 0].max() - X_combined_2d[:, 0].min()
        x2_range = X_combined_2d[:, 1].max() - X_combined_2d[:, 1].min()
        padding = max(x1_range, x2_range) * 0.1  # 10% padding
        
        x1_min, x1_max = X_combined_2d[:, 0].min() - padding, X_combined_2d[:, 0].max() + padding
        x2_min, x2_max = X_combined_2d[:, 1].min() - padding, X_combined_2d[:, 1].max() + padding
        
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
        )
        
        # Create mesh points with all original features
        mesh_2d = np.array([xx1.ravel(), xx2.ravel()]).T
        
        if n_features > 2:
            # Calculate mean values for non-plotted features
            other_features = [i for i in range(n_features) if i not in feature_indices]
            mean_values = np.mean(X_train[:, other_features], axis=0)
            
            # Create full feature vectors for mesh points
            mesh_full = np.zeros((mesh_2d.shape[0], n_features))
            mesh_full[:, feat_x] = mesh_2d[:, 0]
            mesh_full[:, feat_y] = mesh_2d[:, 1]
            mesh_full[:, other_features] = mean_values
        else:
            mesh_full = mesh_2d
        
        # Predict and plot regions
        try:
            predictions = classifier.predict(mesh_full).reshape(xx1.shape)
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")
            predictions = np.zeros(xx1.shape)
        
        contour_levels = max(n_classes, 10)  # Use at least 10 levels for smooth boundaries
        ax.contourf(xx1, xx2, predictions, alpha=0.3, cmap=cmap, levels=contour_levels)
        
        # Plot data points
        markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h']
        
        # Plot training points
        for idx, class_value in enumerate(unique_classes):
            train_mask = (y_train == class_value)
            if np.any(train_mask):
                ax.scatter(
                    X_train_2d[train_mask, 0], X_train_2d[train_mask, 1],
                    c=[colors[idx]], marker=markers[idx % len(markers)],
                    s=60, alpha=0.8, edgecolor='black', linewidth=0.5,
                    label=f'Train Class {class_value}'
                )
        
        # Plot test points if provided
        if X_test_2d is not None and y_test is not None:
            for idx, class_value in enumerate(unique_classes):
                test_mask = (y_test == class_value)
                if np.any(test_mask):
                    ax.scatter(
                        X_test_2d[test_mask, 0], X_test_2d[test_mask, 1],
                        c='none', marker=markers[idx % len(markers)],
                        s=100, alpha=0.8, edgecolor='black', linewidth=2,
                        label=f'Test Class {class_value}'
                    )
        
        # Customize
        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())
        
        if feature_names and len(feature_names) > max(feat_x, feat_y):
            ax.set_xlabel(feature_names[feat_x])
            ax.set_ylabel(feature_names[feat_y])
        else:
            ax.set_xlabel(f'Feature {feat_x}')
            ax.set_ylabel(f'Feature {feat_y}')
        
        if title:
            ax.set_title(title)
        elif n_features > 2:
            ax.set_title(f'Features {feat_x} vs {feat_y}')
        
        ax.legend()


# Convenience function for quick plotting
def plot_decision_regions(X_train: ArrayLike, y_train: ArrayLike, classifier: Any,
                         X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                         feature_indices: Optional[Tuple[int, int]] = None,
                         resolution: float = 0.01, figsize: Tuple[int, int] = (10, 8), 
                         alpha_regions: float = 0.3, alpha_points: float = 0.8,
                         title: Optional[str] = None, feature_names: Optional[List[str]] = None,
                         style: str = 'whitegrid', palette: str = 'Set2') -> Tuple[plt.Figure, plt.Axes]:
    """
    Quick function to plot decision regions with seaborn styling.
    
    Parameters are the same as DecisionRegionPlotter.plot_decision_regions()
    """
    plotter = DecisionRegionPlotter(style=style, palette=palette)
    return plotter.plot_decision_regions(X_train, y_train, classifier, X_test, y_test, 
                                       feature_indices, resolution, figsize, alpha_regions, alpha_points,
                                       title, feature_names)