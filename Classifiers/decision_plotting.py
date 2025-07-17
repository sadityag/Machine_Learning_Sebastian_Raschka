import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
from typing import Optional, List, Tuple, Union, Any, Dict
from numpy.typing import ArrayLike

class DecisionRegionPlotter:
    """
    A seaborn-based decision region plotter for machine learning classifiers.
    
    This class provides a clean interface for visualizing decision boundaries
    and regions for any classifier that implements a predict() method.
    Enhanced with better multiclass support.
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
        
        # Enhanced marker and color support for multiclass
        self.default_markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', 'H', '+', 'x', '8']
        self.marker_sizes = {'train': 60, 'test': 100}
        
    def _get_class_styling(self, unique_classes: np.ndarray, custom_colors: Optional[Dict] = None, 
                          custom_markers: Optional[Dict] = None) -> Tuple[List, List]:
        """
        Get consistent colors and markers for each class.
        
        Parameters:
        -----------
        unique_classes : array-like
            Unique class labels
        custom_colors : dict, optional
            Custom color mapping {class_label: color}
        custom_markers : dict, optional
            Custom marker mapping {class_label: marker}
            
        Returns:
        --------
        colors : list
            Colors for each class
        markers : list
            Markers for each class
        """
        n_classes = len(unique_classes)
        
        # Get colors
        if custom_colors:
            colors = [custom_colors.get(cls, sns.color_palette(self.palette, n_classes)[i]) 
                     for i, cls in enumerate(unique_classes)]
        else:
            colors = sns.color_palette(self.palette, n_classes)
        
        # Get markers
        if custom_markers:
            markers = [custom_markers.get(cls, self.default_markers[i % len(self.default_markers)]) 
                      for i, cls in enumerate(unique_classes)]
        else:
            markers = [self.default_markers[i % len(self.default_markers)] 
                      for i in range(n_classes)]
        
        return colors, markers
    
    def _create_class_legend_labels(self, unique_classes: np.ndarray, class_names: Optional[Dict] = None) -> List[str]:
        """
        Create readable legend labels for classes.
        
        Parameters:
        -----------
        unique_classes : array-like
            Unique class labels
        class_names : dict, optional
            Custom class names mapping {class_label: readable_name}
            
        Returns:
        --------
        labels : list
            Readable labels for each class
        """
        if class_names:
            return [class_names.get(cls, f'Class {cls}') for cls in unique_classes]
        else:
            return [f'Class {cls}' for cls in unique_classes]
        
    def plot_decision_regions(self, X_train: ArrayLike, y_train: ArrayLike, classifier: Any,
                            X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                            feature_indices: Optional[Tuple[int, int]] = None,
                            resolution: float = 0.02, figsize: Tuple[int, int] = (10, 8), 
                            alpha_regions: float = 0.3, alpha_points: float = 0.8,
                            title: Optional[str] = None, feature_names: Optional[List[str]] = None,
                            class_names: Optional[Dict] = None, custom_colors: Optional[Dict] = None,
                            custom_markers: Optional[Dict] = None, show_test_legend: bool = True,
                            legend_loc: str = 'best', fast_mode: bool = False, 
                            max_samples: int = 1000, batch_size: int = 10000) -> Tuple[plt.Figure, plt.Axes]:
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
        class_names : dict, optional
            Custom readable names for classes {class_label: readable_name}
        custom_colors : dict, optional
            Custom colors for classes {class_label: color}
        custom_markers : dict, optional
            Custom markers for classes {class_label: marker}
        show_test_legend : bool, default=True
            Whether to show separate legend entries for test points
        legend_loc : str, default='best'
            Legend location
        fast_mode : bool, default=False
            Enable optimizations for large datasets/slow classifiers like Random Forest
        max_samples : int, default=1000
            Maximum training samples to use for determining mesh bounds in fast_mode
        batch_size : int, default=10000
            Batch size for mesh predictions to avoid memory issues
        
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
        
        # Fast mode optimizations for large datasets
        if fast_mode and len(X_train) > max_samples:
            # Subsample training data for determining plot bounds only
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]
            print(f"Fast mode: Using {max_samples} samples for plot bounds")
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
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
        X_train_subset_2d = X_train_subset[:, [feat_x, feat_y]]
        
        if X_test is not None:
            X_test_2d = X_test[:, [feat_x, feat_y]]
            X_combined_2d = np.vstack((X_train_subset_2d, X_test_2d))
            y_combined = np.hstack((y_train_subset, y_test))
        else:
            X_combined_2d = X_train_subset_2d
            y_combined = y_train_subset
            X_test_2d = None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique classes and styling
        unique_classes = np.unique(y_combined)
        colors, markers = self._get_class_styling(unique_classes, custom_colors, custom_markers)
        class_labels = self._create_class_legend_labels(unique_classes, class_names)
        
        # Create colormap for regions
        cmap = ListedColormap(colors)
        
        # Create mesh for decision regions using combined 2D data range
        # Add padding based on data range for better visualization
        x1_range = X_combined_2d[:, 0].max() - X_combined_2d[:, 0].min()
        x2_range = X_combined_2d[:, 1].max() - X_combined_2d[:, 1].min()
        padding = max(x1_range, x2_range) * 0.1  # 10% padding
        
        x1_min, x1_max = X_combined_2d[:, 0].min() - padding, X_combined_2d[:, 0].max() + padding
        x2_min, x2_max = X_combined_2d[:, 1].min() - padding, X_combined_2d[:, 1].max() + padding
        
        # Smart resolution adjustment to prevent memory issues
        x1_points = int((x1_max - x1_min) / resolution) + 1
        x2_points = int((x2_max - x2_min) / resolution) + 1
        total_points = x1_points * x2_points
        
        # Limit mesh size to prevent memory errors
        max_mesh_points = 1000000 if fast_mode else 2500000  # 1M or 2.5M points max
        
        if total_points > max_mesh_points:
            # Calculate new resolution to stay under limit
            optimal_points_per_dim = int(np.sqrt(max_mesh_points))
            new_resolution_x = (x1_max - x1_min) / optimal_points_per_dim
            new_resolution_y = (x2_max - x2_min) / optimal_points_per_dim
            resolution = max(new_resolution_x, new_resolution_y)
            
            print(f"Mesh too large ({total_points:,} points). Adjusting resolution to {resolution:.3f}")
            print(f"Data ranges: X=[{x1_min:.2f}, {x1_max:.2f}], Y=[{x2_min:.2f}, {x2_max:.2f}]")
        
        # Automatically adjust resolution for fast mode after mesh size check
        if fast_mode and resolution < 0.05:
            resolution = max(resolution, 0.05)  # Use the larger of calculated or minimum
            print(f"Fast mode: Final resolution {resolution:.3f}")
        
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
        
        # Handle potential prediction errors with batching for large meshes
        try:
            mesh_size = mesh_full.shape[0]
            if mesh_size > batch_size:
                # Process in batches to avoid memory issues
                predictions = np.zeros(mesh_size)
                for i in range(0, mesh_size, batch_size):
                    end_idx = min(i + batch_size, mesh_size)
                    batch_predictions = classifier.predict(mesh_full[i:end_idx])
                    predictions[i:end_idx] = batch_predictions
                print(f"Processed {mesh_size} predictions in {(mesh_size + batch_size - 1) // batch_size} batches")
            else:
                predictions = classifier.predict(mesh_full)
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")
            predictions = np.zeros(mesh_full.shape[0])
        
        predictions = predictions.reshape(xx1.shape)
        
        # Plot decision regions
        ax.contourf(xx1, xx2, predictions, alpha=alpha_regions, cmap=cmap, levels=len(unique_classes))
        
        # Plot data points for each class
        legend_handles = []
        
        # Plot training points
        for idx, class_value in enumerate(unique_classes):
            train_mask = (y_train == class_value)
            if np.any(train_mask):
                scatter = ax.scatter(
                    X_train_2d[train_mask, 0], 
                    X_train_2d[train_mask, 1],
                    c=[colors[idx]], 
                    marker=markers[idx],
                    s=self.marker_sizes['train'],
                    alpha=alpha_points,
                    edgecolor='black',
                    linewidth=0.5,
                    label=f'{class_labels[idx]} (Train)'
                )
                legend_handles.append(scatter)
        
        # Plot test points if provided
        if X_test_2d is not None and y_test is not None:
            for idx, class_value in enumerate(unique_classes):
                test_mask = (y_test == class_value)
                if np.any(test_mask):
                    if show_test_legend:
                        label = f'{class_labels[idx]} (Test)'
                    else:
                        label = None
                    
                    scatter = ax.scatter(
                        X_test_2d[test_mask, 0], 
                        X_test_2d[test_mask, 1],
                        c='none',
                        edgecolor=colors[idx],
                        linewidth=2,
                        marker=markers[idx],
                        s=self.marker_sizes['test'],
                        alpha=alpha_points,
                        label=label
                    )
                    if show_test_legend:
                        legend_handles.append(scatter)
        
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
        
        # Add legend with better multiclass support
        if legend_handles:
            ax.legend(handles=legend_handles, loc=legend_loc, frameon=True, fancybox=True, shadow=True)
        
        # Apply seaborn styling
        sns.despine()
        
        return fig, ax
    
    def plot_multiple_classifiers(self, X_train: ArrayLike, y_train: ArrayLike, classifiers: List[Any], 
                                classifier_names: Optional[List[str]] = None,
                                X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                                feature_indices: Optional[Tuple[int, int]] = None,
                                resolution: float = 0.02, figsize: Tuple[int, int] = (15, 10),
                                feature_names: Optional[List[str]] = None,
                                class_names: Optional[Dict] = None, custom_colors: Optional[Dict] = None,
                                custom_markers: Optional[Dict] = None, show_test_legend: bool = True,
                                fast_mode: bool = False, max_samples: int = 1000, 
                                batch_size: int = 10000) -> Tuple[plt.Figure, np.ndarray]:
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
        class_names : dict, optional
            Custom readable names for classes {class_label: readable_name}
        custom_colors : dict, optional
            Custom colors for classes {class_label: color}
        custom_markers : dict, optional
            Custom markers for classes {class_label: marker}
        show_test_legend : bool, default=True
            Whether to show separate legend entries for test points
        fast_mode : bool, default=False
            Enable optimizations for large datasets/slow classifiers
        max_samples : int, default=1000
            Maximum training samples to use for determining mesh bounds in fast_mode
        batch_size : int, default=10000
            Batch size for mesh predictions to avoid memory issues
        
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
                                   feature_indices, resolution, title, feature_names,
                                   class_names, custom_colors, custom_markers, show_test_legend,
                                   fast_mode, max_samples, batch_size)
        
        # Hide unused subplots
        for idx in range(n_classifiers, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        return fig, axes
    
    def _plot_single_region(self, X_train: ArrayLike, y_train: ArrayLike, classifier: Any, ax: plt.Axes, 
                          X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                          feature_indices: Optional[Tuple[int, int]] = None,
                          resolution: float = 0.02, title: Optional[str] = None, 
                          feature_names: Optional[List[str]] = None,
                          class_names: Optional[Dict] = None, custom_colors: Optional[Dict] = None,
                          custom_markers: Optional[Dict] = None, show_test_legend: bool = True,
                          fast_mode: bool = False, max_samples: int = 1000, 
                          batch_size: int = 10000) -> None:
        """Helper method for plotting a single decision region."""
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        if X_test is not None:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        
        # Fast mode optimizations
        if fast_mode and len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        # Set default feature indices
        if feature_indices is None:
            feature_indices = (0, 1)
        
        feat_x, feat_y = feature_indices
        n_features = X_train.shape[1]
        
        # Extract the two features for plotting
        X_train_2d = X_train[:, [feat_x, feat_y]]
        X_train_subset_2d = X_train_subset[:, [feat_x, feat_y]]
        
        # Combine train and test data for plotting range
        if X_test is not None:
            X_test_2d = X_test[:, [feat_x, feat_y]]
            X_combined_2d = np.vstack((X_train_subset_2d, X_test_2d))
            y_combined = np.hstack((y_train_subset, y_test))
        else:
            X_combined_2d = X_train_subset_2d
            y_combined = y_train_subset
            X_test_2d = None
        
        # Get unique classes and styling
        unique_classes = np.unique(y_combined)
        colors, markers = self._get_class_styling(unique_classes, custom_colors, custom_markers)
        class_labels = self._create_class_legend_labels(unique_classes, class_names)
        
        cmap = ListedColormap(colors)
        
        # Create mesh with adaptive padding and size limits
        x1_range = X_combined_2d[:, 0].max() - X_combined_2d[:, 0].min()
        x2_range = X_combined_2d[:, 1].max() - X_combined_2d[:, 1].min()
        padding = max(x1_range, x2_range) * 0.1  # 10% padding
        
        x1_min, x1_max = X_combined_2d[:, 0].min() - padding, X_combined_2d[:, 0].max() + padding
        x2_min, x2_max = X_combined_2d[:, 1].min() - padding, X_combined_2d[:, 1].max() + padding
        
        # Smart resolution adjustment to prevent memory issues
        x1_points = int((x1_max - x1_min) / resolution) + 1
        x2_points = int((x2_max - x2_min) / resolution) + 1
        total_points = x1_points * x2_points
        
        max_mesh_points = 1000000 if fast_mode else 2500000
        
        if total_points > max_mesh_points:
            optimal_points_per_dim = int(np.sqrt(max_mesh_points))
            new_resolution_x = (x1_max - x1_min) / optimal_points_per_dim
            new_resolution_y = (x2_max - x2_min) / optimal_points_per_dim
            resolution = max(new_resolution_x, new_resolution_y)
        
        if fast_mode and resolution < 0.05:
            resolution = max(resolution, 0.05)
        
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
        
        # Predict and plot regions with batching
        try:
            mesh_size = mesh_full.shape[0]
            if mesh_size > batch_size:
                predictions = np.zeros(mesh_size)
                for i in range(0, mesh_size, batch_size):
                    end_idx = min(i + batch_size, mesh_size)
                    predictions[i:end_idx] = classifier.predict(mesh_full[i:end_idx])
            else:
                predictions = classifier.predict(mesh_full)
            predictions = predictions.reshape(xx1.shape)
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")
            predictions = np.zeros(xx1.shape)
        
        ax.contourf(xx1, xx2, predictions, alpha=0.3, cmap=cmap, levels=len(unique_classes))
        
        # Plot data points
        legend_handles = []
        
        # Plot training points
        for idx, class_value in enumerate(unique_classes):
            train_mask = (y_train == class_value)
            if np.any(train_mask):
                scatter = ax.scatter(
                    X_train_2d[train_mask, 0], X_train_2d[train_mask, 1],
                    c=[colors[idx]], marker=markers[idx],
                    s=self.marker_sizes['train'], alpha=0.8, edgecolor='black', linewidth=0.5,
                    label=f'{class_labels[idx]} (Train)'
                )
                legend_handles.append(scatter)
        
        # Plot test points if provided
        if X_test_2d is not None and y_test is not None:
            for idx, class_value in enumerate(unique_classes):
                test_mask = (y_test == class_value)
                if np.any(test_mask):
                    if show_test_legend:
                        label = f'{class_labels[idx]} (Test)'
                    else:
                        label = None
                    
                    scatter = ax.scatter(
                        X_test_2d[test_mask, 0], X_test_2d[test_mask, 1],
                        c='none', marker=markers[idx],
                        s=self.marker_sizes['test'], alpha=0.8, edgecolor=colors[idx], linewidth=2,
                        label=label
                    )
                    if show_test_legend:
                        legend_handles.append(scatter)
        
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
        
        # Add legend
        if legend_handles:
            ax.legend(handles=legend_handles, loc='best', frameon=True, fancybox=True, shadow=True)


def check_data_for_plotting(X_train: ArrayLike, y_train: ArrayLike, 
                           X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                           feature_indices: Optional[Tuple[int, int]] = None,
                           resolution: float = 0.02) -> None:
    """
    Check your data before plotting to estimate memory requirements and suggest settings.
    
    Parameters:
    -----------
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like, optional
        Test data
    feature_indices : tuple, optional
        Which features to analyze
    resolution : float
        Proposed resolution
    """
    X_train = np.array(X_train)
    if X_test is not None:
        X_test = np.array(X_test)
        X_combined = np.vstack((X_train, X_test))
    else:
        X_combined = X_train
    
    if feature_indices is None:
        feature_indices = (0, 1)
    
    feat_x, feat_y = feature_indices
    X_2d = X_combined[:, [feat_x, feat_y]]
    
    # Calculate ranges
    x1_range = X_2d[:, 0].max() - X_2d[:, 0].min()
    x2_range = X_2d[:, 1].max() - X_2d[:, 1].min()
    padding = max(x1_range, x2_range) * 0.1
    
    x1_min, x1_max = X_2d[:, 0].min() - padding, X_2d[:, 0].max() + padding
    x2_min, x2_max = X_2d[:, 1].min() - padding, X_2d[:, 1].max() + padding
    
    # Calculate mesh size
    x1_points = int((x1_max - x1_min) / resolution) + 1
    x2_points = int((x2_max - x2_min) / resolution) + 1
    total_points = x1_points * x2_points
    memory_gb = total_points * 8 * 2 / (1024**3)  # Rough estimate
    
    print("=== DATA ANALYSIS FOR PLOTTING ===")
    print(f"Feature {feat_x} range: [{X_2d[:, 0].min():.2f}, {X_2d[:, 0].max():.2f}] (span: {x1_range:.2f})")
    print(f"Feature {feat_y} range: [{X_2d[:, 1].min():.2f}, {X_2d[:, 1].max():.2f}] (span: {x2_range:.2f})")
    print(f"With resolution {resolution}:")
    print(f"  - Grid size: {x1_points:,} x {x2_points:,} = {total_points:,} points")
    print(f"  - Estimated memory: ~{memory_gb:.1f} GB")
    
    # Recommendations
    if total_points > 2500000:
        print("\n⚠️  WARNING: Very large mesh! Recommendations:")
        optimal_resolution = max((x1_max - x1_min) / 1000, (x2_max - x2_min) / 1000)
        print(f"  - Use fast_mode=True")
        print(f"  - Try resolution={optimal_resolution:.3f} or higher")
        print(f"  - Consider normalizing/scaling your features first")
    elif total_points > 1000000:
        print("\n⚠️  Large mesh. Consider:")
        print(f"  - Use fast_mode=True")
        print(f"  - Try resolution={resolution*2:.3f}")
    else:
        print("\n✅ Mesh size looks reasonable!")
    
    print(f"\nSample usage:")
    print(f"plot_decision_regions(X_train, y_train, classifier,")
    print(f"                     fast_mode=True, resolution={max(resolution, 0.1):.1f})")


# Enhanced convenience function
def plot_decision_regions(X_train: ArrayLike, y_train: ArrayLike, classifier: Any,
                         X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                         feature_indices: Optional[Tuple[int, int]] = None,
                         resolution: float = 0.01, figsize: Tuple[int, int] = (10, 8), 
                         alpha_regions: float = 0.3, alpha_points: float = 0.8,
                         title: Optional[str] = None, feature_names: Optional[List[str]] = None,
                         class_names: Optional[Dict] = None, custom_colors: Optional[Dict] = None,
                         custom_markers: Optional[Dict] = None, show_test_legend: bool = True,
                         style: str = 'whitegrid', palette: str = 'Set2',
                         fast_mode: bool = False, max_samples: int = 1000, 
                         batch_size: int = 10000) -> Tuple[plt.Figure, plt.Axes]:
    """
    Quick function to plot decision regions with enhanced multiclass support.
    
    Parameters:
    -----------
    All parameters from DecisionRegionPlotter.plot_decision_regions() plus:
    class_names : dict, optional
        Custom readable names for classes {class_label: readable_name}
    custom_colors : dict, optional
        Custom colors for classes {class_label: color}
    custom_markers : dict, optional
        Custom markers for classes {class_label: marker}
    show_test_legend : bool, default=True
        Whether to show separate legend entries for test points
    fast_mode : bool, default=False
        Enable optimizations for large datasets/slow classifiers like Random Forest
    max_samples : int, default=1000
        Maximum training samples to use for determining mesh bounds in fast_mode
    batch_size : int, default=10000
        Batch size for mesh predictions to avoid memory issues
    """
    plotter = DecisionRegionPlotter(style=style, palette=palette)
    return plotter.plot_decision_regions(X_train, y_train, classifier, X_test, y_test, 
                                       feature_indices, resolution, figsize, alpha_regions, alpha_points,
                                       title, feature_names, class_names, custom_colors, custom_markers,
                                       show_test_legend, 'best', fast_mode, max_samples, batch_size)