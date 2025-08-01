import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
from typing import Optional, List, Tuple, Union, Any, Dict, Callable
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class RegressionFitPlotter:
    """
    A seaborn-based regression fit plotter for visualizing how continuous variables
    relate to target variables in regression problems.
    
    This class provides clean interfaces for:
    - Single feature vs target plots with regression lines
    - Multi-feature regression surface plots (2D features vs target)
    - Residual analysis plots
    - Feature importance through partial dependence-style plots
    """
    
    def __init__(self, style: str = 'whitegrid', palette: str = 'viridis', context: str = 'notebook') -> None:
        """
        Initialize the plotter with seaborn styling options.
        
        Parameters:
        -----------
        style : str, default='whitegrid'
            Seaborn style for the plot
        palette : str, default='viridis'
            Color palette for continuous variables
        context : str, default='notebook'
            Seaborn context for sizing
        """
        sns.set_style(style)
        sns.set_context(context)
        self.palette = palette
        self.marker_sizes = {'train': 60, 'test': 100}
        
    def plot_feature_vs_target(self, X: ArrayLike, y: ArrayLike, feature_index: int,
                              regressor: Optional[Any] = None, X_test: Optional[ArrayLike] = None, 
                              y_test: Optional[ArrayLike] = None, feature_names: Optional[List[str]] = None,
                              target_name: Optional[str] = None, figsize: Tuple[int, int] = (10, 6),
                              show_ci: bool = True, scatter_alpha: float = 0.6, line_alpha: float = 0.8,
                              color_by_residuals: bool = False, title: Optional[str] = None,
                              plot_regression_line: bool = True, n_points: int = 100,
                              line_color: str = 'red') -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a single feature against the target variable with optional regression line.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target vector
        feature_index : int
            Index of the feature to plot
        regressor : object, optional
            Trained regressor with predict() method. If None, uses simple linear regression
        X_test : array-like, optional
            Test feature matrix
        y_test : array-like, optional
            Test target vector
        feature_names : list, optional
            Names for features
        target_name : str, optional
            Name for target variable
        figsize : tuple, default=(10, 6)
            Figure size
        show_ci : bool, default=True
            Show confidence intervals for regression line
        scatter_alpha : float, default=0.6
            Transparency for scatter points
        line_alpha : float, default=0.8
            Transparency for regression line
        color_by_residuals : bool, default=False
            Color points by residuals if regressor is provided
        title : str, optional
            Plot title
        plot_regression_line : bool, default=True
            Whether to plot regression line
        n_points : int, default=100
            Number of points for smooth regression line
        line_color : str, default='red'
            Color for the regression line
            
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        
        X = np.array(X)
        y = np.array(y)
        
        if X_test is not None:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        
        # Validate feature index
        if feature_index >= X.shape[1] or feature_index < 0:
            raise ValueError(f"Feature index {feature_index} is out of bounds for data with {X.shape[1]} features")
        
        # Extract the feature
        x_feature = X[:, feature_index]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine colors for points
        if color_by_residuals and regressor is not None:
            try:
                y_pred = regressor.predict(X)
                residuals = y - y_pred
                colors = residuals
                cmap = 'RdYlBu_r'  # Red for positive residuals, blue for negative
            except Exception as e:
                warnings.warn(f"Could not compute residuals: {e}")
                colors = 'blue'
                cmap = None
        else:
            colors = 'blue'
            cmap = None
        
        # Plot training points
        scatter = ax.scatter(x_feature, y, c=colors, alpha=scatter_alpha, 
                           s=self.marker_sizes['train'], cmap=cmap,
                           label='Training Data', edgecolor='black', linewidth=0.5)
        
        # Add colorbar for residuals
        if color_by_residuals and regressor is not None and isinstance(colors, np.ndarray):
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Residuals (y - ŷ)')
        
        # Plot test points if provided
        if X_test is not None and y_test is not None:
            x_test_feature = X_test[:, feature_index]
            ax.scatter(x_test_feature, y_test, c='red', alpha=scatter_alpha,
                      s=self.marker_sizes['test'], marker='^', 
                      label='Test Data', edgecolor='black', linewidth=1)
        
        # Plot regression line
        if plot_regression_line:
            x_range = np.linspace(x_feature.min(), x_feature.max(), n_points)
            
            if regressor is not None:
                # Use provided regressor
                try:
                    # Create full feature vectors for prediction
                    if X.shape[1] > 1:
                        # For multi-dimensional data, use mean values for other features
                        other_features = [i for i in range(X.shape[1]) if i != feature_index]
                        mean_values = np.mean(X[:, other_features], axis=0)
                        
                        x_range_full = np.zeros((len(x_range), X.shape[1]))
                        x_range_full[:, feature_index] = x_range
                        x_range_full[:, other_features] = mean_values
                    else:
                        x_range_full = x_range.reshape(-1, 1)
                    
                    y_range_pred = regressor.predict(x_range_full)
                    ax.plot(x_range, y_range_pred, color=line_color, alpha=line_alpha, 
                           linewidth=2, label='Model Prediction')
                    
                    # Add R² score to plot
                    if X.shape[1] == 1:
                        # Only calculate R² for single feature case
                        y_pred_train = regressor.predict(X)
                        r2 = r2_score(y, y_pred_train)
                        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                except Exception as e:
                    warnings.warn(f"Could not plot model prediction: {e}")
            else:
                # Use seaborn regplot for simple linear regression
                if show_ci:
                    sns.regplot(x=x_feature, y=y, ax=ax, scatter=False, 
                               color=line_color, line_kws={'alpha': line_alpha})
                else:
                    # Simple linear regression without confidence intervals
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_feature, y)
                    y_range_pred = slope * x_range + intercept
                    ax.plot(x_range, y_range_pred, color=line_color, alpha=line_alpha, 
                           linewidth=2, label=f'Linear Fit (R² = {r_value**2:.3f})')
        
        # Customize plot
        if feature_names and len(feature_names) > feature_index:
            ax.set_xlabel(feature_names[feature_index])
        else:
            ax.set_xlabel(f'Feature {feature_index}')
        
        if target_name:
            ax.set_ylabel(target_name)
        else:
            ax.set_ylabel('Target')
        
        if title:
            ax.set_title(title)
        else:
            feature_name = feature_names[feature_index] if feature_names else f'Feature {feature_index}'
            target_label = target_name if target_name else 'Target'
            ax.set_title(f'{feature_name} vs {target_label}')
        
        # Add legend
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Apply seaborn styling
        sns.despine()
        
        return fig, ax
    
    def plot_regression_surface(self, X: ArrayLike, y: ArrayLike, regressor: Any,
                               feature_indices: Tuple[int, int] = (0, 1),
                               X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                               resolution: float = 0.02, figsize: Tuple[int, int] = (12, 8),
                               feature_names: Optional[List[str]] = None, target_name: Optional[str] = None,
                               title: Optional[str] = None, show_surface: bool = True,
                               show_contours: bool = True, n_contours: int = 20,
                               alpha_surface: float = 0.6, alpha_points: float = 0.8,
                               fast_mode: bool = False, max_samples: int = 1000,
                               batch_size: int = 10000) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot regression surface for two features vs target.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target vector
        regressor : object
            Trained regressor with predict() method
        feature_indices : tuple, default=(0, 1)
            Which two features to plot
        X_test, y_test : array-like, optional
            Test data
        resolution : float, default=0.02
            Resolution for the surface mesh
        figsize : tuple, default=(12, 8)
            Figure size
        feature_names : list, optional
            Names for features
        target_name : str, optional
            Name for target variable
        title : str, optional
            Plot title
        show_surface : bool, default=True
            Whether to show filled contour surface
        show_contours : bool, default=True
            Whether to show contour lines
        n_contours : int, default=20
            Number of contour levels
        alpha_surface : float, default=0.6
            Transparency for surface
        alpha_points : float, default=0.8
            Transparency for data points
        fast_mode : bool, default=False
            Enable optimizations for large datasets
        max_samples : int, default=1000
            Maximum samples for bounds in fast mode
        batch_size : int, default=10000
            Batch size for predictions
            
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        
        X = np.array(X)
        y = np.array(y)
        
        if X_test is not None:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        
        # Fast mode optimizations
        if fast_mode and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_subset = X[indices]
            y_subset = y[indices]
        else:
            X_subset = X
            y_subset = y
        
        # Validate feature indices
        feat_x, feat_y = feature_indices
        if feat_x >= X.shape[1] or feat_y >= X.shape[1] or feat_x < 0 or feat_y < 0:
            raise ValueError(f"Feature indices {feature_indices} are out of bounds")
        
        # Extract features for plotting
        X_2d = X[:, [feat_x, feat_y]]
        X_subset_2d = X_subset[:, [feat_x, feat_y]]
        
        if X_test is not None:
            X_test_2d = X_test[:, [feat_x, feat_y]]
            X_combined_2d = np.vstack((X_subset_2d, X_test_2d))
            y_combined = np.hstack((y_subset, y_test))
        else:
            X_combined_2d = X_subset_2d
            y_combined = y_subset
            X_test_2d = None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create mesh for surface
        x1_range = X_combined_2d[:, 0].max() - X_combined_2d[:, 0].min()
        x2_range = X_combined_2d[:, 1].max() - X_combined_2d[:, 1].min()
        padding = max(x1_range, x2_range) * 0.1
        
        x1_min, x1_max = X_combined_2d[:, 0].min() - padding, X_combined_2d[:, 0].max() + padding
        x2_min, x2_max = X_combined_2d[:, 1].min() - padding, X_combined_2d[:, 1].max() + padding
        
        # Smart resolution adjustment
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
        
        # Create mesh points with all features
        mesh_2d = np.array([xx1.ravel(), xx2.ravel()]).T
        
        if X.shape[1] > 2:
            # Use mean values for non-plotted features
            other_features = [i for i in range(X.shape[1]) if i not in feature_indices]
            mean_values = np.mean(X[:, other_features], axis=0)
            
            mesh_full = np.zeros((mesh_2d.shape[0], X.shape[1]))
            mesh_full[:, feat_x] = mesh_2d[:, 0]
            mesh_full[:, feat_y] = mesh_2d[:, 1]
            mesh_full[:, other_features] = mean_values
        else:
            mesh_full = mesh_2d
        
        # Predict on mesh with batching
        try:
            mesh_size = mesh_full.shape[0]
            if mesh_size > batch_size:
                predictions = np.zeros(mesh_size)
                for i in range(0, mesh_size, batch_size):
                    end_idx = min(i + batch_size, mesh_size)
                    predictions[i:end_idx] = regressor.predict(mesh_full[i:end_idx])
            else:
                predictions = regressor.predict(mesh_full)
            predictions = predictions.reshape(xx1.shape)
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")
            return fig, ax
        
        # Plot surface
        if show_surface:
            contourf = ax.contourf(xx1, xx2, predictions, levels=n_contours, 
                                  alpha=alpha_surface, cmap=self.palette)
            plt.colorbar(contourf, ax=ax, label=target_name if target_name else 'Predicted Target')
        
        if show_contours:
            contour = ax.contour(xx1, xx2, predictions, levels=n_contours, 
                               colors='black', alpha=0.3, linewidths=0.5)
            ax.clabel(contour, inline=True, fontsize=8)
        
        # Color points by target value
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=self.marker_sizes['train'],
                           alpha=alpha_points, cmap=self.palette, edgecolor='black', 
                           linewidth=0.5, label='Training Data')
        
        # Plot test points if provided
        if X_test_2d is not None:
            ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, s=self.marker_sizes['test'],
                      alpha=alpha_points, cmap=self.palette, edgecolor='red', 
                      linewidth=2, marker='^', label='Test Data')
        
        # Customize plot
        if feature_names:
            ax.set_xlabel(feature_names[feat_x])
            ax.set_ylabel(feature_names[feat_y])
        else:
            ax.set_xlabel(f'Feature {feat_x}')
            ax.set_ylabel(f'Feature {feat_y}')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Regression Surface')
        
        ax.legend()
        sns.despine()
        
        # Add R² score
        try:
            y_pred = regressor.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            ax.text(0.02, 0.98, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            warnings.warn(f"Could not compute metrics: {e}")
        
        return fig, ax
    
    def plot_residuals(self, X: ArrayLike, y: ArrayLike, regressor: Any,
                      X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                      figsize: Tuple[int, int] = (15, 5), target_name: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot residual analysis: predicted vs actual, residuals vs predicted, and residual distribution.
        
        Parameters:
        -----------
        X, y : array-like
            Training features and targets
        regressor : object
            Trained regressor
        X_test, y_test : array-like, optional
            Test data
        figsize : tuple, default=(15, 5)
            Figure size
        target_name : str, optional
            Name for target variable
            
        Returns:
        --------
        fig, axes : matplotlib figure and axis objects
        """
        
        X = np.array(X)
        y = np.array(y)
        
        if X_test is not None:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        
        try:
            y_pred = regressor.predict(X)
            residuals = y - y_pred
            
            if X_test is not None and y_test is not None:
                y_test_pred = regressor.predict(X_test)
                test_residuals = y_test - y_test_pred
        except Exception as e:
            warnings.warn(f"Could not compute predictions: {e}")
            return plt.subplots(1, 3, figsize=figsize)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Predicted vs Actual
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        
        axes[0].scatter(y, y_pred, alpha=0.6, label='Training')
        if X_test is not None:
            axes[0].scatter(y_test, y_test_pred, alpha=0.6, color='red', marker='^', label='Test')
        
        axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.75, zorder=0)
        axes[0].set_xlabel(f'Actual {target_name}' if target_name else 'Actual')
        axes[0].set_ylabel(f'Predicted {target_name}' if target_name else 'Predicted')
        axes[0].set_title('Predicted vs Actual')
        axes[0].legend()
        
        # Add metrics
        r2_train = r2_score(y, y_pred)
        mae_train = mean_absolute_error(y, y_pred)
        text = f'Train R² = {r2_train:.3f}\nTrain MAE = {mae_train:.3f}'
        
        if X_test is not None:
            r2_test = r2_score(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            text += f'\nTest R² = {r2_test:.3f}\nTest MAE = {mae_test:.3f}'
        
        axes[0].text(0.05, 0.95, text, transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residuals vs Predicted
        axes[1].scatter(y_pred, residuals, alpha=0.6, label='Training')
        if X_test is not None:
            axes[1].scatter(y_test_pred, test_residuals, alpha=0.6, color='red', marker='^', label='Test')
        
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.75)
        axes[1].set_xlabel(f'Predicted {target_name}' if target_name else 'Predicted')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals vs Predicted')
        axes[1].legend()
        
        # 3. Residual Distribution
        axes[2].hist(residuals, bins=30, alpha=0.7, density=True, label='Training')
        if X_test is not None:
            axes[2].hist(test_residuals, bins=30, alpha=0.7, density=True, label='Test')
        
        # Add normal distribution overlay
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[2].plot(x, stats.norm.pdf(x, mu, sigma), 'k-', linewidth=2, label='Normal Fit')
        
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Residual Distribution')
        axes[2].legend()
        
        plt.tight_layout()
        sns.despine()
        
        return fig, axes
    
    def plot_multiple_features(self, X: ArrayLike, y: ArrayLike, regressor: Optional[Any] = None,
                              feature_indices: Optional[List[int]] = None,
                              X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                              feature_names: Optional[List[str]] = None, target_name: Optional[str] = None,
                              figsize: Tuple[int, int] = (15, 10), max_features: int = 6,
                              line_color: str = 'red') -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot multiple features vs target in subplots.
        
        Parameters:
        -----------
        X, y : array-like
            Features and target
        regressor : object, optional
            Trained regressor
        feature_indices : list, optional
            Which features to plot (if None, plots first max_features)
        X_test, y_test : array-like, optional
            Test data
        feature_names : list, optional
            Feature names
        target_name : str, optional
            Target name
        figsize : tuple, default=(15, 10)
            Figure size
        max_features : int, default=6
            Maximum number of features to plot
        line_color : str, default='red'
            Color for the regression lines
            
        Returns:
        --------
        fig, axes : matplotlib figure and axis objects
        """
        
        X = np.array(X)
        y = np.array(y)
        
        if feature_indices is None:
            feature_indices = list(range(min(max_features, X.shape[1])))
        else:
            feature_indices = feature_indices[:max_features]
        
        n_features = len(feature_indices)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten()
        
        for idx, feat_idx in enumerate(feature_indices):
            ax = axes_flat[idx]
            plt.sca(ax)
            
            # Plot single feature
            self._plot_single_feature(X, y, feat_idx, regressor, X_test, y_test,
                                    feature_names, target_name, ax, line_color)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        return fig, axes
    
    def _plot_single_feature(self, X: ArrayLike, y: ArrayLike, feature_index: int,
                           regressor: Optional[Any], X_test: Optional[ArrayLike], 
                           y_test: Optional[ArrayLike], feature_names: Optional[List[str]],
                           target_name: Optional[str], ax: plt.Axes, 
                           line_color: str = 'red') -> None:
        """Helper method for plotting a single feature."""
        
        x_feature = X[:, feature_index]
        
        # Plot training points
        ax.scatter(x_feature, y, alpha=0.6, s=self.marker_sizes['train'],
                  label='Training', edgecolor='black', linewidth=0.5)
        
        # Plot test points
        if X_test is not None and y_test is not None:
            x_test_feature = X_test[:, feature_index]
            ax.scatter(x_test_feature, y_test, alpha=0.6, s=self.marker_sizes['test'],
                      color='red', marker='^', label='Test', edgecolor='black', linewidth=1)
        
        # Plot regression line
        if regressor is not None:
            x_range = np.linspace(x_feature.min(), x_feature.max(), 100)
            
            try:
                if X.shape[1] > 1:
                    # Use mean values for other features
                    other_features = [i for i in range(X.shape[1]) if i != feature_index]
                    mean_values = np.mean(X[:, other_features], axis=0)
                    
                    x_range_full = np.zeros((len(x_range), X.shape[1]))
                    x_range_full[:, feature_index] = x_range
                    x_range_full[:, other_features] = mean_values
                else:
                    x_range_full = x_range.reshape(-1, 1)
                
                y_range_pred = regressor.predict(x_range_full)
                ax.plot(x_range, y_range_pred, color=line_color, linewidth=2, 
                       alpha=0.8, label='Model')
                
            except Exception as e:
                warnings.warn(f"Could not plot regression line: {e}")
        else:
            # Simple linear regression
            sns.regplot(x=x_feature, y=y, ax=ax, scatter=False, color=line_color)
        
        # Customize
        if feature_names and len(feature_names) > feature_index:
            ax.set_xlabel(feature_names[feature_index])
        else:
            ax.set_xlabel(f'Feature {feature_index}')
        
        if target_name:
            ax.set_ylabel(target_name)
        else:
            ax.set_ylabel('Target')
        
        feature_name = feature_names[feature_index] if feature_names else f'Feature {feature_index}'
        ax.set_title(f'{feature_name} vs Target')
        
        ax.legend()
        sns.despine()


# Convenience functions
def plot_feature_vs_target(X: ArrayLike, y: ArrayLike, feature_index: int,
                          regressor: Optional[Any] = None, X_test: Optional[ArrayLike] = None, 
                          y_test: Optional[ArrayLike] = None, feature_names: Optional[List[str]] = None,
                          target_name: Optional[str] = None, figsize: Tuple[int, int] = (10, 6),
                          show_ci: bool = True, color_by_residuals: bool = False,
                          style: str = 'whitegrid', palette: str = 'viridis',
                          line_color: str = 'red') -> Tuple[plt.Figure, plt.Axes]:
    """
    Quick function to plot a single feature vs target.
    """
    plotter = RegressionFitPlotter(style=style, palette=palette)
    return plotter.plot_feature_vs_target(X, y, feature_index, regressor, X_test, y_test,
                                        feature_names, target_name, figsize, show_ci,
                                        color_by_residuals=color_by_residuals,
                                        line_color=line_color)


def plot_regression_surface(X: ArrayLike, y: ArrayLike, regressor: Any,
                           feature_indices: Tuple[int, int] = (0, 1),
                           X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                           resolution: float = 0.02, figsize: Tuple[int, int] = (12, 8),
                           feature_names: Optional[List[str]] = None, target_name: Optional[str] = None,
                           fast_mode: bool = False, style: str = 'whitegrid', 
                           palette: str = 'viridis') -> Tuple[plt.Figure, plt.Axes]:
    """
    Quick function to plot regression surface for two features vs target.
    """
    plotter = RegressionFitPlotter(style=style, palette=palette)
    return plotter.plot_regression_surface(X, y, regressor, feature_indices, X_test, y_test,
                                         resolution, figsize, feature_names, target_name,
                                         fast_mode=fast_mode)


def plot_residuals(X: ArrayLike, y: ArrayLike, regressor: Any,
                  X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                  figsize: Tuple[int, int] = (15, 5), target_name: Optional[str] = None,
                  style: str = 'whitegrid', palette: str = 'viridis') -> Tuple[plt.Figure, np.ndarray]:
    """
    Quick function to plot residual analysis.
    """
    plotter = RegressionFitPlotter(style=style, palette=palette)
    return plotter.plot_residuals(X, y, regressor, X_test, y_test, figsize, target_name)


def plot_multiple_features(X: ArrayLike, y: ArrayLike, regressor: Optional[Any] = None,
                          feature_indices: Optional[List[int]] = None,
                          X_test: Optional[ArrayLike] = None, y_test: Optional[ArrayLike] = None,
                          feature_names: Optional[List[str]] = None, target_name: Optional[str] = None,
                          figsize: Tuple[int, int] = (15, 10), max_features: int = 6,
                          style: str = 'whitegrid', palette: str = 'viridis',
                          line_color: str = 'red') -> Tuple[plt.Figure, np.ndarray]:
    """
    Quick function to plot multiple features vs target in subplots.
    """
    plotter = RegressionFitPlotter(style=style, palette=palette)
    return plotter.plot_multiple_features(X, y, regressor, feature_indices, X_test, y_test,
                                        feature_names, target_name, figsize, max_features,
                                        line_color=line_color)


def check_regression_data(X: ArrayLike, y: ArrayLike, feature_indices: Optional[List[int]] = None,
                         resolution: float = 0.02) -> None:
    """
    Check your regression data and provide recommendations for plotting.
    
    Parameters:
    -----------
    X, y : array-like
        Features and target data
    feature_indices : list, optional
        Features to analyze (if None, analyzes first 6)
    resolution : float
        Proposed resolution for surface plots
    """
    X = np.array(X)
    y = np.array(y)
    
    if feature_indices is None:
        feature_indices = list(range(min(6, X.shape[1])))
    
    print("=== REGRESSION DATA ANALYSIS ===")
    print(f"Dataset shape: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}] (span: {y.max() - y.min():.3f})")
    print(f"Target mean: {y.mean():.3f}, std: {y.std():.3f}")
    
    print("\n--- Feature Analysis ---")
    for i, feat_idx in enumerate(feature_indices):
        if feat_idx < X.shape[1]:
            feat_vals = X[:, feat_idx]
            correlation = np.corrcoef(feat_vals, y)[0, 1]
            print(f"Feature {feat_idx}: range [{feat_vals.min():.3f}, {feat_vals.max():.3f}], "
                  f"correlation with target: {correlation:.3f}")
    
    # Check for surface plotting
    if X.shape[1] >= 2:
        print(f"\n--- Surface Plot Analysis (resolution {resolution}) ---")
        feat_x, feat_y = feature_indices[0], feature_indices[1] if len(feature_indices) > 1 else 1
        
        x1_range = X[:, feat_x].max() - X[:, feat_x].min()
        x2_range = X[:, feat_y].max() - X[:, feat_y].min()
        padding = max(x1_range, x2_range) * 0.1
        
        x1_points = int((x1_range + 2*padding) / resolution) + 1
        x2_points = int((x2_range + 2*padding) / resolution) + 1
        total_points = x1_points * x2_points
        
        print(f"Surface mesh size: {x1_points:,} x {x2_points:,} = {total_points:,} points")
        
        if total_points > 2500000:
            print("⚠️  WARNING: Very large mesh for surface plot!")
            optimal_resolution = max(x1_range, x2_range) / 500
            print(f"   Recommended: resolution={optimal_resolution:.3f} or fast_mode=True")
        elif total_points > 1000000:
            print("⚠️  Large mesh. Consider fast_mode=True or higher resolution")
        else:
            print("✅ Mesh size looks good for surface plotting!")
    
    print(f"\n--- Plotting Recommendations ---")
    print("For individual features:")
    print("  plot_feature_vs_target(X, y, feature_index=0, regressor=your_model, line_color='green')")
    
    if X.shape[1] >= 2:
        print("For regression surface:")
        print("  plot_regression_surface(X, y, regressor=your_model)")
    
    print("For residual analysis:")
    print("  plot_residuals(X, y, regressor=your_model)")
    
    if X.shape[1] > 1:
        print("For multiple features:")
        print("  plot_multiple_features(X, y, regressor=your_model, line_color='darkblue')")


# Example usage and demo function
def demo_regression_plotter():
    """
    Demonstrate the regression plotter with synthetic data.
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 300
    
    # Create features
    X1 = np.random.uniform(-3, 3, n_samples)
    X2 = np.random.uniform(-2, 2, n_samples) 
    X3 = np.random.normal(0, 1, n_samples)
    
    # Create target with known relationship
    y = 2 * X1 + X1**2 - 1.5 * X2 + 0.5 * X2**2 + 0.3 * X3 + np.random.normal(0, 0.5, n_samples)
    
    X = np.column_stack([X1, X2, X3])
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fit a model
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=50, random_state=42)
    regressor.fit(X_train, y_train)
    
    feature_names = ['X1 (Main Effect)', 'X2 (Secondary)', 'X3 (Noise)']
    
    print("=== REGRESSION PLOTTER DEMO ===")
    print("Generated synthetic data with known relationships")
    print("Fitted Random Forest Regressor")
    print("\nTry these plotting functions:")
    print("1. plot_feature_vs_target(X_train, y_train, 0, regressor, X_test, y_test, feature_names, line_color='green')")
    print("2. plot_regression_surface(X_train, y_train, regressor, (0,1), X_test, y_test, feature_names=feature_names)")
    print("3. plot_residuals(X_train, y_train, regressor, X_test, y_test)")
    print("4. plot_multiple_features(X_train, y_train, regressor, feature_names=feature_names, line_color='purple')")
    
    return X_train, y_train, X_test, y_test, regressor, feature_names


if __name__ == "__main__":
    # Run demo if script is executed directly
    demo_regression_plotter() 