import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
import streamlit as st
from app.utils import enforce_diversity, calculate_novelty


class BayesianOptimizer:
    """
    Enhanced Bayesian optimization for materials discovery with multiple acquisition functions
    and support for exploration-exploitation trade-off.
    """
    
    def __init__(self, bounds=None, kernel=None, alpha=1e-6, n_restarts=10, normalize_y=True):
        """
        Initialize the Bayesian optimizer.
        
        Parameters:
        -----------
        bounds : dict
            Dictionary mapping feature names to (lower, upper) bounds
        kernel : sklearn.gaussian_process.kernels.Kernel
            Kernel for Gaussian Process
        alpha : float
            Noise parameter for GP
        n_restarts : int
            Number of restarts for GP optimizer
        normalize_y : bool
            Whether to normalize targets for GP
        """
        self.bounds = bounds
        
        # Default kernel: Matern 5/2 with noise component
        if kernel is None:
            self.kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        else:
            self.kernel = kernel
            
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts,
            normalize_y=normalize_y,
            random_state=42
        )
        
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the Gaussian Process model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features (n_samples, n_features)
        y : numpy.ndarray
            Target values (n_samples, n_targets)
        """
        # Handle 1D y
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Filter out NaN/Inf values
        valid_idx = np.isfinite(y).all(axis=1)
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        
        if len(X_valid) == 0:
            raise ValueError("No valid data points after filtering NaN/Inf values")
        
        self.X_train = X_valid
        self.y_train = y_valid
        
        self.gp.fit(X_valid, y_valid)
        self.is_fitted = True
        
        # Log kernel parameters after fitting
        st.info(f"Fitted GP model with kernel: {self.gp.kernel_}")
        
        return self
    
    def predict(self, X, return_std=False):
        """
        Make predictions with the Gaussian Process model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features (n_samples, n_features)
        return_std : bool
            Whether to return standard deviation
            
        Returns:
        --------
        y_pred : numpy.ndarray
            Predicted values
        y_std : numpy.ndarray (optional)
            Standard deviation of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.gp.predict(X, return_std=return_std)
    
    def acquisition_function(self, X, acquisition="UCB", xi=0.01, kappa=2.0, curiosity=0.0):
        """
        Compute acquisition function value for input points.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features (n_samples, n_features)
        acquisition : str
            Acquisition function type: "UCB", "EI", "PI", or "MaxEntropy"
        xi : float
            Exploration parameter for EI and PI
        kappa : float
            Exploration parameter for UCB
        curiosity : float
            Exploration vs exploitation parameter (-2 to +2)
            
        Returns:
        --------
        values : numpy.ndarray
            Acquisition function values (higher is better)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Get predictions and uncertainties
        mu, sigma = self.predict(X, return_std=True)
        
        # Ensure sigma is not zero
        sigma = np.maximum(sigma, 1e-6)
        
        # Adjust parameters based on curiosity
        kappa_adjusted = kappa * (1.0 + 0.5 * curiosity)
        xi_adjusted = xi * (1.0 + 0.5 * curiosity)
        
        # Compute acquisition function based on type
        if acquisition == "UCB":
            # Upper Confidence Bound
            return mu + kappa_adjusted * sigma
        
        elif acquisition == "EI":
            # Expected Improvement
            y_max = np.max(self.y_train)
            
            # Calculate improvement
            imp = mu - y_max - xi_adjusted
            
            # Calculate Z-score
            z = imp / sigma
            
            # Calculate Expected Improvement
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            
            # Set EI to 0 where it's negative
            ei[imp < 0] = 0.0
            
            return ei
        
        elif acquisition == "PI":
            # Probability of Improvement
            y_max = np.max(self.y_train)
            
            # Calculate Z-score
            z = (mu - y_max - xi_adjusted) / sigma
            
            # Calculate Probability of Improvement
            return norm.cdf(z)
        
        elif acquisition == "MaxEntropy":
            # Maximum Entropy (pure exploration)
            return sigma
        
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition}")
    
    def optimize(self, bounds_dict, n_restarts=20, acquisition="UCB", curiosity=0.0, 
                n_points=1, min_distance=0.1, max_iter=500):
        """
        Find the optimal point(s) according to the acquisition function.
        
        Parameters:
        -----------
        bounds_dict : dict
            Dictionary mapping feature names to (lower, upper) bounds
        n_restarts : int
            Number of restarts for acquisition function optimization
        acquisition : str
            Acquisition function type
        curiosity : float
            Exploration vs exploitation parameter
        n_points : int
            Number of diverse points to return
        min_distance : float
            Minimum distance between returned points
        max_iter : int
            Maximum number of iterations for optimization
            
        Returns:
        --------
        X_best : numpy.ndarray
            Optimal input points
        values_best : numpy.ndarray
            Acquisition function values at optimal points
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Extract bounds as list of (lower, upper) tuples
        feature_names = list(bounds_dict.keys())
        bounds = [bounds_dict[name] for name in feature_names]
        
        # Function to minimize (negative of acquisition function)
        def objective(x):
            x_reshaped = x.reshape(1, -1)
            return -self.acquisition_function(x_reshaped, acquisition, curiosity=curiosity).item()
        
        # Initialize best points and values
        X_best = []
        values_best = []
        
        # Run optimization from multiple starting points
        for _ in range(n_restarts):
            # Generate random starting point
            x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
            
            # Run optimization
            result = minimize(
                objective,
                x0,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": max_iter}
            )
            
            # Get optimal point and value
            if result.success:
                X_best.append(result.x)
                values_best.append(-result.fun)
        
        # Convert to arrays
        X_best = np.array(X_best)
        values_best = np.array(values_best)
        
        # Sort by acquisition function value
        sorted_indices = np.argsort(-values_best)
        X_best = X_best[sorted_indices]
        values_best = values_best[sorted_indices]
        
        # Enforce diversity between returned points
        if n_points > 1:
            # Initialize with the best point
            diverse_indices = [0]
            
            # Add diverse points
            for i in range(1, len(X_best)):
                is_diverse = True
                
                # Check distance to all selected points
                for j in diverse_indices:
                    distance = np.linalg.norm(X_best[i] - X_best[j])
                    
                    if distance < min_distance:
                        is_diverse = False
                        break
                
                # Add if diverse enough
                if is_diverse:
                    diverse_indices.append(i)
                    
                    # Break if we have enough points
                    if len(diverse_indices) >= n_points:
                        break
            
            # Select diverse points
            X_diverse = X_best[diverse_indices]
            values_diverse = values_best[diverse_indices]
            
            return X_diverse, values_diverse
        
        # Return top n_points
        return X_best[:n_points], values_best[:n_points]


def bayesian_optimization(train_inputs, train_targets, candidate_inputs, curiosity=0.0, 
                         acquisition="UCB", n_calls=50, min_distance=0.1, normalize=True):
    """
    Perform Bayesian optimization to select the next best sample for testing.
    
    Parameters:
    -----------
    train_inputs : numpy.ndarray
        Input features of labeled samples
    train_targets : numpy.ndarray
        Target values of labeled samples
    candidate_inputs : numpy.ndarray
        Input features of candidate samples
    curiosity : float
        Exploration vs exploitation parameter (-2 to +2)
    acquisition : str
        Acquisition function type: "UCB", "EI", "PI", or "MaxEntropy"
    n_calls : int
        Number of optimization iterations
    min_distance : float
        Minimum distance between selected samples
    normalize : bool
        Whether to normalize inputs and targets
        
    Returns:
    --------
    best_sample_idx : int
        Index of the best candidate sample
    """
    # Handle NaN/Inf values in targets
    train_targets = np.array(train_targets).reshape(-1, 1)
    valid_indices = np.isfinite(train_targets).all(axis=1)
    train_targets = train_targets[valid_indices]
    train_inputs = train_inputs[valid_indices]
    
    if len(train_inputs) == 0:
        st.warning("No valid training data. Selecting a random sample.")
        return np.random.randint(len(candidate_inputs))
    
    # Normalize the data
    if normalize:
        input_mean = np.mean(train_inputs, axis=0)
        input_std = np.std(train_inputs, axis=0) + 1e-8
        
        target_mean = np.mean(train_targets, axis=0)
        target_std = np.std(train_targets, axis=0) + 1e-8
        
        train_inputs_norm = (train_inputs - input_mean) / input_std
        train_targets_norm = (train_targets - target_mean) / target_std
        
        candidate_inputs_norm = (candidate_inputs - input_mean) / input_std
    else:
        train_inputs_norm = train_inputs
        train_targets_norm = train_targets
        candidate_inputs_norm = candidate_inputs
    
    # Select kernel based on data size
    if len(train_inputs) < 10:
        # For very small datasets, use a simpler RBF kernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    else:
        # For larger datasets, use Matern kernel which is better for physical processes
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
    
    # Initialize and fit Bayesian optimizer
    optimizer = BayesianOptimizer(kernel=kernel)
    optimizer.fit(train_inputs_norm, train_targets_norm)
    
    # Compute acquisition function values for all candidates
    acq_values = optimizer.acquisition_function(
        candidate_inputs_norm, 
        acquisition=acquisition, 
        curiosity=curiosity
    )
    
    # Calculate novelty scores
    novelty_scores = calculate_novelty(candidate_inputs_norm, train_inputs_norm)
    
    # Adjust acquisition values with novelty if curiosity is positive
    if curiosity > 0:
        acq_values = acq_values.flatten() + 0.2 * curiosity * novelty_scores
    
    # Sort candidates by acquisition value
    sorted_indices = np.argsort(-acq_values)
    
    # Get the best candidate index
    best_candidates = sorted_indices[:min(10, len(sorted_indices))]
    
    # Enforce diversity with previously selected samples
    diverse_candidates = enforce_diversity(
        candidate_inputs_norm[best_candidates], 
        train_inputs_norm, 
        min_distance
    )
    
    if len(diverse_candidates) > 0:
        # Find index of the most diverse candidate among top candidates
        for i, candidate_idx in enumerate(best_candidates):
            if any(np.allclose(candidate_inputs_norm[candidate_idx], diverse_candidate) 
                  for diverse_candidate in diverse_candidates):
                best_sample_idx = candidate_idx
                break
        else:
            # If no match found, use the best candidate
            best_sample_idx = best_candidates[0]
    else:
        # If no diverse candidates, use the best candidate
        best_sample_idx = best_candidates[0]
    
    return best_sample_idx


def multi_objective_bayesian_optimization(train_inputs, train_targets, candidate_inputs, weights, 
                                       max_or_min, curiosity=0.0, acquisition="UCB", n_calls=50):
    """
    Perform multi-objective Bayesian optimization for materials discovery.
    
    Parameters:
    -----------
    train_inputs : numpy.ndarray
        Input features of labeled samples
    train_targets : numpy.ndarray
        Target values of labeled samples (multi-objective)
    candidate_inputs : numpy.ndarray
        Input features of candidate samples
    weights : numpy.ndarray
        Importance weights for each objective
    max_or_min : list
        Direction of optimization for each objective ('max' or 'min')
    curiosity : float
        Exploration vs exploitation parameter (-2 to +2)
    acquisition : str
        Acquisition function type
    n_calls : int
        Number of optimization iterations
        
    Returns:
    --------
    best_sample_idx : int
        Index of the best candidate sample
    """
    # Handle data dimensionality
    train_targets = np.array(train_targets)
    weights = np.array(weights)
    
    if train_targets.ndim == 1:
        train_targets = train_targets.reshape(-1, 1)
        weights = np.array([1.0])
        max_or_min = ["max"]
    
    # Filter out invalid targets
    valid_indices = np.isfinite(train_targets).all(axis=1)
    train_targets = train_targets[valid_indices]
    train_inputs = train_inputs[valid_indices]
    
    if len(train_inputs) == 0:
        st.warning("No valid training data. Selecting a random sample.")
        return np.random.randint(len(candidate_inputs))
    
    # Normalize inputs
    input_mean = np.mean(train_inputs, axis=0)
    input_std = np.std(train_inputs, axis=0) + 1e-8
    
    train_inputs_norm = (train_inputs - input_mean) / input_std
    candidate_inputs_norm = (candidate_inputs - input_mean) / input_std
    
    # Initialize model for each objective
    n_objectives = train_targets.shape[1]
    models = []
    
    for i in range(n_objectives):
        # Get target for this objective
        y = train_targets[:, i].reshape(-1, 1)
        
        # Filter out invalid values
        valid_indices = np.isfinite(y.flatten())
        y_valid = y[valid_indices]
        X_valid = train_inputs_norm[valid_indices]
        
        if len(X_valid) == 0 or len(y_valid) == 0:
            continue
        
        # Choose kernel based on data size
        if len(X_valid) < 10:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        else:
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        
        # Normalize target
        y_mean = np.mean(y_valid)
        y_std = np.std(y_valid) + 1e-8
        y_norm = (y_valid - y_mean) / y_std
        
        # Fit model
        model = BayesianOptimizer(kernel=kernel)
        model.fit(X_valid, y_norm)
        
        # Store model along with normalization parameters
        models.append((model, y_mean, y_std))
    
    if not models:
        st.warning("Failed to build models for all objectives. Selecting a random sample.")
        return np.random.randint(len(candidate_inputs))
    
    # Compute weighted acquisition function
    acq_values_total = np.zeros(len(candidate_inputs_norm))
    uncertainty_values_total = np.zeros(len(candidate_inputs_norm))
    
    for i, (model, y_mean, y_std) in enumerate(models):
        # Get prediction and uncertainty
        mu, sigma = model.predict(candidate_inputs_norm, return_std=True)
        
        # Compute acquisition function
        acq_values = model.acquisition_function(
            candidate_inputs_norm, 
            acquisition=acquisition, 
            curiosity=curiosity
        )
        
        # Apply direction
        if max_or_min[i].lower() == "min":
            acq_values = -acq_values
        
        # Add weighted acquisition values
        acq_values_total += weights[i] * acq_values.flatten()
        uncertainty_values_total += weights[i] * sigma.flatten()
    
    # Calculate novelty
    novelty_scores = calculate_novelty(candidate_inputs_norm, train_inputs_norm)
    
    # Adjust with novelty if curiosity is positive
    if curiosity > 0:
        acq_values_total += 0.2 * curiosity * novelty_scores
    
    # Get best candidate index
    best_sample_idx = np.argmax(acq_values_total)
    
    return best_sample_idx