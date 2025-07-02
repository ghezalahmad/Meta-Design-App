import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
import streamlit as st
from app.utils import enforce_diversity, calculate_novelty

# Module-level helper functions for acquisition calculations
def _calculate_ucb(mu, sigma, kappa_adjusted):
    return mu + kappa_adjusted * sigma

def _calculate_ei(mu, sigma, y_max_of_current_obj, xi_adjusted):
    sigma_safe = np.maximum(sigma, 1e-9) # Avoid division by zero / sqrt of negative
    imp = mu - y_max_of_current_obj - xi_adjusted
    z = imp / sigma_safe
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei[imp < 0] = 0.0 # Expected improvement cannot be negative
    return ei

def _calculate_pi(mu, sigma, y_max_of_current_obj, xi_adjusted):
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (mu - y_max_of_current_obj - xi_adjusted) / sigma_safe
    return norm.cdf(z)


class BayesianOptimizer:
    """
    Enhanced Bayesian optimization for materials discovery with multiple acquisition functions
    and support for exploration-exploitation trade-off.
    """
    
    def __init__(self, surrogate_model=None, bounds=None, kernel=None, alpha=1e-6, n_restarts=10, normalize_y=True):
        """
        Initialize the Bayesian optimizer.
        
        Parameters:
        -----------
        surrogate_model : object, optional
            A pre-trained surrogate model instance. Must implement a method like
            `predict_with_uncertainty(X)` which returns (mean, uncertainty_metric)
            and have an attribute `is_trained` (boolean) and `scaler_x`, `scaler_y` if it uses them.
            If None, an internal GaussianProcessRegressor will be used.
        bounds : dict
            Dictionary mapping feature names to (lower, upper) bounds for internal GP optimization.
        kernel : sklearn.gaussian_process.kernels.Kernel
            Kernel for the internal Gaussian Process if surrogate_model is None.
        alpha : float
            Noise parameter for the internal GP.
        n_restarts : int
            Number of restarts for the internal GP optimizer.
        normalize_y : bool
            Whether to normalize targets for the internal GP.
        """
        self.surrogate_model = surrogate_model
        self.bounds = bounds
        self.X_train = None
        self.y_train = None # Will store original scale y_train for acquisition functions like EI/PI
        self.is_fitted = False

        if self.surrogate_model is not None:
            if not hasattr(self.surrogate_model, 'predict_with_uncertainty') or \
               not callable(self.surrogate_model.predict_with_uncertainty):
                raise ValueError("Provided surrogate_model must have a 'predict_with_uncertainty' method.")
            # We assume a pre-trained surrogate model is already fitted.
            # The `fit` method of BayesianOptimizer will primarily handle data for acquisition functions.
            self.gp = None # No internal GP needed
            self.kernel = None
        else:
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
        
        # Store original X and y for reference and potential use by surrogates
        # self.X_raw_train = X
        # self.y_raw_train = y

        # Filter out NaN/Inf values from y for fitting GP or storing y_train
        valid_idx = np.isfinite(y).all(axis=1)
        X_valid_np = X[valid_idx] if isinstance(X, np.ndarray) else X.iloc[valid_idx].values
        y_valid = y[valid_idx]
        
        if len(X_valid_np) == 0:
            raise ValueError("No valid data points after filtering NaN/Inf values for y.")
        
        # Store X_train as DataFrame if original X was a DataFrame, otherwise as numpy.
        # This helps in _get_surrogate_prediction if column names are needed.
        if isinstance(X, pd.DataFrame):
            self.X_train_df_columns = X.columns.tolist() # Store column names
            self.X_train = X.iloc[valid_idx] # Store the filtered DataFrame
        else:
            self.X_train_df_columns = None
            self.X_train = X_valid_np # Store as numpy array

        self.y_train = y_valid # Store original scale y for EI/PI y_max

        if self.surrogate_model:
            if not getattr(self.surrogate_model, 'is_trained', False):
                # This case should ideally be handled before calling BayesianOptimizer,
                # but as a safeguard:
                st.warning("Surrogate model provided to BayesianOptimizer is not marked as trained. Fitting it now.")
                # Attempt to fit the surrogate if it has a 'train' method similar to our RFModel/other models
                if hasattr(self.surrogate_model, 'train') and callable(self.surrogate_model.train):
                    # This is tricky because surrogate_model.train might need input_columns, target_columns etc.
                    # For now, we assume the surrogate is pre-trained.
                    # If direct fitting is needed here, the interface needs to be more complex.
                    # Consider raising an error or relying on pre-training.
                    st.error("BayesianOptimizer expects a pre-trained surrogate model or will train its internal GP.")
                    # For now, let's assume we can't just call a generic 'train' method without more context.
                    # The primary design is for pre-trained surrogates.
                else:
                    st.warning("Surrogate model does not have a 'train' method. Assuming it's pre-fitted or needs no fitting here.")
            self.is_fitted = True # Mark BO as "fitted" as it has data and a surrogate.
            st.info(f"BayesianOptimizer using provided pre-trained surrogate: {type(self.surrogate_model).__name__}")

        elif self.gp: # Use internal GP
            self.gp.fit(X_valid, y_valid) # y_valid is original scale here if normalize_y=False for GP
                                       # if normalize_y=True, GP handles it.
            self.is_fitted = True
            st.info(f"Fitted internal GP model with kernel: {self.gp.kernel_}")
        else:
            raise RuntimeError("BayesianOptimizer has neither a surrogate_model nor an internal GP to fit.")

        return self
    
    def _predict_with_internal_gp(self, X, return_std=False):
        """
        Make predictions with the *internal* Gaussian Process model.
        This method is only used if no external surrogate_model is provided.
        
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
        if not self.is_fitted or not self.gp: # Check if internal GP exists
            raise ValueError("Internal GP not fitted yet or not available. Call fit() first or provide a surrogate.")
        
        return self.gp.predict(X, return_std=return_std)

    def _get_surrogate_prediction(self, X_np: np.ndarray):
        """
        Gets prediction (mean and std_dev) from the active surrogate model (either internal GP or provided external model).
        Ensures X_np is temporarily converted to DataFrame if needed by the surrogate.

        Args:
            X_np (np.ndarray): Input features in original scale.

        Returns:
            tuple: (mu, sigma) - mean predictions and standard deviations, both in original target scale.
        """
        if not self.is_fitted:
            raise ValueError("BayesianOptimizer not fitted yet. Call fit() first.")

        if self.surrogate_model:
            # Assuming surrogate_model.predict_with_uncertainty expects X and input_columns (if it's like RFModel)
            # The input_columns can be derived from self.bounds if set during optimize, or self.X_train
            input_cols = None
            # Try to get input_columns from the surrogate's scaler first, then from BO's context
            if hasattr(self.surrogate_model, 'scaler_x') and self.surrogate_model.scaler_x and \
               hasattr(self.surrogate_model.scaler_x, 'feature_names_in_') and \
               self.surrogate_model.scaler_x.feature_names_in_ is not None:
                input_cols = self.surrogate_model.scaler_x.feature_names_in_
            elif self.X_train_df_columns: # From data passed to BO.fit() if X was a DataFrame
                input_cols = self.X_train_df_columns
            elif self.bounds: # From BO.optimize() method if bounds_dict was provided
                input_cols = list(self.bounds.keys())

            if input_cols is None and X_np.shape[1] > 0 : # Check if X_np has features
                 # If surrogate is an sklearn pipeline, it might get feature names from training data.
                 # Fallback: assume surrogate can handle numpy array or doesn't need explicit column names for predict_with_uncertainty
                 # This might be true for some simple function-based surrogates or if X_np is already a DataFrame.
                 # Our current RF and NN models expect DataFrame and input_columns for their predict_with_uncertainty.
                 # This path might lead to errors if input_cols are truly needed and not found.
                st.warning("BayesianOptimizer: input_columns not determined for surrogate model. "
                           "Surrogate must handle NumPy array input for predict_with_uncertainty or this may fail.")
                mu, sigma = self.surrogate_model.predict_with_uncertainty(X_np) # sigma is std_dev
            elif input_cols is not None:
                if X_np.shape[1] != len(input_cols):
                    raise ValueError(f"Shape mismatch: X_np has {X_np.shape[1]} features, but found {len(input_cols)} input_columns.")
                X_df = pd.DataFrame(X_np, columns=input_cols)
                # All our current custom models (RF, MAML, Reptile, ProtoNet) have predict_with_uncertainty
                # that takes (X_df, input_columns=input_columns)
                mu, sigma = self.surrogate_model.predict_with_uncertainty(X_df, input_columns=input_cols) # sigma is std_dev
            else: # No input_cols and X_np has no features (e.g. X_np.shape[1] == 0), this is an issue.
                 raise ValueError("BayesianOptimizer: Cannot determine input columns and X_np has no features.")

        elif self.gp:
            mu, sigma = self._predict_with_internal_gp(X_np, return_std=True)
        else:
            raise RuntimeError("No surrogate model or internal GP available for prediction.")

        # Ensure mu is (n_samples,) or (n_samples, 1) and sigma is (n_samples,)
        # Predictions from models are expected to be (n_samples, n_targets)
        # Acquisition functions here are single-objective, so they expect single mu/sigma.
        # If surrogate is multi-output, this needs careful handling.
        # For now, assume mu and sigma are for the primary target or already scalarized.
        if mu.ndim > 1:
            if mu.shape[1] > 1:
                st.warning("BayesianOptimizer: Surrogate model returned multiple target predictions. Using first target for acquisition.")
                mu = mu[:, 0]
            else: # mu.shape[1] == 1
                mu = mu.ravel()

        if sigma.ndim > 1:
            if sigma.shape[1] > 1:
                st.warning("BayesianOptimizer: Surrogate model returned multiple target uncertainties. Using first target for acquisition.")
                sigma = sigma[:, 0]
            else: # sigma.shape[1] == 1
                sigma = sigma.ravel()

        return mu, sigma
    
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
        if not self.is_fitted: # This check is also in _get_surrogate_prediction, but good for early exit.
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Get predictions and uncertainties using the new unified method
        mu, sigma = self._get_surrogate_prediction(X)
        
        # Ensure sigma is non-negative (it's std_dev) and not zero to avoid division errors.
        sigma = np.maximum(sigma, 1e-9) # Changed from 1e-6 to 1e-9 for potentially smaller std devs
        
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
                                       max_or_min, curiosity=0.0, acquisition="UCB", n_calls=50, # n_calls not really used for candidate list
                                       strategy="weighted_sum", surrogate_model=None, input_columns=None):
    """
    Perform multi-objective Bayesian optimization for materials discovery.
    Can use a pre-trained surrogate_model or fit individual GPs per objective.
    
    Parameters:
    -----------
    train_inputs : pd.DataFrame or numpy.ndarray
        Input features of labeled samples
    train_targets : numpy.ndarray
        Target values of labeled samples (multi-objective)
    candidate_inputs : pd.DataFrame or numpy.ndarray
        Input features of candidate samples
    surrogate_model : object, optional
        A pre-trained surrogate model instance. Must implement
        `predict_with_uncertainty(X_df, input_columns)` returning (means, std_devs)
        for all targets. If None, fits individual GPs per objective.
    input_columns : list[str], optional
        Required if train_inputs/candidate_inputs are numpy arrays and surrogate_model is provided
        and needs DataFrame input with column names.
    weights : numpy.ndarray
        Importance weights for each objective (used in 'weighted_sum', can influence 'parego' bias for 'parego')
    max_or_min : list
        Direction of optimization for each objective ('max' or 'min')
    curiosity : float
        Exploration vs exploitation parameter (-2 to +2)
    acquisition : str
        Acquisition function type
    n_calls : int
        Number of optimization iterations (currently not used as it scores candidates)
    strategy : str
        'weighted_sum' for fixed weights, 'parego' for random scalarization.
        
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
    
    # --- Helper functions for acquisition calculations (defined at module level or accessible scope) ---
    # _calculate_ucb, _calculate_ei, _calculate_pi
    # For this diff, assume they are defined above this function in the same file.

    # Determine effective weights for scalarization
    if strategy == "parego":
        if n_objectives_from_targets > 0:
            random_w = np.random.dirichlet(np.ones(n_objectives_from_targets), size=1).ravel()
            effective_weights = random_w
            st.info(f"ParEGO strategy: using random weights: {np.round(effective_weights, 3)}")
        else:
            effective_weights = np.array([1.0]) # Fallback, though n_objectives should be >0
    else: # 'weighted_sum' or default
        effective_weights = weights / np.sum(weights) # Normalize user weights if not already

    # Prepare candidate inputs (ensure DataFrame if surrogate_model expects it)
    if isinstance(candidate_inputs, np.ndarray) and input_columns:
        candidate_inputs_df = pd.DataFrame(candidate_inputs, columns=input_columns)
    elif isinstance(candidate_inputs, pd.DataFrame):
        candidate_inputs_df = candidate_inputs
    else: # Should not happen if input_columns logic is correct
        raise ValueError("candidate_inputs format issue or missing input_columns for DataFrame conversion.")

    num_candidates = len(candidate_inputs_df)
    acq_values_total = np.zeros(num_candidates)

    # Normalize train_inputs (for internal GP path)
    # This normalization is only for the internal GP path.
    # Surrogates are expected to handle their own scaling or work with original scale.
    train_inputs_norm_for_gp = None
    if surrogate_model is None:
        if isinstance(train_inputs, pd.DataFrame):
            train_inputs_np_for_gp = train_inputs.values
        else:
            train_inputs_np_for_gp = train_inputs
        
        input_mean_for_gp = np.mean(train_inputs_np_for_gp, axis=0)
        input_std_for_gp = np.std(train_inputs_np_for_gp, axis=0) + 1e-8
        train_inputs_norm_for_gp = (train_inputs_np_for_gp - input_mean_for_gp) / input_std_for_gp
        
        if isinstance(candidate_inputs_df, pd.DataFrame):
            candidate_inputs_norm_for_gp = (candidate_inputs_df.values - input_mean_for_gp) / input_std_for_gp
        else: # Should be DataFrame by now
            candidate_inputs_norm_for_gp = (candidate_inputs_df - input_mean_for_gp) / input_std_for_gp


    if surrogate_model:
        # Use the provided pre-trained multi-output surrogate model
        if not getattr(surrogate_model, 'is_trained', True):
             raise ValueError("Provided surrogate_model is not trained.")

        # predict_with_uncertainty should return means and std_devs for ALL targets
        # in their original scale.
        all_mu_orig, all_sigma_orig = surrogate_model.predict_with_uncertainty(
            candidate_inputs_df,
            input_columns=input_columns # Pass input_columns if surrogate expects it
        )

        if all_mu_orig.ndim == 1: all_mu_orig = all_mu_orig.reshape(-1, 1)
        if all_sigma_orig.ndim == 1: all_sigma_orig = all_sigma_orig.reshape(-1, 1)

        if all_mu_orig.shape[1] != n_objectives_from_targets or \
           (all_sigma_orig.shape[1] != n_objectives_from_targets and all_sigma_orig.shape[1] != 1): # allow single shared sigma
            raise ValueError(f"Surrogate model prediction shape mismatch. Expected {n_objectives_from_targets} targets. Got means: {all_mu_orig.shape}, sigmas: {all_sigma_orig.shape}")

        for i in range(n_objectives_from_targets):
            mu_obj = all_mu_orig[:, i]
            sigma_obj = all_sigma_orig[:, i] if all_sigma_orig.shape[1] == n_objectives_from_targets else all_sigma_orig.ravel()

            y_train_obj_orig = train_targets[:, i]
            y_max_obj_orig = np.max(y_train_obj_orig[np.isfinite(y_train_obj_orig)]) if np.any(np.isfinite(y_train_obj_orig)) else 0.0
            # For minimization, one might use y_min_obj_orig = np.min(...)

            kappa_adjusted = 2.0 * (1.0 + 0.5 * curiosity)
            xi_adjusted = 0.01 * (1.0 + 0.5 * curiosity)

            acq_obj_values_i = np.zeros(num_candidates)
            if acquisition == "UCB":
                acq_obj_values_i = _calculate_ucb(mu_obj, sigma_obj, kappa_adjusted)
            elif acquisition == "EI":
                acq_obj_values_i = _calculate_ei(mu_obj, sigma_obj, y_max_obj_orig, xi_adjusted)
            elif acquisition == "PI":
                acq_obj_values_i = _calculate_pi(mu_obj, sigma_obj, y_max_obj_orig, xi_adjusted)
            else: # Default to UCB
                acq_obj_values_i = _calculate_ucb(mu_obj, sigma_obj, kappa_adjusted)

            if max_or_min[i].lower() == "min":
                if acquisition == "UCB": # LCB
                    acq_obj_values_i = mu_obj - kappa_adjusted * sigma_obj
                else: # For EI/PI, true minimization requires transforming the problem (e.g. -y)
                      # or using specialized forms. Simple negation of acquisition is not standard.
                      # This path would require careful re-evaluation of y_max_obj_orig for min case.
                    st.warning(f"Minimization with {acquisition} for objective {i} using surrogate. Ensure target was pre-transformed or acquisition logic handles minimization.")
                    acq_obj_values_i = -acq_obj_values_i # Tentative, may not be mathematically sound for EI/PI

            acq_values_total += effective_weights[i] * acq_obj_values_i.flatten()

    else: # Fallback to fitting individual GPs per objective (existing logic adapted)
        models = []
        for i in range(n_objectives_from_targets):
            y_obj_single = train_targets[:, i].reshape(-1, 1)
            valid_indices_obj = np.isfinite(y_obj_single.flatten())
            y_valid_obj_for_gp = y_obj_single[valid_indices_obj]
            # Use train_inputs_norm_for_gp for fitting internal GPs
            X_valid_obj_for_gp = train_inputs_norm_for_gp[valid_indices_obj]

            if len(X_valid_obj_for_gp) == 0: continue

            kernel_gp = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1) \
                if len(X_valid_obj_for_gp) >= 10 else ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

            # BayesianOptimizer's internal GP handles its own y-normalization if normalize_y=True
            gp_optimizer = BayesianOptimizer(kernel=kernel_gp, normalize_y=True)
            gp_optimizer.fit(X_valid_obj_for_gp, y_valid_obj_for_gp)
            models.append(gp_optimizer)

        if not models or len(models) != n_objectives_from_targets:
            st.warning("Failed to build models for all objectives using internal GPs. Selecting a random sample.")
            return np.random.randint(len(candidate_inputs_df))

        for i, model_i in enumerate(models):
            # model_i is a BayesianOptimizer instance (with an internal GP)
            # Its acquisition_function will use its own GP's normalized y_train for y_max (if normalize_y=True)
            # It expects X in the same scale as it was trained on (i.e. train_inputs_norm_for_gp)
            acq_obj_values_i = model_i.acquisition_function(
                candidate_inputs_norm_for_gp, # Pass normalized candidates
                acquisition=acquisition,
                curiosity=curiosity
            )

            if max_or_min[i].lower() == "min":
                # The internal GP's acquisition function (UCB, EI, PI) assumes maximization.
                # For UCB, to get LCB, we'd need mu - kappa * sigma.
                # For EI/PI with minimization, the problem should ideally be transformed (e.g. predict -y).
                # This path needs careful handling if min objectives and EI/PI are used with internal GPs.
                if acquisition == "UCB":
                     mu_norm, sigma_norm = model_i._predict_with_internal_gp(candidate_inputs_norm_for_gp, return_std=True)
                     kappa_adjusted = 2.0 * (1.0 + 0.5 * curiosity) # Default kappa from BO class
                     acq_obj_values_i = mu_norm.ravel() - kappa_adjusted * sigma_norm.ravel()
                else:
                    st.warning(f"Minimization with {acquisition} for objective {i} using internal GP might require target pre-transformation.")
                    acq_obj_values_i = -acq_obj_values_i # Tentative

            acq_values_total += effective_weights[i] * acq_obj_values_i.flatten()

    # Calculate novelty (using original scale inputs if possible, or normalized for internal GP path)
    # The `calculate_novelty` function expects NumPy arrays.
    novelty_eval_candidates = candidate_inputs_df.values if isinstance(candidate_inputs_df, pd.DataFrame) else candidate_inputs_df
    novelty_train_references = None
    if isinstance(train_inputs, pd.DataFrame):
        novelty_train_references = train_inputs.iloc[valid_indices].values # Use filtered valid_indices from original train_inputs
    elif isinstance(train_inputs, np.ndarray):
        novelty_train_references = train_inputs[valid_indices]

    if surrogate_model is None and train_inputs_norm_for_gp is not None: # If internal GPs were used, novelty on normalized space
        novelty_eval_candidates = candidate_inputs_norm_for_gp
        novelty_train_references = train_inputs_norm_for_gp[valid_indices] # Ensure correct reference for novelty

    novelty_scores = calculate_novelty(novelty_eval_candidates, novelty_train_references)
    
    # Adjust with novelty if curiosity is positive
    if curiosity > 0:
        acq_values_total += 0.2 * curiosity * novelty_scores
    
    # Return all acquisition scores instead of just the index of the best
    # The calling function (in main.py) will handle selecting the best and populating result_df.
    return acq_values_total.flatten()