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
    
    def __init__(self, surrogate_model=None, bounds=None, kernel=None, alpha=1e-6, n_restarts=10, normalize_y=True, target_index_for_surrogate=0):
        """
        Initialize the Bayesian optimizer.
        
        Parameters:
        -----------
        surrogate_model : object, optional
            A pre-trained surrogate model instance. If provided, it must adhere to the following interface:
            - `is_trained` (bool attribute): Indicates if the model has been trained.
            - `predict_with_uncertainty(X, input_columns=None, num_samples=None)` (method):
                - `X`: Input data (pd.DataFrame or np.ndarray).
                - `input_columns` (list[str], optional): Names of input columns if X is a DataFrame or if needed by the model.
                - `num_samples` (int, optional): Hint for the number of MC samples if the model uses sampling for uncertainty/posterior.
                - Returns: A tuple.
                    - If `num_samples` (or an internal equivalent for posterior sampling) is used and successful:
                      `(mean_preds, std_devs, posterior_samples)`
                    - Else:
                      `(mean_preds, std_devs)`
                - `mean_preds` (np.ndarray): Shape (n_points, n_surrogate_targets).
                - `std_devs` (np.ndarray): Shape (n_points, n_surrogate_targets) or (n_points, 1).
                - `posterior_samples` (np.ndarray, optional): Shape (n_mc_samples, n_points, n_surrogate_targets).
                - All predictions and uncertainties should be in the *original scale* of the target(s).
            - `scaler_x` (object, optional): If present and has `feature_names_in_`, these will be used as input_columns.
            The BayesianOptimizer's `fit` method does NOT train this external surrogate.
            If None, an internal GaussianProcessRegressor will be used for single-objective optimization.
        bounds : dict, optional
            Dictionary mapping feature names to (lower, upper) bounds. Used if `X_train` passed to `fit`
            is a NumPy array, or for the `optimize` method's bounds.
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
        self.target_index = target_index_for_surrogate

        if self.surrogate_model is not None:
            if not hasattr(self.surrogate_model, 'predict_with_uncertainty') or \
               not callable(self.surrogate_model.predict_with_uncertainty):
                raise ValueError("Provided surrogate_model must have a 'predict_with_uncertainty' method.")
            self.gp = None
            self.kernel = None
        else:
            if kernel is None:
                self.kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
            else:
                self.kernel = kernel
            self.gp = GaussianProcessRegressor(
                kernel=self.kernel, alpha=alpha, n_restarts_optimizer=n_restarts,
                normalize_y=normalize_y, random_state=42
            )
    
    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray):
        """
        Fits the Bayesian Optimizer.

        If an internal Gaussian Process (GP) is used (i.e., `surrogate_model` was None
        at initialization), this method fits the GP to the provided data `X` and `y`.

        If an external `surrogate_model` was provided, this method primarily stores `X`
        and `y` (specifically `y_train` in original scale) for use by acquisition
        functions like Expected Improvement (EI) and Probability of Improvement (PI)
        which require knowledge of the best observed `y` value. The external
        surrogate model is assumed to be already trained.

        Args:
            X: Input features. Can be a pandas DataFrame or a NumPy array.
               Shape (n_samples, n_features).
            y: Target values. NumPy array, shape (n_samples,) or (n_samples, n_targets).
               For single-objective BO, if `y` is multi-target, the target specified
               by `self.target_index` during `__init__` will be used implicitly by
               acquisition functions if the surrogate model returns sliced outputs,
               or `y_train` will store all targets for reference.
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

    def _get_surrogate_prediction(self, X_np: np.ndarray, return_posterior_samples: bool = False,
                                   n_posterior_samples: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Retrieves predictions (mean, std_dev) and optionally posterior samples
        from the active surrogate model (either internal GP or a provided external model).

        Handles conversion of `X_np` to a pandas DataFrame if the surrogate model
        is expected to require it (e.g., based on `input_columns` derived from
        training data or bounds).

        If the surrogate model is multi-output, this method will select the target
        slice corresponding to `self.target_index` for `mu`, `sigma`, and
        `posterior_samples` before returning them.

        Args:
            X_np: Input features as a NumPy array, in their original scale.
                  Shape (n_points, n_features).
            return_posterior_samples: If True, attempts to obtain posterior samples
                                      from the surrogate.
            n_posterior_samples: The number of posterior samples to request if
                                 `return_posterior_samples` is True.

        Returns:
            A tuple `(mu, sigma, posterior_samples)`:
            - `mu`: Mean predictions, shape (n_points,).
            - `sigma`: Standard deviation of predictions, shape (n_points,).
            - `posterior_samples`: Posterior samples, shape (n_posterior_samples, n_points)
                                   if `return_posterior_samples` is True and samples are
                                   obtained; otherwise None. All outputs are for the
                                   selected `self.target_index` and in original target scale.
        """
        if not self.is_fitted:
            raise ValueError("BayesianOptimizer not fitted yet. Call fit() first.")

        posterior_samples = None

        if self.surrogate_model:
            input_cols = None
            if hasattr(self.surrogate_model, 'scaler_x') and self.surrogate_model.scaler_x and \
               hasattr(self.surrogate_model.scaler_x, 'feature_names_in_') and \
               self.surrogate_model.scaler_x.feature_names_in_ is not None:
                input_cols = self.surrogate_model.scaler_x.feature_names_in_
            elif self.X_train_df_columns:
                input_cols = self.X_train_df_columns
            elif self.bounds:
                input_cols = list(self.bounds.keys())

            X_input_for_surrogate = X_np
            if input_cols:
                if X_np.shape[1] != len(input_cols):
                    raise ValueError(f"Shape mismatch: X_np has {X_np.shape[1]} features, but found {len(input_cols)} input_columns.")
                X_input_for_surrogate = pd.DataFrame(X_np, columns=input_cols)
            elif not isinstance(X_np, pd.DataFrame):
                 st.warning("BayesianOptimizer: input_columns not determined for surrogate, and X_np is not DataFrame. Surrogate must handle raw NumPy array.")


            # Check if the surrogate's predict_with_uncertainty supports returning samples
            # This is a bit of duck typing based on our MAMLModel change.
            # A more robust way would be an explicit capability flag on the surrogate.
            try:
                # Attempt to unpack three values if return_posterior_samples is True
                if return_posterior_samples and hasattr(self.surrogate_model, 'predict_with_uncertainty'):
                    # Assuming our NN models now might take num_samples for their predict_with_uncertainty
                    # to control the number of MC samples for the posterior.
                    # This is a bit of an API assumption for external surrogates.
                    # For now, let's assume if the method exists, it behaves like our updated MAMLModel.
                    pred_output = self.surrogate_model.predict_with_uncertainty(
                        X_input_for_surrogate,
                        input_columns=input_cols, # Pass None if not determined, surrogate must handle
                        num_samples=n_posterior_samples # For MAMLModel and similar
                    )
                    if len(pred_output) == 3:
                        mu, sigma, posterior_samples = pred_output
                    elif len(pred_output) == 2:
                        mu, sigma = pred_output
                        if return_posterior_samples:
                            st.warning("Surrogate model's predict_with_uncertainty did not return posterior samples as requested.")
                    else:
                        raise ValueError("Surrogate model's predict_with_uncertainty returned unexpected number of values.")
                elif hasattr(self.surrogate_model, 'predict_with_uncertainty'):
                     mu, sigma = self.surrogate_model.predict_with_uncertainty(
                        X_input_for_surrogate,
                        input_columns=input_cols
                        # num_samples not passed if not requesting posterior
                    )
                else: # Fallback if no predict_with_uncertainty, try standard predict and assign high uncertainty
                    mu = self.surrogate_model.predict(X_input_for_surrogate)
                    sigma = np.ones_like(mu) * np.std(mu) # A very rough placeholder for uncertainty
                    st.warning("Surrogate model lacks 'predict_with_uncertainty'. Using 'predict' and estimating sigma.")

            except TypeError as e: # Handles cases where num_samples is not an arg for some surrogates
                if "unexpected keyword argument 'num_samples'" in str(e) and hasattr(self.surrogate_model, 'predict_with_uncertainty'):
                    mu, sigma = self.surrogate_model.predict_with_uncertainty(X_input_for_surrogate, input_columns=input_cols)
                    if return_posterior_samples:
                        st.warning("Surrogate model's predict_with_uncertainty does not support 'num_samples' and did not return posterior samples.")
                else:
                    raise e # Re-raise other TypeErrors

        elif self.gp:
            if return_posterior_samples:
                mu = self.gp.predict(X_np) # mean
                posterior_samples_gp = self.gp.sample_y(X_np, n_samples=n_posterior_samples, random_state=np.random.randint(10000))
                # sample_y returns (n_points, n_samples). Need (n_samples, n_points)
                posterior_samples = posterior_samples_gp.T
                if posterior_samples.ndim == 2 and self.gp.y_train_.ndim > 1 and self.gp.y_train_.shape[1] > 1:
                    # If internal GP is multi-output, sample_y might be more complex.
                    # For now, assume single primary output for GP's sample_y in this context or it handles multi-output correctly.
                    # If gp.y_train_ was (n_samples, n_targets), sample_y might be (n_points, n_targets, n_mc_samples)
                    # The current sklearn GP sample_y is (n_points, n_mc_samples) for single target.
                    # And (n_points, n_targets, n_mc_samples) for multi-target if kernel supports it.
                    # We need (n_mc_samples, n_points, n_targets)
                    if posterior_samples.shape[-1] == X_np.shape[0] and self.gp.y_train_.shape[1] > 1: # (n_targets, n_mc_samples, n_points) -> (n_mc_samples, n_points, n_targets)
                         # This part needs careful check of sklearn GP's sample_y for multi-output
                         st.warning("Handling of multi-output GP posterior samples needs verification.")
                         # Assuming sample_y for multi-output GP might be (n_points, n_targets, n_mc_samples)
                         # We need to transpose to (n_mc_samples, n_points, n_targets)
                         if posterior_samples_gp.ndim ==3 and posterior_samples_gp.shape[0] == X_np.shape[0] and posterior_samples_gp.shape[2] == n_posterior_samples:
                              posterior_samples = np.transpose(posterior_samples_gp, (2,0,1))


                sigma = np.std(posterior_samples, axis=0) # Calculate sigma from samples
            else:
                mu, sigma = self._predict_with_internal_gp(X_np, return_std=True)
        else:
            raise RuntimeError("No surrogate model or internal GP available for prediction.")

        # Ensure mu is (n_samples,) or (n_samples, 1) and sigma is (n_samples,) for single-objective BO.
        # If surrogate is multi-output, select the target specified by self.target_index.
        if mu.ndim > 1 and mu.shape[1] > 1:
            if self.target_index >= mu.shape[1]:
                st.error(f"target_index {self.target_index} is out of bounds for surrogate model with {mu.shape[1]} outputs. Using index 0.")
                current_target_index = 0
            else:
                current_target_index = self.target_index
            st.info(f"Surrogate is multi-output. Using target index {current_target_index} for BO.")
            mu = mu[:, current_target_index]
        elif mu.ndim > 1 and mu.shape[1] == 1:
            mu = mu.ravel()

        if sigma.ndim > 1 and sigma.shape[1] > 1:
            if self.target_index >= sigma.shape[1]:
                # This case implies sigma has per-target uncertainties
                st.error(f"target_index {self.target_index} is out of bounds for surrogate uncertainty with {sigma.shape[1]} outputs. Using index 0.")
                current_target_index_sigma = 0
            else:
                current_target_index_sigma = self.target_index
            sigma = sigma[:, current_target_index_sigma]
        elif sigma.ndim > 1 and sigma.shape[1] == 1:
            sigma = sigma.ravel()

        # Posterior samples also need to be sliced if multi-target
        if posterior_samples is not None and posterior_samples.ndim == 3 and posterior_samples.shape[2] > 1:
            if self.target_index >= posterior_samples.shape[2]:
                current_target_index_samples = 0 # Fallback
            else:
                current_target_index_samples = self.target_index
            posterior_samples = posterior_samples[:, :, current_target_index_samples]
            # Shape becomes (n_posterior_samples, num_input_points)

        return mu, sigma
    
    def acquisition_function(self, X_np: np.ndarray, acquisition: str = "UCB",
                             xi: float = 0.01, kappa: float = 2.0, curiosity: float = 0.0) -> np.ndarray:
        """
        Computes the value of a specified acquisition function for given input points.

        This method uses the surrogate model (internal GP or external) to get mean
        and standard deviation predictions, then calculates the acquisition value.
        The `curiosity` parameter adjusts `kappa` (for UCB) and `xi` (for EI, PI)
        to balance exploration and exploitation.

        Args:
            X_np: Input points (NumPy array, shape (n_points, n_features)) for which
                  to calculate acquisition values.
            acquisition: The type of acquisition function to use.
                         Options: "UCB", "EI", "PI", "MaxEntropy".
            xi: Exploration parameter primarily for Expected Improvement (EI) and
                Probability of Improvement (PI). Higher `xi` encourages more exploration.
            kappa: Exploration parameter for Upper Confidence Bound (UCB).
                   Higher `kappa` encourages more exploration.
            curiosity: A general parameter (-2 to +2) that scales `kappa` and `xi`
                       to tune the exploration-exploitation balance. Positive values
                       increase exploration.

        Returns:
            A NumPy array of acquisition function values, one for each input point.
            Higher values indicate more promising points to evaluate next.
        """
        if not self.is_fitted: # This check is also in _get_surrogate_prediction, but good for early exit.
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Get predictions and uncertainties using the new unified method
        # We don't need posterior samples for these standard acquisition functions.
        mu, sigma, _ = self._get_surrogate_prediction(X_np, return_posterior_samples=False)
        
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
    
    def optimize(self, bounds_dict: dict[str, tuple[float, float]], n_restarts: int = 20,
                 acquisition: str = "UCB", curiosity: float = 0.0,
                 n_points: int = 1, min_distance: float = 0.1, max_iter: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the optimal point(s) by maximizing the specified acquisition function.

        This method uses `scipy.optimize.minimize` (on the negative acquisition
        function) with L-BFGS-B algorithm, starting from multiple random points
        within the given bounds, to find the point(s) that maximize the acquisition value.

        If `n_points > 1`, it attempts to return a diverse set of points by ensuring
        a minimum distance between them, selected greedily from the best points found.

        Args:
            bounds_dict: A dictionary where keys are feature names (strings) and
                         values are tuples `(lower_bound, upper_bound)` for each feature.
            n_restarts: Number of random starting points for the L-BFGS-B optimization.
            acquisition: The acquisition function to optimize (e.g., "UCB", "EI").
            curiosity: The curiosity parameter passed to the acquisition function.
            n_points: The number of optimal points to return. If > 1, diversity is enforced.
            min_distance: Minimum Euclidean distance between returned points if `n_points > 1`.
            max_iter: Maximum number of iterations for each L-BFGS-B optimization run.

        Returns:
            A tuple `(X_optimal, acquisition_values)`:
            - `X_optimal`: NumPy array of shape (n_points, n_features) containing the
                           coordinates of the optimal point(s).
            - `acquisition_values`: NumPy array of shape (n_points,) containing the
                                    acquisition function values at the optimal point(s).
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

    def calculate_qUCB(self, X_batch_np: np.ndarray, n_posterior_samples: int = 50, kappa: float = 2.0) -> float:
        """
        Calculates the q-Upper Confidence Bound (qUCB) for a given batch of candidate points.

        qUCB is a batch acquisition function that estimates the expected maximum value
        among a batch of `q` points, considering the posterior distribution of the
        surrogate model. This implementation uses a common approximation:
        1. Draw `N` samples from the surrogate model's posterior distribution over the `q` batch points.
           Each sample path gives `q` predicted values.
        2. For each of these `N` sample paths, find the maximum predicted value among the `q` points.
        3. The qUCB value is the average of these `N` maximums.

        If the surrogate model is multi-output, calculations are based on the target
        specified by `self.target_index`.

        Args:
            X_batch_np: A NumPy array representing the batch of `q` candidate points.
                        Shape (q, n_features).
            n_posterior_samples: The number of samples to draw from the surrogate's posterior.
            kappa: A parameter that can be used to scale the exploration term if a
                   different qUCB formulation (e.g., involving std dev of max values)
                   was used. Currently, this implementation directly averages max values
                   from posterior samples, so kappa is not directly used in that formula
                   but is kept for interface consistency or future variations.

        Returns:
            The calculated qUCB value (float) for the provided batch of points.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Get posterior samples for the batch
        # _get_surrogate_prediction returns (mean, std, samples)
        # samples shape: (n_posterior_samples, q_points, n_targets)
        _, _, posterior_samples = self._get_surrogate_prediction(
            X_batch_np,
            return_posterior_samples=True,
            n_posterior_samples=n_posterior_samples
        )

        if posterior_samples is None:
            st.error("Could not obtain posterior samples for qUCB calculation. Falling back to standard UCB on batch mean.")
            # Fallback: calculate standard UCB for each point and average, or use mean prediction.
            # This is a rough approximation.
            mu, sigma, _ = self._get_surrogate_prediction(X_batch_np, return_posterior_samples=False)
            ucb_values = mu + kappa * sigma
            return np.mean(ucb_values)


        # If multi-target, qUCB needs to be defined for a specific target or a scalarized objective.
        # Posterior samples from _get_surrogate_prediction are already sliced by self.target_index if they were 3D.
        # So, posterior_samples here should be (n_posterior_samples, q_points)
        if posterior_samples.ndim == 3: # Should not happen if _get_surrogate_prediction sliced correctly
            st.warning("qUCB received 3D posterior samples; expected 2D after target selection. Using target 0.")
            posterior_samples_target = posterior_samples[:, :, 0]
        elif posterior_samples.ndim == 2: # Expected: (n_posterior_samples, q_points)
            posterior_samples_target = posterior_samples
        else:
            raise ValueError(f"Unexpected shape for posterior_samples in qUCB: {posterior_samples.shape}")

        # For each posterior sample path, find the maximum value within the batch
        # max_over_batch_for_each_sample will be shape (n_posterior_samples,)
        max_over_batch_for_each_sample = np.max(posterior_samples_target, axis=1)

        # The qUCB is the mean of these maximums
        qUCB_value = np.mean(max_over_batch_for_each_sample)

        # Optionally, one could use kappa * np.std(max_over_batch_for_each_sample) as an exploration term here,
        # but the standard qUCB (based on fantasizing) is often the expected value of the maximum.
        # The current implementation is more direct: mean of max values of posterior samples.

        return qUCB_value


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


def multi_objective_bayesian_optimization(
    train_inputs: pd.DataFrame | np.ndarray,
    train_targets: np.ndarray,
    candidate_inputs: pd.DataFrame | np.ndarray,
    weights: np.ndarray,
    max_or_min: list[str],
    curiosity: float = 0.0,
    acquisition: str = "UCB",
    strategy: str = "weighted_sum",
    surrogate_model = None, # Type hint would be complex: BaseEstimator or similar with specific methods
    input_columns: list[str] | None = None
) -> np.ndarray | None:
    """
    Performs multi-objective Bayesian optimization (MOBO) to score candidate samples.

    This function scalarizes the multi-objective problem using either a fixed
    'weighted_sum' strategy or 'parego' (randomized scalarization). It then
    calculates acquisition scores for each candidate. It can use a pre-trained
    multi-output surrogate model or fit individual Gaussian Processes (GPs)
    per objective if no surrogate is provided.

    Args:
        train_inputs: Input features of labeled samples (pd.DataFrame or np.ndarray).
        train_targets: Target values of labeled samples (np.ndarray, shape (n_samples, n_objectives)).
        candidate_inputs: Input features of candidate samples to be scored.
        weights: Importance weights for each objective. Used directly in 'weighted_sum'
                 and can bias sampling in 'parego'.
        max_or_min: List of strings ('max' or 'min') indicating optimization
                    direction for each objective.
        curiosity: Exploration vs. exploitation parameter (-2 to +2), adjusts acquisition.
        acquisition: Acquisition function type (e.g., "UCB", "EI", "PI").
        strategy: MOBO strategy:
                  - "weighted_sum": Uses fixed `weights` for scalarization.
                  - "parego": Uses randomly drawn weights (Dirichlet distribution).
        surrogate_model: Optional pre-trained surrogate model. Must implement
                         `predict_with_uncertainty(X, input_columns)` returning
                         (means, std_devs) for all targets. If None, individual
                         GPs are fitted per objective.
        input_columns: Required if `train_inputs` or `candidate_inputs` are NumPy
                       arrays and a `surrogate_model` is provided that expects
                       DataFrame input with column names.

    Returns:
        A NumPy array of acquisition scores for `candidate_inputs`. Higher scores
        are better. Returns None or random scores if issues arise (e.g., no valid data).
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