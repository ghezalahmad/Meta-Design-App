import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import norm
from app.utils import enforce_diversity


def bayesian_optimization(train_inputs, train_targets, candidate_inputs, curiosity=0.0, n_calls=50, min_distance=1.0):
    """
    Uses Bayesian Optimization to select the most promising sample while enforcing diversity.
    Now includes Expected Improvement (EI) for enhanced exploration.
    """
    train_targets = np.array(train_targets).reshape(-1, 1)
    valid_indices = (~np.isnan(train_targets.flatten())) & (~np.isinf(train_targets.flatten()))
    train_targets = train_targets[valid_indices].reshape(-1, 1)
    train_inputs = train_inputs[valid_indices]

    kernel = C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10), nu=1.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10)

    scaler_x = train_inputs.std(axis=0, keepdims=True) + 1e-6
    scaler_y = train_targets.std(axis=0, keepdims=True) + 1e-6

    train_inputs_scaled = (train_inputs - train_inputs.mean(axis=0)) / scaler_x
    train_targets_scaled = (train_targets - train_targets.mean(axis=0)) / scaler_y

    gpr.fit(train_inputs_scaled, train_targets_scaled)
    candidate_inputs_scaled = (candidate_inputs - train_inputs.mean(axis=0)) / scaler_x

    def objective_function(sample_idx):
        sample_scaled = candidate_inputs_scaled[sample_idx].reshape(1, -1)
        mean, std = gpr.predict(sample_scaled, return_std=True)
        mean_unscaled = mean * scaler_y + train_targets.mean(axis=0)
        std_unscaled = std * scaler_y

        # Expected Improvement (EI) calculation
        best_target = np.max(train_targets)
        z = (mean_unscaled - best_target) / std_unscaled
        expected_improvement = (mean_unscaled - best_target) * norm.cdf(z) + std_unscaled * norm.pdf(z)

        # Exploration vs Exploitation balance
        kappa = np.clip(1.96 + curiosity * std_unscaled * 0.01, -2, 2)
        acquisition_value = -(expected_improvement + kappa * std_unscaled)

        return float(acquisition_value) if np.isfinite(acquisition_value) else 1e6

    search_space = [Integer(0, len(candidate_inputs_scaled) - 1, name="sample_idx")]
    result = gp_minimize(objective_function, search_space, n_calls=min(len(candidate_inputs_scaled), n_calls), random_state=42)
    best_sample_idx = np.argmin(result.func_vals)

    # Enforce diversity before returning the best candidate
    candidate_inputs_diverse = enforce_diversity(candidate_inputs_scaled, train_inputs_scaled, min_distance)
    if len(candidate_inputs_diverse) > 0:
        best_sample_idx = np.random.choice(len(candidate_inputs_diverse))

    return best_sample_idx
