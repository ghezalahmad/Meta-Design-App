import numpy as np
from scipy.spatial import distance_matrix
import random
import torch
import torch.optim as optim
from scipy.stats import norm
import streamlit as st

# Force deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set random seed for reproducibility
def set_seed(seed):
    """
    Set random seed for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Comprehensive utility calculation with multiple acquisition functions
def calculate_utility(predictions, uncertainties, novelty, curiosity, weights, max_or_min, 
                      thresholds=None, acquisition="UCB", for_visualization=False):
    valid_acquisitions = ["UCB", "PI", "MaxEntropy", "EI"]
    if acquisition not in valid_acquisitions:
        acquisition = "UCB"
    
    predictions = np.array(predictions)
    if np.isnan(predictions).any() or np.isinf(predictions).any():
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if uncertainties is None:
        uncertainties = np.ones((predictions.shape[0], 1)) * 0.1
    else:
        uncertainties = np.array(uncertainties)
        if uncertainties.ndim == 1:
            uncertainties = uncertainties.reshape(-1, 1)
        uncertainties = np.nan_to_num(uncertainties, nan=0.1, posinf=1.0, neginf=0.1)
    
    if novelty is None:
        novelty = np.zeros((predictions.shape[0], 1))
    else:
        novelty = np.array(novelty)
        if novelty.ndim == 1:
            novelty = novelty.reshape(-1, 1)

    weights = np.array(weights).reshape(1, -1)

    # Normalize predictions to [0, 1]
    min_vals = np.min(predictions, axis=0, keepdims=True)
    max_vals = np.max(predictions, axis=0, keepdims=True)
    range_vals = np.where((max_vals - min_vals) < 1e-10, 1.0, (max_vals - min_vals))
    norm_predictions = (predictions - min_vals) / range_vals

    for i, direction in enumerate(max_or_min):
        if direction == "min":
            norm_predictions[:, i] = 1.0 - norm_predictions[:, i]

    weighted_predictions = weights * norm_predictions

    # Apply thresholds (optional)
    if thresholds is not None:
        thresholds = np.array([t if t is not None else -np.inf for t in thresholds])
        mask = np.ones_like(predictions, dtype=bool)
        for i, (threshold, direction) in enumerate(zip(thresholds, max_or_min)):
            if not np.isnan(threshold):
                if direction == "max":
                    mask[:, i] = predictions[:, i] >= threshold
                else:
                    mask[:, i] = predictions[:, i] <= threshold
        threshold_factor = np.mean(mask.astype(float), axis=1, keepdims=True)
        weighted_predictions = weighted_predictions * threshold_factor

    norm_uncertainties = uncertainties / (np.max(uncertainties) + 1e-10)
    norm_novelty = novelty / (np.max(novelty) + 1e-10)

    curiosity_factor = np.clip(1.0 + 0.5 * curiosity, 0.1, 2.0)

    if acquisition == "UCB":
        kappa = 0.5 * curiosity_factor
        utility = weighted_predictions.sum(axis=1, keepdims=True) + kappa * norm_uncertainties
        if curiosity > 0:
            utility += 0.2 * curiosity * norm_novelty

    elif acquisition == "EI":
        mean_pred = weighted_predictions.sum(axis=1, keepdims=True)
        best_pred = np.max(mean_pred)
        improvement = np.maximum(0, mean_pred - best_pred)
        sigma = norm_uncertainties + 1e-10
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        utility = ei * (1.0 + 0.2 * curiosity) + 0.1 * curiosity * norm_novelty

    elif acquisition == "PI":
        mean_pred = weighted_predictions.sum(axis=1, keepdims=True)
        best_pred = np.max(mean_pred)
        sigma = norm_uncertainties + 1e-10
        z = (mean_pred - best_pred) / sigma
        pi = norm.cdf(z)
        utility = pi * (1.0 + 0.1 * curiosity) + 0.2 * curiosity * norm_novelty

    elif acquisition == "MaxEntropy":
        utility = norm_uncertainties + 0.5 * norm_novelty * (1.0 + curiosity)
        utility += 0.1 * weighted_predictions.sum(axis=1, keepdims=True)

    utility = np.clip(utility, 1e-10, None)

    # ✅ KEY DIFFERENCE FOR PLOT VS RANKING
    if for_visualization:
        return utility
    else:
        return np.log1p(utility * 100)



# Novelty calculation using distance from known samples
def calculate_novelty(features, labeled_features):
    """
    Calculate novelty scores as the minimum distance to labeled samples.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Features of unlabeled candidates
    labeled_features : numpy.ndarray
        Features of existing labeled samples
        
    Returns:
    --------
    novelty : numpy.ndarray
        Novelty scores for each unlabeled candidate
    """
    # Handle case where no labeled features exist
    if labeled_features is None or labeled_features.shape[0] == 0:
        return np.ones(features.shape[0])
    
    # Check for NaN/Inf values and replace them
    features = np.nan_to_num(features, nan=0.0)
    labeled_features = np.nan_to_num(labeled_features, nan=0.0)
    
    # Calculate distance matrix
    distances = distance_matrix(features, labeled_features)
    
    # Minimum distance to any labeled sample
    min_distances = distances.min(axis=1)
    
    # Normalize by maximum distance to avoid scale issues
    max_distance = min_distances.max()
    if max_distance > 0:
        novelty = min_distances / (max_distance + 1e-6)
    else:
        novelty = np.ones_like(min_distances)
    
    return novelty


# Enhanced uncertainty estimation with Monte Carlo sampling
def calculate_uncertainty(model, inputs, num_perturbations=50, noise_scale=0.1, dropout_rate=0.3):
    """
    Calculate prediction uncertainty using Monte Carlo dropout and input perturbation.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Neural network model
    inputs : torch.Tensor
        Input tensor
    num_perturbations : int
        Number of Monte Carlo samples
    noise_scale : float
        Scale of Gaussian noise for input perturbation
    dropout_rate : float
        Dropout probability
        
    Returns:
    --------
    uncertainty : numpy.ndarray
        Uncertainty scores
    """
    model.train()  # Set model to training mode to enable dropout
    
    # Update dropout rate if needed
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate
    
    # Store predictions from multiple forward passes
    perturbations = []
    
    # Perform MC dropout with input perturbation
    for _ in range(num_perturbations):
        # Add Gaussian noise to inputs
        noise = torch.normal(0, noise_scale, size=inputs.shape)
        perturbed_input = inputs + noise
        
        with torch.no_grad():
            pred = model(perturbed_input)
            perturbations.append(pred)
    
    # Stack predictions and compute statistics
    perturbations = torch.stack(perturbations)
    
    # Calculate variance across MC samples
    variance = torch.var(perturbations, dim=0)
    
    # Calculate mean prediction
    mean_pred = torch.mean(perturbations, dim=0)
    
    # Total uncertainty: variance + small factor proportional to prediction magnitude
    # This combines epistemic (model) and aleatoric (data) uncertainty
    total_uncertainty = variance + torch.abs(mean_pred) * 0.05
    
    # Average uncertainty across output dimensions
    uncertainty = total_uncertainty.mean(dim=1, keepdim=True)
    
    # Apply minimum threshold and return
    return np.clip(uncertainty.numpy(), 1e-6, None)


# Learning Rate Scheduler Configuration
def initialize_scheduler(optimizer, scheduler_type, **kwargs):
   
    if scheduler_type == "CosineAnnealing":
        # Cosine annealing with warm restarts
        T_max = kwargs.get("T_max", 100)
        eta_min = kwargs.get("eta_min", 1e-5)
        T_mult = kwargs.get("T_mult", 2)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_max, T_mult=T_mult, eta_min=eta_min
        )
    
    elif scheduler_type == "ReduceLROnPlateau":
        # Reduce learning rate when a metric stops improving
        factor = kwargs.get("factor", 0.5)
        patience = kwargs.get("patience", 5)
        threshold = kwargs.get("threshold", 1e-4)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, 
            threshold=threshold, verbose=True
        )
    
    elif scheduler_type == "StepLR":
        # Step learning rate by a factor every n epochs
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    
    elif scheduler_type == "Adaptive":
        # Custom adaptive learning rate based on validation loss
        class AdaptiveLR:
            def __init__(self, optimizer, factor=0.5, patience=5, min_lr=1e-6):
                self.optimizer = optimizer
                self.factor = factor
                self.patience = patience
                self.min_lr = min_lr
                self.best_loss = float('inf')
                self.bad_epochs = 0
                
            def step(self, metrics=None):
                # If metrics provided, check if we should reduce LR
                if metrics is not None:
                    if metrics < self.best_loss - 1e-4:
                        self.best_loss = metrics
                        self.bad_epochs = 0
                    else:
                        self.bad_epochs += 1
                        
                    if self.bad_epochs >= self.patience:
                        for param_group in self.optimizer.param_groups:
                            old_lr = param_group['lr']
                            new_lr = max(old_lr * self.factor, self.min_lr)
                            param_group['lr'] = new_lr
                            print(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                        self.bad_epochs = 0
            
            def get_last_lr(self):
                return [group['lr'] for group in self.optimizer.param_groups]
        
        factor = kwargs.get("factor", 0.5)
        patience = kwargs.get("patience", 5)
        min_lr = kwargs.get("min_lr", 1e-6)
        return AdaptiveLR(optimizer, factor=factor, patience=patience, min_lr=min_lr)
    
    # Default to no scheduler
    return None


# Enforce diversity in sample selection
def enforce_diversity(candidate_inputs, selected_inputs, min_distance=0.1):
   
    if len(selected_inputs) == 0:
        return candidate_inputs
    
    # Compute distances between candidates and selected samples
    distances = distance_matrix(candidate_inputs, selected_inputs)
    
    # Identify candidates that are sufficiently distant from all selected samples
    diverse_indices = np.where(np.min(distances, axis=1) > min_distance)[0]
    
    # If no candidates meet the diversity criterion, gradually reduce distance threshold
    if len(diverse_indices) == 0:
        for reduction in [0.75, 0.5, 0.25, 0.1, 0.05]:
            reduced_threshold = min_distance * reduction
            diverse_indices = np.where(np.min(distances, axis=1) > reduced_threshold)[0]
            if len(diverse_indices) > 0:
                st.info(f"Reduced diversity threshold to {reduced_threshold:.4f} to find diverse candidates")
                break
    
    # If still no diverse candidates, return the most distant candidates
    if len(diverse_indices) == 0:
        st.warning("No sufficiently diverse candidates found. Selecting based on maximum distance.")
        max_min_distance_idx = np.argmax(np.min(distances, axis=1))
        return candidate_inputs[np.array([max_min_distance_idx])]
    
    return candidate_inputs[diverse_indices]


# Balance exploration and exploitation
def balance_exploration_exploitation(utility_scores, uncertainty_scores, novelty_scores, curiosity):
    """
    Adjust utility scores based on exploration-exploitation balance.
    
    Parameters:
    -----------
    utility_scores : numpy.ndarray
        Base utility scores
    uncertainty_scores : numpy.ndarray
        Uncertainty scores for each candidate
    novelty_scores : numpy.ndarray
        Novelty scores for each candidate
    curiosity : float
        Exploration vs exploitation parameter (-2 to +2)
        
    Returns:
    --------
    balanced_scores : numpy.ndarray
        Utility scores adjusted for exploration-exploitation balance
    """
    # Convert to numpy arrays and ensure correct shape
    utility_scores = np.array(utility_scores).flatten()
    uncertainty_scores = np.array(uncertainty_scores).flatten()
    novelty_scores = np.array(novelty_scores).flatten()
    
    # Normalize all scores to [0, 1]
    norm_utility = utility_scores / (np.max(utility_scores) + 1e-10)
    norm_uncertainty = uncertainty_scores / (np.max(uncertainty_scores) + 1e-10)
    norm_novelty = novelty_scores / (np.max(novelty_scores) + 1e-10)
    
    # Create exploration score as combination of uncertainty and novelty
    exploration_score = 0.7 * norm_uncertainty + 0.3 * norm_novelty
    
    # Map curiosity from [-2, 2] to [0, 1] for weighting
    # -2: pure exploitation, +2: pure exploration
    exploration_weight = (curiosity + 2) / 4.0
    
    # Calculate balanced score as weighted combination
    balanced_scores = (1 - exploration_weight) * norm_utility + exploration_weight * exploration_score
    
    return balanced_scores


# Generate diverse batch of candidates
def generate_diverse_batch(utility_scores, features, batch_size=5, diversity_weight=0.5):
    """
    Select a diverse batch of candidates with high utility.
    
    Parameters:
    -----------
    utility_scores : numpy.ndarray
        Utility scores for candidates
    features : numpy.ndarray
        Feature vectors for candidates
    batch_size : int
        Number of candidates to select
    diversity_weight : float
        Weight for diversity vs utility (0-1)
        
    Returns:
    --------
    selected_indices : numpy.ndarray
        Indices of selected candidates
    """
    if batch_size >= len(utility_scores):
        return np.arange(len(utility_scores))
    
    # Normalize utility scores
    norm_utility = utility_scores / (np.max(utility_scores) + 1e-10)
    
    selected_indices = []
    remaining_indices = np.arange(len(utility_scores))
    
    # Select first candidate based on highest utility
    first_idx = np.argmax(norm_utility)
    selected_indices.append(first_idx)
    remaining_indices = np.setdiff1d(remaining_indices, selected_indices)
    
    # Select remaining candidates
    for _ in range(batch_size - 1):
        if len(remaining_indices) == 0:
            break
            
        # Calculate distances to already selected candidates
        remaining_features = features[remaining_indices]
        selected_features = features[selected_indices]
        
        distances = distance_matrix(remaining_features, selected_features)
        min_distances = distances.min(axis=1)
        
        # Normalize distances
        norm_distances = min_distances / (np.max(min_distances) + 1e-10)
        
        # Calculate combined score (higher is better)
        combined_scores = (1 - diversity_weight) * norm_utility[remaining_indices] + \
                         diversity_weight * norm_distances
        
        # Select candidate with highest combined score
        next_local_idx = np.argmax(combined_scores)
        next_global_idx = remaining_indices[next_local_idx]
        
        selected_indices.append(next_global_idx)
        remaining_indices = np.setdiff1d(remaining_indices, [next_global_idx])
    
    return np.array(selected_indices)


# Pareto front identification
def identify_pareto_front(predictions, max_or_min):
    """
    Identify the Pareto front of non-dominated solutions.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Predictions for multiple objectives
    max_or_min : list
        Direction of optimization for each objective
        
    Returns:
    --------
    pareto_indices : numpy.ndarray
        Indices of samples on the Pareto front
    """
    n_samples = predictions.shape[0]
    n_objectives = predictions.shape[1]
    
    # Convert all objectives to maximization
    mod_predictions = predictions.copy()
    for i, direction in enumerate(max_or_min):
        if direction == "min":
            mod_predictions[:, i] = -mod_predictions[:, i]
    
    # Initialize domination count array
    is_dominated = np.zeros(n_samples, dtype=bool)
    
    # Compare each sample with all others
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                # Check if j dominates i
                dominates = True
                
                for k in range(n_objectives):
                    # j doesn't dominate i if any objective is worse
                    if mod_predictions[j, k] < mod_predictions[i, k]:
                        dominates = False
                        break
                
                # If all objectives are at least as good and at least one is better
                if dominates and np.any(mod_predictions[j, :] > mod_predictions[i, :]):
                    is_dominated[i] = True
                    break
    
    # Return indices of non-dominated solutions
    return np.where(~is_dominated)[0]

def select_acquisition_function(curiosity, num_labeled_samples):
    """
    Dynamically select acquisition function based on SLAMD-style strategy.

    Parameters:
    -----------
    curiosity : float
        User-defined exploration-exploitation balance (-2 to +2).
    num_labeled_samples : int
        Number of labeled samples available.

    Returns:
    --------
    acquisition : str
        Selected acquisition function name.
    """
    if num_labeled_samples < 10:
        if curiosity > 0.5:
            return "MaxEntropy"  # Strong exploration
        elif curiosity < -0.5:
            return "EI"          # Strong exploitation
        else:
            return "UCB"         # Balanced
    else:
        return "PI" if curiosity > 1.0 else "UCB"


import numpy as np
from scipy.stats import norm

def compute_acquisition_utility(predictions, uncertainties, novelty, curiosity, weights, max_or_min, acquisition):
    """
    Compute utility scores based on selected acquisition function.
    """
    # Ensure proper array shapes
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties).reshape(-1)
    novelty = np.array(novelty).reshape(-1)
    weights = np.array(weights).reshape(-1)

    # Normalize predictions [0,1] with max/min direction
    min_vals = predictions.min(axis=0, keepdims=True)
    max_vals = predictions.max(axis=0, keepdims=True)
    norm_preds = (predictions - min_vals) / (max_vals - min_vals + 1e-10)

    for i, direction in enumerate(max_or_min):
        if direction == "min":
            norm_preds[:, i] = 1.0 - norm_preds[:, i]

    # Weighted predictions: shape [n_samples]
    weighted_preds = norm_preds @ weights  # no reshape needed
    weighted_preds = weighted_preds.flatten()

    # Normalize uncertainties/novelty
    norm_uncertainty = uncertainties / (uncertainties.max() + 1e-10)
    norm_novelty = novelty / (novelty.max() + 1e-10)

    curiosity_factor = np.clip(1.0 + 0.5 * curiosity, 0.1, 2.0)

    if acquisition == "UCB":
        utility = weighted_preds + curiosity_factor * norm_uncertainty
    elif acquisition == "EI":
        best = weighted_preds.max()
        improvement = np.maximum(0, weighted_preds - best)
        z = improvement / (norm_uncertainty + 1e-10)
        ei = improvement * norm.cdf(z) + norm_uncertainty * norm.pdf(z) * curiosity_factor
        utility = ei
    elif acquisition == "PI":
        best = weighted_preds.max()
        z = (weighted_preds - best) / (norm_uncertainty + 1e-10)
        utility = norm.cdf(z)
    elif acquisition == "MaxEntropy":
        utility = norm_uncertainty + 0.5 * norm_novelty * curiosity_factor
    else:
        utility = weighted_preds

    return np.log1p(np.clip(utility, 1e-8, None))  # final shape [n_samples]

