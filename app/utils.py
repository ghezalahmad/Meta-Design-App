import numpy as np
from scipy.spatial import distance_matrix
import random
import torch
import torch.optim as optim

# Force deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Utility Function with Advanced Acquisition Strategies
def calculate_utility(predictions, uncertainties, apriori, curiosity, weights, max_or_min, thresholds=None, acquisition="UCB"):
    """
    Compute utility scores using advanced acquisition functions such as UCB and PI.

    Args:
        predictions (array): Model predictions.
        uncertainties (array): Uncertainty estimates.
        apriori (array, optional): Prior knowledge (not used currently but reserved for extensions).
        curiosity (float): Exploration factor (-2 for exploitation, +2 for exploration).
        weights (array): Importance of each target variable.
        max_or_min (list): Optimization direction ("max" or "min").
        thresholds (array, optional): Minimum/maximum constraints for selection.
        acquisition (str, optional): Acquisition function ("UCB" or "PI").

    Returns:
        utility (array): Computed utility scores.
    """
    # Add these debug print statements at the start of the function:
    print("Predictions - Min:", np.min(predictions), "Max:", np.max(predictions), "Mean:", np.mean(predictions))
    print("Uncertainty - Min:", np.min(uncertainties), "Max:", np.max(uncertainties), "Mean:", np.mean(uncertainties))

    # Add this line after loading and shaping the predictions array:
    predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions) + 1e-6)

    # The code should look like this:
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    weights = np.array(weights).reshape(1, -1)

    # ✅ Normalize predictions to the [0, 1] range
    predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions) + 1e-6)

    mean_pred = predictions.mean(axis=0, keepdims=True)
    std_pred = predictions.std(axis=0, keepdims=True).clip(min=1e-6)
    normalized_preds = (predictions - mean_pred) / std_pred

    # Compute Expected Improvement (EI)
    expected_improvement = np.maximum(1e-6, normalized_preds)  # Ensure non-zero utility
    for i, direction in enumerate(max_or_min):
        if direction == "min":
            expected_improvement[:, i] = -expected_improvement[:, i]

    # Threshold Handling
    if thresholds is not None:
        thresholds = np.array(thresholds).reshape(1, -1)
        mask = (predictions >= thresholds) if max_or_min[0] == "max" else (predictions <= thresholds)
        expected_improvement *= mask  

    # Curiosity Factor
    curiosity_factor = np.exp(2.0 * curiosity)  # Stronger curiosity influence

    if acquisition == "UCB":
        kappa = 1.96 + curiosity * 0.5  # More dynamic exploration
        utility = (weights * predictions).sum(axis=1) + kappa * uncertainties.flatten()
    elif acquisition == "PI":
        prob_improvement = np.maximum(predictions - predictions.max(), 0)
        utility = (weights * prob_improvement).sum(axis=1) + curiosity_factor * uncertainties.flatten()
    elif acquisition == "EI":
        expected_improvement = np.maximum(predictions - predictions.max(), 0)
        utility = (weights * expected_improvement).sum(axis=1) + curiosity_factor * uncertainties.flatten()
    else:
        utility = (weights * predictions).sum(axis=1)
    
    print(f"✅ Acquisition Function in Use: {acquisition}")

    utility = utility * 1000  # Scale utility to increase differentiation

    # The code should look like this:
    utility = np.clip(utility, 1e-6, None)  # Avoid true zeros by setting a minimum
    utility = utility * 1000  # Scale utility for better spread
    return np.log1p(utility)  # Apply log scaling to smooth large values



# Novelty calculation
def calculate_novelty(features, labeled_features):
    if labeled_features.shape[0] == 0:
        return np.zeros(features.shape[0])
    distances = distance_matrix(features, labeled_features)
    min_distances = distances.min(axis=1)
    max_distance = min_distances.max()
    novelty = min_distances / (max_distance + 1e-6)
    return novelty

# Monte Carlo (MC) Dropout for Uncertainty Estimation
def calculate_uncertainty(model, inputs, num_perturbations=500, noise_scale=0.1, dropout_rate=0.5):
    perturbations = []
    model.train()  # ✅ Keep dropout active
    for _ in range(num_perturbations):
        noise = torch.normal(0, noise_scale, size=inputs.shape)  # Add input noise
        perturbed_input = inputs + noise
        perturbed_prediction = model(perturbed_input).detach().numpy()
        perturbations.append(perturbed_prediction)
    
    # ✅ Convert the list to a NumPy array before calculating std
    perturbations = np.array(perturbations)

    # Calculate the standard deviation as a measure of uncertainty
    uncertainty = perturbations.std(axis=0).mean(axis=1, keepdims=True)
    
    # ✅ Apply a minimum threshold to avoid zero uncertainty
    uncertainty = np.clip(uncertainty, 1e-6, None)
    
    print("Uncertainty - Min:", np.min(uncertainty), "Max:", np.max(uncertainty), "Mean:", np.mean(uncertainty))
    
    return uncertainty




# Learning Rate Scheduler Initialization
def initialize_scheduler(optimizer, scheduler_type, **kwargs):
    if scheduler_type == "CosineAnnealing":
        T_max = kwargs.get("T_max", 100)
        eta_min = kwargs.get("eta_min", 1e-5)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler_type == "ReduceLROnPlateau":
        factor = kwargs.get("factor", 0.5)
        patience = kwargs.get("patience", 5)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=True)

    return None

# Enforcing Diversity in Sample Selection
def enforce_diversity(candidate_inputs, selected_inputs, min_distance=1.0):
    diverse_candidates = []
    for candidate in candidate_inputs:
        distances = [np.linalg.norm(candidate - existing) for existing in selected_inputs]
        if all(d > min_distance for d in distances):
            diverse_candidates.append(candidate)

    return np.array(diverse_candidates) if diverse_candidates else candidate_inputs
