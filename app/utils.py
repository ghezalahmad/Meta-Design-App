import numpy as np
from scipy.spatial import distance_matrix
import random
import numpy as np
import torch


# Force deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)  # Choose a fixed seed for reproducibility



# Utility function
def calculate_utility(predictions, uncertainties, apriori, curiosity, weights, max_or_min, thresholds=None):
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    weights = np.array(weights).reshape(1, -1)
    max_or_min = np.array(max_or_min)

    # Normalize predictions
    prediction_std = predictions.std(axis=0, keepdims=True).clip(min=1e-6)
    prediction_mean = predictions.mean(axis=0, keepdims=True)
    normalized_predictions = (predictions - prediction_mean) / prediction_std

    # Adjust for min/max optimization
    for i, mode in enumerate(max_or_min[:predictions.shape[1]]):  # Match predictions dimensions
        if mode == "min":
            normalized_predictions[:, i] *= -1

    # Apply weights to predictions
    weighted_predictions = normalized_predictions * weights[:, :predictions.shape[1]]

    # Normalize uncertainties
    uncertainty_std = uncertainties.std(axis=0, keepdims=True).clip(min=1e-6)
    normalized_uncertainties = uncertainties / uncertainty_std
    weighted_uncertainties = normalized_uncertainties * weights[:, :uncertainties.shape[1]]

    apriori_utility = np.zeros(predictions.shape[0])  # Default if no apriori
    if apriori is not None and apriori.shape[1] > 0:
        apriori = np.array(apriori)
        apriori_std = apriori.std(axis=0, keepdims=True).clip(min=1e-6)
        apriori_mean = apriori.mean(axis=0, keepdims=True)
        normalized_apriori = (apriori - apriori_mean) / apriori_std

        # Align thresholds with apriori dimensions
        if thresholds is not None:
            thresholds = np.array(thresholds[:apriori.shape[1]]).reshape(1, -1)

        for i in range(apriori.shape[1]):  # Iterate over apriori columns
            if thresholds is not None and thresholds[0, i] is not None:
                if max_or_min[predictions.shape[1] + i] == "min":
                    normalized_apriori[:, i] = np.where(
                        apriori[:, i] > thresholds[0, i], 0, normalized_apriori[:, i]
                    )
                elif max_or_min[predictions.shape[1] + i] == "max":
                    normalized_apriori[:, i] = np.where(
                        apriori[:, i] < thresholds[0, i], 0, normalized_apriori[:, i]
                    )

        weighted_apriori = normalized_apriori * weights[:, predictions.shape[1]:]
        apriori_utility = weighted_apriori.sum(axis=1)

    # Combine all utility components
    utility = (
        weighted_predictions.sum(axis=1)
        + (curiosity * 10) * weighted_uncertainties.sum(axis=1)  # Amplify uncertainty impact
        + apriori_utility
    )
    return utility




# Novelty calculation
def calculate_novelty(features, labeled_features):
    if labeled_features.shape[0] == 0:
        return np.zeros(features.shape[0])
    distances = distance_matrix(features, labeled_features)
    min_distances = distances.min(axis=1)
    max_distance = min_distances.max()
    novelty = min_distances / (max_distance + 1e-6)
    return novelty



def calculate_uncertainty(model, inputs, num_perturbations=50, noise_scale=0.5):
    """
    Calculate uncertainty based on input perturbations.

    Args:
        model: Trained model to predict outputs.
        inputs: Input tensor.
        num_perturbations: Number of perturbations to apply.
        noise_scale: Standard deviation of noise added to inputs.

    Returns:
        Uncertainty scores as the standard deviation of predictions.
    """
    perturbations = []
    torch.manual_seed(42)  # Ensure fixed noise generation
    for _ in range(num_perturbations):
        noise = torch.normal(0, noise_scale, size=inputs.shape)  # Fixed noise
        perturbed_input = inputs + noise
        perturbed_prediction = model(perturbed_input).detach().numpy()
        perturbations.append(perturbed_prediction)
    perturbations = np.stack(perturbations, axis=0)
    return perturbations.std(axis=0).mean(axis=1, keepdims=True)