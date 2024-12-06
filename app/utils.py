import numpy as np
from scipy.spatial import distance_matrix


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


import numpy as np
import torch

def calculate_uncertainty(model, inputs, num_perturbations=20, noise_scale=0.1):
    """
    Calculate uncertainty by adding perturbations to inputs.

    Args:
        model (torch.nn.Module): The trained model.
        inputs (torch.Tensor): The input data as a PyTorch tensor.
        num_perturbations (int): Number of perturbations to calculate uncertainty.
        noise_scale (float): Scale of noise to add for perturbations.

    Returns:
        np.ndarray: Uncertainty scores for each input sample.
    """
    perturbed_predictions = []

    for _ in range(num_perturbations):
        # Add Gaussian noise to inputs
        perturbed_inputs = inputs + torch.normal(0, noise_scale, size=inputs.shape)
        with torch.no_grad():
            predictions = model(perturbed_inputs)
        perturbed_predictions.append(predictions.numpy())

    # Stack predictions and calculate standard deviation
    perturbed_predictions = np.stack(perturbed_predictions, axis=0)
    uncertainty_scores = np.std(perturbed_predictions, axis=0).mean(axis=1, keepdims=True)

    return uncertainty_scores
