import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd


def calculate_utility(predictions, uncertainties, apriori, curiosity, weights, max_or_min, thresholds=None):
    """
    Calculate the utility scores for a given set of predictions, uncertainties, and optional a priori constraints.

    Args:
        predictions (np.ndarray): Predicted values for target properties (num_samples x num_targets).
        uncertainties (np.ndarray): Uncertainty estimates for predictions (num_samples x num_targets).
        apriori (np.ndarray): A priori property constraints (num_samples x num_constraints), optional.
        curiosity (float): A coefficient balancing exploration (uncertainty) and exploitation (predictions).
        weights (list): Weight factors for each target or constraint.
        max_or_min (list): List of optimization goals ("max" or "min") for each property/constraint.
        thresholds (list): Optional thresholds for constraints (list of floats or None).

    Returns:
        np.ndarray: Utility scores for each sample.
    """
    if predictions.shape != uncertainties.shape:
        raise ValueError("Predictions and uncertainties must have the same shape.")
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    weights = np.array(weights).reshape(1, -1)
    max_or_min = np.array(max_or_min)

    # Normalize predictions
    prediction_std = predictions.std(axis=0, keepdims=True).clip(min=1e-6)
    prediction_mean = predictions.mean(axis=0, keepdims=True)
    normalized_predictions = (predictions - prediction_mean) / prediction_std

    # Adjust for min/max optimization
    for i, mode in enumerate(max_or_min):
        if mode == "min":
            normalized_predictions[:, i] *= -1

    # Apply weights to predictions
    weighted_predictions = normalized_predictions * weights[:, :predictions.shape[1]]

    # Normalize uncertainties
    uncertainty_std = uncertainties.std(axis=0, keepdims=True).clip(min=1e-6)
    normalized_uncertainties = uncertainties / uncertainty_std
    weighted_uncertainties = normalized_uncertainties * weights[:, :uncertainties.shape[1]]

    # Handle a priori constraints
    if apriori is not None and apriori.shape[1] > 0:
        apriori = np.array(apriori)
        apriori_std = apriori.std(axis=0, keepdims=True).clip(min=1e-6)
        apriori_mean = apriori.mean(axis=0, keepdims=True)
        normalized_apriori = (apriori - apriori_mean) / apriori_std

        # Align apriori dimensions with predictions
        if apriori.shape[1] != predictions.shape[1]:
            apriori = np.resize(apriori, (apriori.shape[0], predictions.shape[1]))

        # Apply thresholds
        if thresholds is not None:
            thresholds = np.array(thresholds).reshape(1, -1)
            for i, (thresh, mode) in enumerate(zip(thresholds[0], max_or_min)):
                if i < apriori.shape[1] and thresh is not None:  # Ensure index validity
                    if mode == "min":
                        normalized_apriori[:, i] = np.where(
                            apriori[:, i] > thresh, 0, normalized_apriori[:, i]
                        )
                    elif mode == "max":
                        normalized_apriori[:, i] = np.where(
                            apriori[:, i] < thresh, 0, normalized_apriori[:, i]
                        )

        weighted_apriori = normalized_apriori * weights[:, :apriori.shape[1]]
        apriori_utility = weighted_apriori.sum(axis=1)
    else:
        apriori_utility = np.zeros(predictions.shape[0])

    # Combine all utility components
    utility = (
        weighted_predictions.sum(axis=1)
        + (curiosity * 10) * weighted_uncertainties.sum(axis=1)  # Amplify uncertainty impact
        + apriori_utility
    )
    return utility


from scipy.spatial import distance_matrix
import numpy as np



def calculate_novelty(features, labeled_features):
    """
    Calculate novelty scores based on distances to labeled features.

    Args:
        features (np.ndarray): Feature matrix for unlabeled data (num_samples x num_features).
        labeled_features (np.ndarray): Feature matrix for labeled data (num_labeled_samples x num_features).

    Returns:
        np.ndarray: Novelty scores for each sample.
    """
    if labeled_features.shape[0] == 0:
        # Return zeros if there are no labeled features
        return np.zeros(features.shape[0])

    # Compute the distance matrix between features and labeled_features
    distances = distance_matrix(features, labeled_features)

    # Compute minimum distances for each feature
    min_distances = distances.min(axis=1)

    # Normalize novelty scores by dividing by the maximum distance
    max_distance = min_distances.max()
    novelty_scores = min_distances / (max_distance + 1e-6)

    return novelty_scores


