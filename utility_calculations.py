import os
import pandas as pd
import numpy as np
import torch
import streamlit as st
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
import plotly.express as px
import torch.optim as optim  # Import PyTorch's optimizer module
from skopt import gp_minimize
from skopt.space import Real
import json
import plotly.graph_objects as go


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
    for i, mode in enumerate(max_or_min[:predictions.shape[1]]):
        if mode == "min":
            normalized_predictions[:, i] *= -1

    # Apply weights to predictions
    weighted_predictions = normalized_predictions * weights[:, :predictions.shape[1]]

    # Normalize uncertainties
    uncertainty_std = uncertainties.std(axis=0, keepdims=True).clip(min=1e-6)
    normalized_uncertainties = uncertainties / uncertainty_std
    weighted_uncertainties = normalized_uncertainties * weights[:, :uncertainties.shape[1]]

    # Handle apriori constraints
    apriori_utility = np.zeros(predictions.shape[0])  # Default to zeros
    if apriori is not None and apriori.shape[1] > 0:
        apriori = np.array(apriori)
        apriori_std = apriori.std(axis=0, keepdims=True).clip(min=1e-6)
        apriori_mean = apriori.mean(axis=0, keepdims=True)
        normalized_apriori = (apriori - apriori_mean) / apriori_std

        # Adjust weights to exclude targets
        if apriori.shape[1] != len(weights[:, predictions.shape[1]:].flatten()):
            raise ValueError(
                f"A priori data columns ({apriori.shape[1]}) do not match the expected number of weights for a priori data: "
                f"{len(weights[:, predictions.shape[1]:].flatten())}."
            )
        apriori_weights = weights[:, predictions.shape[1]:]



        # Apply thresholds
        if thresholds is not None:
            thresholds = np.array(thresholds).reshape(1, -1)
            for i in range(min(len(thresholds[0]), apriori.shape[1])):
                thresh = thresholds[0][i]
                mode = max_or_min[predictions.shape[1] + i]  # Offset for a priori indices
                if thresh is not None:
                    if mode == "min":
                        normalized_apriori[:, i] = np.where(
                            apriori[:, i] > thresh, 0, normalized_apriori[:, i]
                        )
                    elif mode == "max":
                        normalized_apriori[:, i] = np.where(
                            apriori[:, i] < thresh, 0, normalized_apriori[:, i]
                        )

        weighted_apriori = normalized_apriori * apriori_weights
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

def calculate_uncertainty(meta_model, inputs_tensor, num_perturbations=20):
    noise_scale = 0.1
    perturbed_predictions = []
    for _ in range(num_perturbations):
        perturbed_input = inputs_tensor + torch.normal(0, noise_scale, size=inputs_tensor.shape)
        perturbed_prediction = meta_model(perturbed_input).detach().numpy()
        perturbed_predictions.append(perturbed_prediction)
    perturbed_predictions = np.stack(perturbed_predictions, axis=0)
    return perturbed_predictions.std(axis=0).mean(axis=1, keepdims=True)

def prepare_results_dataframe(predictions, inputs_infer, novelty_scores, uncertainty_scores, target_columns):
    result_df = pd.DataFrame({
        **{col: predictions[:, i] for i, col in enumerate(target_columns)},
        **inputs_infer.reset_index(drop=True).to_dict(orient="list"),
        "Novelty": novelty_scores,
        "Uncertainty": uncertainty_scores.flatten(),
    }).sort_values(by="Novelty", ascending=False).reset_index(drop=True)
    return result_df
