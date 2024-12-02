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
    for i, mode in enumerate(max_or_min):
        if mode == "min":
            normalized_predictions[:, i] *= -1

    # Apply weights to predictions
    weighted_predictions = normalized_predictions * weights[:, :predictions.shape[1]]

    # Normalize uncertainties
    uncertainty_std = uncertainties.std(axis=0, keepdims=True).clip(min=1e-6)
    normalized_uncertainties = uncertainties / uncertainty_std
    weighted_uncertainties = normalized_uncertainties * weights[:, :uncertainties.shape[1]]

    # Handle apriori constraints
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
                if i < apriori.shape[1] and thresh is not None:  # Ensure index is valid
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



# Novelty calculation
def calculate_novelty(features, labeled_features):
    if labeled_features.shape[0] == 0:
        return np.zeros(features.shape[0])
    distances = distance_matrix(features, labeled_features)
    min_distances = distances.min(axis=1)
    max_distance = min_distances.max()
    novelty = min_distances / (max_distance + 1e-6)
    return novelty
