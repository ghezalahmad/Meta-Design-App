
import numpy as np
import pandas as pd
from lolopy.learners import RandomForestRegressor
import streamlit as st
from app.utils import calculate_utility

class LolopyRFModel:
    """A wrapper class for the lolopy RandomForestRegressor to maintain a consistent interface."""
    def __init__(self, num_trees=100):
        self.model = RandomForestRegressor(num_trees=num_trees)
        self.is_trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_uncertainty(self, X):
        predictions, uncertainties = self.model.predict(X, return_std=True)
        return predictions, uncertainties

def train_lolopy_model(data: pd.DataFrame, input_columns: list, target_columns: list, n_estimators: int = 100):
    """Trains a lolopy RandomForestRegressor model."""
    train_df = data.dropna(subset=target_columns)
    X_train = train_df[input_columns].values
    # Ensure y_train is a 1D array if there's only one target, as expected by lolopy
    y_train = train_df[target_columns].values.squeeze()

    model_wrapper = LolopyRFModel(num_trees=n_estimators)

    with st.spinner("Training Lolopy Random Forest model..."):
        model_wrapper.train(X_train, y_train)

    st.success("Lolopy Random Forest model trained successfully!")
    return model_wrapper, None, None # Returning None for history and loss for consistency

def evaluate_lolopy_model(model, data, input_columns, target_columns, curiosity, weights_targets, max_or_min_targets):
    """Evaluates the lolopy model and returns a scored DataFrame."""
    candidate_df = data[data[target_columns[0]].isnull()].copy()
    X_candidate = candidate_df[input_columns].values

    predictions, uncertainties = model.predict_with_uncertainty(X_candidate)

    # Ensure predictions and uncertainties are 2D
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if uncertainties.ndim == 1:
        uncertainties = uncertainties.reshape(-1, 1)

    # Populate the candidate DataFrame with the results
    for i, col in enumerate(target_columns):
        candidate_df[col] = predictions[:, i]
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    # Extract predictions and uncertainties for the utility calculation
    predictions_for_utility = candidate_df[target_columns].values
    uncertainties_for_utility = candidate_df[[f"Uncertainty ({col})" for col in target_columns]].values

    # Calculate utility and other metrics
    utility_scores = calculate_utility(
        predictions=predictions_for_utility,
        uncertainties=uncertainties_for_utility,
        novelty=None,  # Lolopy model does not produce a novelty score
        curiosity=curiosity,
        weights=weights_targets,
        max_or_min=max_or_min_targets
    )

    candidate_df["Utility"] = utility_scores

    # Add other required columns for consistency with visualization components
    candidate_df["Uncertainty"] = np.mean(uncertainties_for_utility, axis=1)
    candidate_df["Novelty"] = 0  # Lolopy doesn't have a novelty metric
    candidate_df["Exploration"] = candidate_df["Uncertainty"] * (1 + max(0, curiosity))
    candidate_df["Exploitation"] = utility_scores - candidate_df["Exploration"]
    candidate_df["Selected for Testing"] = False
    if not candidate_df.empty:
        candidate_df.loc[candidate_df["Utility"].idxmax(), "Selected for Testing"] = True

    # Sort by utility score to find the best candidates
    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)

    return result_df
