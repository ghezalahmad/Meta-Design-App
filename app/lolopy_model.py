
import numpy as np
import pandas as pd
from lolopy.learners import RandomForestRegressor
import streamlit as st
from app.utils import calculate_utility

class LolopyRFModel:
    """A wrapper class for the lolopy RandomForestRegressor to maintain a consistent interface."""
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
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
    y_train = train_df[target_columns].values

    model_wrapper = LolopyRFModel(n_estimators=n_estimators)

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

    # Calculate utility and other metrics
    result_df = calculate_utility(
        candidate_df,
        target_columns,
        weights_targets,
        max_or_min_targets,
        curiosity
    )
    return result_df
