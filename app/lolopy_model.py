
import numpy as np
import pandas as pd
from lolopy.learners import RandomForestRegressor
import streamlit as st
from app.utils import calculate_utility

class LolopyRFModel:
    """
    A wrapper class for lolopy's RandomForestRegressor that handles both
    single-target and multi-target regression by training a separate model for each target.
    """
    def __init__(self, num_trees=100):
        self.num_trees = num_trees
        self.models = []
        self.is_trained = False

    def train(self, X, y):
        """Trains the model. If y is 2D, trains one model per column."""
        self.models = []
        if y.ndim == 1:
            # Single-target case
            model = RandomForestRegressor(num_trees=self.num_trees)
            model.fit(X, y)
            self.models.append(model)
        else:
            # Multi-target case: train one model per target column
            num_targets = y.shape[1]
            for i in range(num_targets):
                model = RandomForestRegressor(num_trees=self.num_trees)
                model.fit(X, y[:, i])
                self.models.append(model)
        self.is_trained = True

    def predict_with_uncertainty(self, X):
        """Generates predictions and uncertainties from all trained models."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        all_predictions = []
        all_uncertainties = []

        for model in self.models:
            predictions, uncertainties = model.predict(X, return_std=True)
            all_predictions.append(predictions.reshape(-1, 1))
            all_uncertainties.append(uncertainties.reshape(-1, 1))

        # Concatenate results into 2D arrays
        final_predictions = np.hstack(all_predictions)
        final_uncertainties = np.hstack(all_uncertainties)

        return final_predictions, final_uncertainties

def train_lolopy_model(data: pd.DataFrame, input_columns: list, target_columns: list, n_estimators: int = 100):
    """Trains a lolopy RandomForestRegressor model."""
    train_df = data.dropna(subset=target_columns)
    X_train = train_df[input_columns].values
    y_train = train_df[target_columns].values

    model_wrapper = LolopyRFModel(num_trees=n_estimators)

    with st.spinner("Training Lolopy Random Forest model..."):
        model_wrapper.train(X_train, y_train)

    st.success("Lolopy Random Forest model trained successfully!")
    return model_wrapper, None, None

def evaluate_lolopy_model(model, data, input_columns, target_columns, curiosity, weights_targets, max_or_min_targets):
    """Evaluates the lolopy model and returns a scored DataFrame."""
    candidate_df = data[data[target_columns[0]].isnull()].copy()
    X_candidate = candidate_df[input_columns].values

    predictions, uncertainties = model.predict_with_uncertainty(X_candidate)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if uncertainties.ndim == 1:
        uncertainties = uncertainties.reshape(-1, 1)

    for i, col in enumerate(target_columns):
        candidate_df[col] = predictions[:, i]
        candidate_df[f"Uncertainty ({col})"] = uncertainties[:, i]

    predictions_for_utility = candidate_df[target_columns].values
    uncertainties_for_utility = candidate_df[[f"Uncertainty ({col})" for col in target_columns]].values

    utility_scores = calculate_utility(
        predictions=predictions_for_utility,
        uncertainties=uncertainties_for_utility,
        novelty=None,
        curiosity=curiosity,
        weights=weights_targets,
        max_or_min=max_or_min_targets
    )

    candidate_df["Utility"] = utility_scores
    candidate_df["Uncertainty"] = np.mean(uncertainties_for_utility, axis=1)
    candidate_df["Novelty"] = 0
    candidate_df["Exploration"] = candidate_df["Uncertainty"] * (1 + max(0, curiosity))
    candidate_df["Exploitation"] = utility_scores - candidate_df["Exploration"]
    candidate_df["Selected for Testing"] = False
    if not candidate_df.empty:
        candidate_df.loc[candidate_df["Utility"].idxmax(), "Selected for Testing"] = True

    result_df = candidate_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)

    return result_df
