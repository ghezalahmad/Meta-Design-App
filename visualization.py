
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
from sklearn.manifold import TSNE


# Scatter plot
def plot_scatter_matrix(result_df, target_columns, utility_scores):
    scatter_data = result_df[target_columns + ["Utility"]].copy()
    scatter_data["Utility"] = utility_scores

    fig = px.scatter_matrix(
        scatter_data,
        dimensions=target_columns,
        color="Utility",
        color_continuous_scale="Viridis",
        title="Scatter Matrix of Target Properties",
        labels={col: col for col in target_columns},
    )
    fig.update_traces(diagonal_visible=False)
    return fig



def create_tsne_plot(data, features, utility_col="Utility", perplexity=20, learning_rate=200):
    #st.write("Debug: Entered create_tsne_plot")
    #st.write(f"Features passed: {features}")
    #st.write(f"Data shape: {data.shape}")

    if len(features) == 0:
        raise ValueError("No features selected for t-SNE.")

    if utility_col not in data.columns:
        raise ValueError(f"The column '{utility_col}' is not in the dataset.")

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(data) - 1),
        n_iter=350,
        random_state=42,
        init="pca",
        learning_rate=learning_rate,
    )

    tsne_result = tsne.fit_transform(data[features])

    st.write("Debug: t-SNE computation complete")
    tsne_result_df = pd.DataFrame({
        "t-SNE-1": tsne_result[:, 0],
        "t-SNE-2": tsne_result[:, 1],
        utility_col: data[utility_col].values,
    })

    fig = px.scatter(
        tsne_result_df,
        x="t-SNE-1",
        y="t-SNE-2",
        color=utility_col,
        title="t-SNE Visualization of Data",
        labels={"t-SNE-1": "t-SNE Dimension 1", "t-SNE-2": "t-SNE Dimension 2"},
        color_continuous_scale="Viridis",
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(height=800, legend_title_text="Utility")

    st.write("Debug: Returning t-SNE plot")
    return fig
