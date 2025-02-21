import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots




def plot_scatter_matrix_with_uncertainty(result_df, target_columns):
    """
    Plots a scatter matrix of target properties colored by utility.
    Handles cases where only one target is selected.
    """
    if not target_columns:
        return None  # Avoid error when no target columns are selected

    if len(target_columns) == 1:
        # ✅ Special case: If only one target is selected, create a simple scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result_df[target_columns[0]],
            y=result_df["Utility"],
            mode="markers",
            marker=dict(
                size=7,
                color=result_df["Utility"],
                colorbar=dict(title="Utility"),
                colorscale="Plasma"
            ),
            hovertemplate=f"{target_columns[0]}: %{{x:.2f}}, Utility: %{{y:.2f}}"
        ))

        fig.update_layout(
            title=f"Scatter Plot of {target_columns[0]} vs Utility",
            xaxis_title=target_columns[0],
            yaxis_title="Utility"
        )

        return fig  # ✅ Return single scatter plot

    # ✅ General case: More than one target property, use scatter matrix
    matrix_size = len(target_columns) - 1
    fig = make_subplots(
        rows=matrix_size, cols=matrix_size, start_cell="top-left",
        horizontal_spacing=0.01, vertical_spacing=0.01,
        shared_xaxes=True, shared_yaxes=True
    )
    
    fig.update_layout(title="Scatter Matrix of Target Properties", showlegend=False)

    row_indices, col_indices = np.tril_indices(n=matrix_size, k=0)
    row_indices += 1  # Adjust for subplot indexing
    col_indices += 1

    for (row, col) in zip(row_indices, col_indices):
        x_col = target_columns[col - 1]
        y_col = target_columns[row]

        scatter_plot = go.Scatter(
            x=result_df[x_col],
            y=result_df[y_col],
            mode="markers",
            marker=dict(
                size=7,
                color=result_df["Utility"],
                colorbar=dict(title="Utility"),
                colorscale="Plasma"
            ),
            hovertemplate=f"{x_col}: %{{x:.2f}}, {y_col}: %{{y:.2f}}, Utility: %{{marker.color:.2f}}"
        )

        fig.add_trace(scatter_plot, row=row, col=col)

        if row == matrix_size:
            fig.update_xaxes(title_text=x_col, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=y_col, row=row, col=col)

    fig.update_layout(height=1000)
    return fig  # ✅ Return scatter matrix


def plot_scatter_matrix(result_df, target_columns):
    """
    Plots a scatter matrix of target properties colored by utility.
    """
    fig = px.scatter_matrix(
        result_df,
        dimensions=target_columns,
        color="Utility",
        color_continuous_scale="Viridis",
        title="Scatter Matrix of Target Properties",
    )
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig)

def create_tsne_plot(result_df, feature_cols, utility_col):
    """
    Creates a t-SNE visualization of the data colored by utility.
    """
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=350, random_state=42)
    tsne_result = tsne.fit_transform(result_df[feature_cols])
    tsne_df = pd.DataFrame({"t-SNE-1": tsne_result[:, 0], "t-SNE-2": tsne_result[:, 1], "Utility": result_df[utility_col]})
    fig = px.scatter(tsne_df, x="t-SNE-1", y="t-SNE-2", color="Utility", title="t-SNE Plot")
    st.plotly_chart(fig)

import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE

def create_tsne_plot_with_hover(result_df, feature_cols, utility_col):
    """
    Creates a t-SNE visualization with hover information similar to SLAMD.
    """
    if len(result_df) < 2:
        return None  # Avoid errors if dataset is too small for t-SNE

    tsne = TSNE(n_components=2, perplexity=min(20, len(result_df) - 1),
                n_iter=350, random_state=42, init="pca", learning_rate=100)

    tsne_result = tsne.fit_transform(result_df[feature_cols])

    tsne_df = pd.DataFrame({
        "t-SNE-1": tsne_result[:, 0],
        "t-SNE-2": tsne_result[:, 1],
        "Utility": result_df[utility_col],
        "Idx_Sample": result_df.index
    })

    fig = px.scatter(
        tsne_df,
        x="t-SNE-1",
        y="t-SNE-2",
        color="Utility",
        title="t-SNE Plot with Hover",
        hover_data=["Idx_Sample"],
        color_continuous_scale="Plasma"
    )

    fig.update_traces(
        hovertemplate="Sample: %{customdata}, t-SNE-1: %{x:.2f}, t-SNE-2: %{y:.2f}, Utility: %{marker.color:.2f}"
    )

    # Update layout to make the plot larger
    fig.update_layout(
        width=1000,  # Set width of the plot in pixels
        height=800   # Set height of the plot in pixels
    )

    return fig


def create_3d_scatter(result_df, x_column, y_column, z_column, color_column="Utility"):
    """
    Creates a 3D scatter plot of selected columns.
    """
    fig = px.scatter_3d(
        result_df,
        x=x_column,
        y=y_column,
        z=z_column,
        color=color_column,
        title=f"3D Scatter Plot ({x_column} vs {y_column} vs {z_column})"
    )
    return fig

import plotly.express as px

def create_parallel_coordinates(result_df, target_columns):
    """
    Generates a parallel coordinates plot of selected target properties.
    """
    if not target_columns:
        return None  # Avoid error when no target columns are selected

    # Ensure all target columns exist in the DataFrame
    valid_columns = [col for col in target_columns if col in result_df.columns]

    if not valid_columns:
        return None  # Avoid error if no valid columns remain

    fig = px.parallel_coordinates(
        result_df,
        dimensions=valid_columns + ["Utility"],
        color="Utility",
        title="Parallel Coordinate Plot",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    return fig  # ✅ Ensure this function returns a valid Plotly figure



def visualize_exploration_exploitation(result_df, curiosity):
    # Exploration vs. Exploitation Scatter Plot
    fig = px.scatter(
        result_df,
        x="Novelty",
        y="Utility",
        color="Uncertainty",
        size_max=8,  # Reduce the size of the scatter points
        color_continuous_scale="Viridis",
        hover_data=["Idx_Sample"],
        title=f"Exploration vs. Exploitation (Curiosity = {curiosity})",
        width=1200,  # Increase plot width
        height=700  # Increase plot height
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig)

    # Bar Chart for Selected Candidates
    fig = px.bar(
        result_df, 
        x="Idx_Sample", 
        y="Utility", 
        color="Utility",
        title="Utility Scores of Selected Candidates",
        color_continuous_scale="Blues",
        width=1200,
        height=500
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

    # Distribution of Utility, Novelty, Uncertainty
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Utility Score Distribution", "Novelty Score Distribution", "Uncertainty Score Distribution"))

    fig.add_trace(go.Histogram(x=result_df["Utility"], nbinsx=20, marker_color='salmon', opacity=0.7), row=1, col=1)
    fig.add_trace(go.Histogram(x=result_df["Novelty"], nbinsx=20, marker_color='lightgreen', opacity=0.7), row=1, col=2)
    fig.add_trace(go.Histogram(x=result_df["Uncertainty"], nbinsx=20, marker_color='skyblue', opacity=0.7), row=1, col=3)

    fig.update_layout(
        title_text="Distributions of Utility, Novelty, and Uncertainty",
        showlegend=False,
        width=1200,
        height=400
    )

    st.plotly_chart(fig)
