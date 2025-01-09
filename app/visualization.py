import plotly.graph_objects as go
from sklearn.manifold import TSNE
import plotly.express as px

import pandas as pd

# Scatter plot
def plot_scatter_matrix_with_uncertainty(result_df, target_columns, utility_scores, uncertainty_columns=None):
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

    # Add error bars for uncertainties
    if uncertainty_columns:
        for dim, uncertainty_col in zip(target_columns, uncertainty_columns):
            fig.add_scatter(
                x=scatter_data[dim],
                y=scatter_data["Utility"],
                error_x=scatter_data[uncertainty_col],
                mode="markers",
                marker=dict(color="gray", opacity=0.5),
            )

    return fig




def create_tsne_plot_with_hover(data, feature_cols, utility_col, row_number_col, label_col=None):
    """
    Generates a t-SNE plot with custom hover labels.

    Args:
        data (DataFrame): Data for t-SNE computation.
        feature_cols (list): Columns to use for t-SNE embeddings.
        utility_col (str): Column name for coloring by utility.
        row_number_col (str): Column name for hover labels (e.g., "Idx_Sample").
        label_col (str): Optional column for labeling (e.g., 'Labeled').

    Returns:
        Figure: Plotly scatter plot with t-SNE results.
    """
    # Normalize features for t-SNE
    feature_df = data[feature_cols].copy()
    feature_std = feature_df.std().replace(0, 1)
    feature_mean = feature_df.mean()
    normalized_features = (feature_df - feature_mean) / feature_std

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,  # Adjust if needed
        learning_rate=200,
        n_iter=350,
        random_state=42,
        init="pca",
    )
    tsne_result = tsne.fit_transform(normalized_features)

    # Create t-SNE result DataFrame
    tsne_result_df = pd.DataFrame({
        "Row number": data[row_number_col],
        "t-SNE-1": tsne_result[:, 0],
        "t-SNE-2": tsne_result[:, 1],
        utility_col: data[utility_col],
    })

    if label_col:
        tsne_result_df[label_col] = data[label_col]

    # Generate Plotly Scatter Plot
    fig = px.scatter(
        tsne_result_df,
        x="t-SNE-1",
        y="t-SNE-2",
        color=utility_col,
        symbol=label_col if label_col else None,
        custom_data=["Row number"],
        title="Materials data in t-SNE coordinates",
        render_mode="svg",
    )
    fig.update_traces(
        hovertemplate="Row number: %{customdata}, Utility: %{marker.color:.2f}",
        marker=dict(size=7),
    )
    fig.update_layout(
        height=1000,
        legend_title_text="",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig



def plot_histogram(data, column_name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(data[column_name], bins=20, alpha=0.75, color="blue")
    plt.title(f"Histogram of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    st.pyplot(plt)


def plot_scatter(data, x, y, labels=None):
    fig = px.scatter(data, x=x, y=y, color=labels, title=f"{x} vs {y}")
    st.plotly_chart(fig)



import plotly.express as px

def create_parallel_coordinates(result_df, dimensions, color_column):
    """
    Create a parallel coordinate plot using Plotly.
    
    Args:
        result_df (pd.DataFrame): DataFrame containing the results.
        dimensions (list): List of column names to include as axes.
        color_column (str): Column name for coloring the lines.

    Returns:
        fig: Plotly figure for the parallel coordinate plot.
    """
    fig = px.parallel_coordinates(
        result_df,
        dimensions=dimensions,
        color=color_column,
        title="Parallel Coordinate Plot",
        labels={col: col for col in dimensions + [color_column]},
        template="plotly_white",
    )
    return fig


# Function for 3D scatter plot
import plotly.express as px

def create_3d_scatter(result_df, x_column, y_column, z_column, color_column):
    """
    Create a 3D scatter plot using Plotly with larger dimensions.
    
    Args:
        result_df (pd.DataFrame): DataFrame containing the results.
        x_column (str): Column name for the x-axis.
        y_column (str): Column name for the y-axis.
        z_column (str): Column name for the z-axis.
        color_column (str): Column name for coloring the points.

    Returns:
        fig: Plotly figure for the 3D scatter plot.
    """
    fig = px.scatter_3d(
        result_df,
        x=x_column,
        y=y_column,
        z=z_column,
        color=color_column,
        title=f"3D Scatter Plot ({x_column} vs {y_column} vs {z_column})",
        labels={x_column: x_column, y_column: y_column, z_column: z_column, color_column: color_column},
        template="plotly_white",
    )
    # Increase marker size
    fig.update_traces(marker=dict(size=8))
    # Increase the figure size
    fig.update_layout(
        scene=dict(
            xaxis_title=x_column,
            yaxis_title=y_column,
            zaxis_title=z_column,
        ),
        height=900,  # Increased height
        width=1200,   # Increased width
        margin=dict(l=0, r=0, t=50, b=0)  # Adjusted margins
    )
    return fig



import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_scatter_matrix_with_hover(data, dimensions, utility_col, idx_col):
    matrix_size = len(dimensions)
    fig = make_subplots(
        rows=matrix_size - 1, cols=matrix_size - 1,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.01, vertical_spacing=0.01
    )
    
    # Add scatter plots in lower triangle
    for i in range(1, matrix_size):
        for j in range(i):
            fig.add_trace(
                go.Scatter(
                    x=data[dimensions[j]],
                    y=data[dimensions[i]],
                    mode='markers',
                    marker=dict(
                        size=7,
                        color=data[utility_col],
                        colorscale='Plasma',
                        showscale=(i == 1 and j == 0),  # Show color bar on the first subplot
                        colorbar=dict(title='Utility')
                    ),
                    customdata=data[idx_col],  # Sample indices or other metadata
                    hovertemplate="Idx: %{customdata}<br>%{x:.2f}, %{y:.2f}<br>Utility: %{marker.color:.2f}"
                ),
                row=i, col=j + 1
            )
    
    fig.update_layout(height=800, title="Scatter Matrix with Utility")
    return fig
