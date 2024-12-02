import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE  # Import TSNE
import pandas as pd


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
    """
    Create a t-SNE plot for the dataset.

    Args:
        data (pd.DataFrame): The dataset with features and utility.
        features (list): The list of feature column names.
        utility_col (str): Column name representing utility scores.
        perplexity (int): Perplexity parameter for t-SNE.
        learning_rate (int): Learning rate for t-SNE optimization.

    Returns:
        plotly.graph_objects.Figure: A scatter plot in t-SNE space.
    """
    # Validate input data
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

    # Fit t-SNE on the selected feature columns
    tsne_result = tsne.fit_transform(data[features])

    # Create a dataframe with t-SNE results
    tsne_result_df = pd.DataFrame({
        "t-SNE-1": tsne_result[:, 0],
        "t-SNE-2": tsne_result[:, 1],
        utility_col: data[utility_col].values,
    })

    # Generate scatter plot
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
    return fig