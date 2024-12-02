import os
import json
import numpy as np
import pandas as pd
import torch
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.spatial import distance_matrix
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances


# Import project-specific modules
from models.maml_model import MAMLModel, meta_train
from utils.utility_calculations import calculate_utility
from utils.visualization import plot_scatter_matrix, create_tsne_plot
from utils.session_management import restore_session, save_session
from configs.settings import UPLOAD_FOLDER, RESULT_FOLDER, DEFAULT_HIDDEN_SIZE, DEFAULT_LEARNING_RATE
from utils.utility_calculations import calculate_novelty

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Streamlit configuration
st.set_page_config(page_title="MAML Dashboard", layout="wide")
st.title("MAML Dashboard")

# Sidebar Configuration
with st.sidebar.expander("Model Configuration", expanded=True):
    hidden_size = st.slider("Hidden Size (MAML):", 64, 256, step=16, value=DEFAULT_HIDDEN_SIZE)
    learning_rate = st.slider("Learning Rate:", 0.001, 0.1, step=0.001, value=DEFAULT_LEARNING_RATE)
    curiosity = st.slider("Curiosity (Explore vs Exploit):", -2.0, 2.0, step=0.1, value=0.0)

with st.sidebar.expander("Meta-Training Configuration", expanded=False):
    meta_epochs = st.slider("Meta-Training Epochs:", 10, 100, step=10, value=50)
    inner_lr = st.slider("Inner Loop Learning Rate:", 0.001, 0.1, step=0.001, value=0.01)
    outer_lr = st.slider("Outer Loop Learning Rate:", 0.001, 0.1, step=0.001, value=DEFAULT_LEARNING_RATE)
    num_tasks = st.slider("Number of Tasks:", 2, 10, step=1, value=5)

# Restore Session
st.sidebar.header("Restore Session")
uploaded_session = st.sidebar.file_uploader("Upload Session File (JSON):", type=["json"])

input_columns, target_columns, apriori_columns = [], [], []
weights_targets, weights_apriori = [], []
thresholds_targets, thresholds_apriori = [], []

if uploaded_session:
    try:
        session_data = json.load(uploaded_session)
        restored_session = restore_session(session_data)
        input_columns = restored_session["input_columns"]
        target_columns = restored_session["target_columns"]
        apriori_columns = restored_session["apriori_columns"]
        weights_targets = restored_session["weights_targets"]
        weights_apriori = restored_session["weights_apriori"]
        thresholds_targets = restored_session["thresholds_targets"]
        thresholds_apriori = restored_session["thresholds_apriori"]
        curiosity = restored_session["curiosity"]
        st.sidebar.success("Session restored successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to restore session: {str(e)}")

# File Upload
uploaded_file = st.file_uploader("Upload Dataset (CSV format):", type=["csv"])
if uploaded_file:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    data = pd.read_csv(file_path)
    st.success("Dataset uploaded successfully!")
    st.dataframe(data)

    # Feature Selection
    st.header("Select Features")
    input_columns = st.multiselect("Input Features:", options=data.columns.tolist(), default=input_columns)
    target_columns = st.multiselect("Target Properties:", options=[col for col in data.columns if col not in input_columns], default=target_columns)
    apriori_columns = st.multiselect("A Priori Properties:", options=[col for col in data.columns if col not in input_columns + target_columns], default=apriori_columns)

    # Target Configuration
    st.header("Target Settings")
    max_or_min_targets, weights_targets, thresholds_targets = [], [], []
    for col in target_columns:
        with st.expander(f"Target: {col}"):
            optimize_for = st.radio(f"Optimize {col} for:", ["Maximize", "Minimize"], index=0)
            weight = st.number_input(f"Weight for {col}:", value=1.0, step=0.1)
            threshold = st.text_input(f"Threshold (optional) for {col}:", value="")
            max_or_min_targets.append("max" if optimize_for == "Maximize" else "min")
            weights_targets.append(weight)
            thresholds_targets.append(float(threshold) if threshold else None)

    # Apriori Configuration
    st.header("Apriori Settings")
    max_or_min_apriori, weights_apriori, thresholds_apriori = [], [], []
    for col in apriori_columns:
        with st.expander(f"A Priori: {col}"):
            optimize_for = st.radio(f"Optimize {col} for:", ["Maximize", "Minimize"], index=0)
            weight = st.number_input(f"Weight for {col}:", value=1.0, step=0.1)
            threshold = st.text_input(f"Threshold (optional) for {col}:", value="")
            max_or_min_apriori.append("max" if optimize_for == "Maximize" else "min")
            weights_apriori.append(weight)
            thresholds_apriori.append(float(threshold) if threshold else None)

    # Run Experiment
    # Run Experiment
    if st.button("Run Experiment"):
        if not input_columns or not target_columns:
            st.error("Please select at least one input feature and one target property.")
        else:
            try:
                # Data Preprocessing
                known_targets = ~data[target_columns[0]].isna()
                inputs_train = data.loc[known_targets, input_columns]
                targets_train = data.loc[known_targets, target_columns]
                inputs_infer = data.loc[~known_targets, input_columns]

                # Index samples for inference
                if "Idx_Sample" in data.columns:
                    idx_samples = data.loc[~known_targets, "Idx_Sample"].reset_index(drop=True)
                else:
                    idx_samples = pd.Series(range(1, len(inputs_infer) + 1), name="Idx_Sample")

                # Scaling
                scaler_inputs = StandardScaler()
                scaler_targets = StandardScaler()

                inputs_train_scaled = scaler_inputs.fit_transform(inputs_train)
                targets_train_scaled = scaler_targets.fit_transform(targets_train)
                inputs_infer_scaled = scaler_inputs.transform(inputs_infer)

                # Initialize the MAML model
                meta_model = MAMLModel(len(input_columns), len(target_columns), hidden_size=hidden_size)

                # Meta-training
                st.write("### Meta-Training Phase")
                meta_model = meta_train(
                    meta_model=meta_model,
                    data=data,
                    input_columns=input_columns,
                    target_columns=target_columns,
                    epochs=meta_epochs,
                    inner_lr=inner_lr,
                    outer_lr=learning_rate,
                    num_tasks=num_tasks,
                )
                st.write("Meta-training completed!")

                # Inference
                inputs_infer_tensor = torch.tensor(inputs_infer_scaled, dtype=torch.float32)
                meta_model.eval()
                with torch.no_grad():
                    predictions_scaled = meta_model(inputs_infer_tensor).numpy()
                predictions = scaler_targets.inverse_transform(predictions_scaled)

                # Ensure predictions are always 2D
                if predictions_scaled.ndim == 1:
                    predictions_scaled = predictions_scaled.reshape(-1, 1)

                # Uncertainty Calculation
                noise_scale = 0.1
                num_perturbations = 20
                perturbed_predictions = [
                    meta_model(inputs_infer_tensor + torch.normal(0, noise_scale, size=inputs_infer_tensor.shape)).detach().numpy()
                    for _ in range(num_perturbations)
                ]
                perturbed_predictions = np.stack(perturbed_predictions, axis=0)
                uncertainty_scores = perturbed_predictions.std(axis=0)  # Shape: (num_samples, target_dim)

                # Ensure uncertainty is 2D
                if uncertainty_scores.ndim == 1:
                    uncertainty_scores = uncertainty_scores.reshape(-1, 1)

                # Novelty Calculation
                #novelty_scores = calculate_novelty(inputs_infer_scaled, inputs_train_scaled).reshape(-1, 1)
                novelty_scores = calculate_novelty(inputs_infer_scaled, inputs_train_scaled)

                # Utility Calculation
                apriori_infer_scaled = np.zeros((inputs_infer.shape[0], 1))  # Handle absence of a priori
                utility_scores = calculate_utility(
                    predictions,
                    uncertainty_scores,
                    apriori_infer_scaled,
                    curiosity=curiosity,
                    weights=weights_targets,
                    max_or_min=max_or_min_targets,
                )

                # Result Compilation
                result_df = pd.DataFrame({
                    "Idx_Sample": idx_samples,
                    "Utility": utility_scores.flatten(),
                    "Novelty": novelty_scores.flatten(),
                    **{col: predictions[:, i] for i, col in enumerate(target_columns)},
                    **{f"Uncertainty_{col}": uncertainty_scores[:, i] for i, col in enumerate(target_columns)},
                    **inputs_infer.reset_index(drop=True).to_dict(orient="list"),
                }).sort_values(by="Utility", ascending=False).reset_index(drop=True)

                # Display Results
                st.write("### Results Table")
                st.dataframe(result_df, use_container_width=True)

                # t-SNE Plot
                if len(input_columns) >= 2:
                    tsne_fig = create_tsne_plot(result_df, input_columns, "Utility")
                    st.write("### t-SNE Visualization")
                    st.plotly_chart(tsne_fig)

                # Scatter Plot
                if len(target_columns) > 1:
                    scatter_fig = plot_scatter_matrix(result_df, target_columns, result_df["Utility"])
                    st.write("### Scatter Matrix")
                    st.plotly_chart(scatter_fig)
                else:
                    scatter_fig = px.scatter(
                        result_df, x=target_columns[0], y="Utility",
                        title="Utility vs Target Property",
                        color="Utility", color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(scatter_fig)

                # Download Options
                csv = result_df.to_csv(index=False)
                st.download_button("Download Results as CSV", data=csv, file_name="results.csv", mime="text/csv")

                # Save Session
                session_data = {
                    "input_columns": input_columns,
                    "target_columns": target_columns,
                    "apriori_columns": apriori_columns,
                    "weights_targets": weights_targets,
                    "curiosity": curiosity,
                    "results": result_df.to_dict(orient="records"),
                }
                session_json = json.dumps(session_data, indent=4)
                st.download_button(
                    "Download Session as JSON", data=session_json, file_name="session.json", mime="application/json"
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")



















