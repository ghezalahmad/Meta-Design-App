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


from train_and_infer import meta_train
from train_and_infer import MAMLModel
from visualization import plot_scatter_matrix, create_tsne_plot
from session_management import restore_session
from utility_calculations import calculate_novelty, calculate_utility
from gan_model import Generator, Discriminator, train_gan, generate_material_designs
from utility_calculations import calculate_uncertainty
from utility_calculations import prepare_results_dataframe



# Set up directories
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Streamlit layout
st.set_page_config(page_title="MAML Dashboard", layout="wide")
st.title("MAML Dashboard")

# Sidebar Configuration
# Initialize global variables with default values
# Default global configurations
hidden_size = 128  # Default hidden layer size for MAML
learning_rate = 0.01  # Default learning rate
curiosity = 0.0  # Default curiosity setting for explore/exploit
scheduler_type = "None"  # Default scheduler type
scheduler_params = {}  # Default scheduler parameters
meta_epochs = 50  # Default number of meta-training epochs
inner_lr = 0.01  # Default inner loop learning rate
outer_lr = 0.01  # Default outer loop learning rate
num_tasks = 5  # Default number of simulated tasks



# Sidebar: Model Selection
with st.sidebar.expander("Model Selection", expanded=True):  # Expanded by default
    model_type = st.selectbox(
        "Choose Model Type:",
        ["MAML", "GAN"],  # Add available models here
        help="Select the model type to use for material mix design."
    )


# Model Configuration Section
if model_type == "MAML":
    with st.sidebar.expander("MAML Configuration", expanded=True):
        hidden_size = st.slider(
            "Hidden Size (MAML):", 
            min_value=64, 
            max_value=256, 
            step=16, 
            value=128, 
            help="The number of neurons in the hidden layers of the MAML model. Larger sizes capture more complex patterns but increase training time."
        )

        learning_rate = st.slider(
            "Learning Rate:", 
            min_value=0.001, 
            max_value=0.1, 
            step=0.001, 
            value=0.01, 
            help="The step size for updating model weights during optimization. Higher values accelerate training but may overshoot optimal solutions."
        )

        curiosity = st.slider(
            "Curiosity (Explore vs Exploit):", 
            min_value=-2.0, 
            max_value=2.0, 
            step=0.1, 
            value=0.0, 
            help="Balances exploration and exploitation. Negative values focus on high-confidence predictions, while positive values prioritize exploring uncertain regions."
        )

    with st.sidebar.expander("Learning Rate Scheduler", expanded=False):
        scheduler_type = st.selectbox(
            "Scheduler Type:", 
            ["None", "CosineAnnealing", "ReduceLROnPlateau"], 
            index=0, 
            help="Choose the learning rate scheduler type:\n- None: Keeps the learning rate constant.\n- CosineAnnealing: Gradually reduces the learning rate in a cosine curve.\n- ReduceLROnPlateau: Lowers the learning rate when the loss plateaus."
        )

        if scheduler_type == "CosineAnnealing":
            scheduler_params["T_max"] = st.slider(
                "T_max (CosineAnnealing):", 
                min_value=10, 
                max_value=100, 
                step=10, 
                value=50, 
                help="The number of iterations over which the learning rate decreases in a cosine curve."
            )
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler_params["factor"] = st.slider(
                "Factor (ReduceLROnPlateau):", 
                min_value=0.1, 
                max_value=0.9, 
                step=0.1, 
                value=0.5, 
                help="The factor by which the learning rate is reduced when the loss plateaus."
            )
            scheduler_params["patience"] = st.slider(
                "Patience (ReduceLROnPlateau):", 
                min_value=1, 
                max_value=10, 
                step=1, 
                value=5, 
                help="The number of epochs to wait before reducing the learning rate after the loss plateaus."
            )

    with st.sidebar.expander("Meta-Training Configuration", expanded=False):
        meta_epochs = st.slider(
            "Meta-Training Epochs:", 
            min_value=10, 
            max_value=100, 
            step=10, 
            value=50, 
            help="The number of iterations over all simulated tasks during meta-training. Higher values improve adaptability but increase training time."
        )

        inner_lr = st.slider(
            "Inner Loop Learning Rate:", 
            min_value=0.001, 
            max_value=0.1, 
            step=0.001, 
            value=0.01, 
            help="The learning rate for task-specific adaptation during the inner loop. Controls how quickly the model adapts to a single task."
        )

        outer_lr = st.slider(
            "Outer Loop Learning Rate:", 
            min_value=0.001, 
            max_value=0.1, 
            step=0.001, 
            value=0.01, 
            help="The learning rate for updating meta-parameters in the outer loop. A lower value ensures stability, while a higher value speeds up training."
        )

        num_tasks = st.slider(
            "Number of Tasks:", 
            min_value=2, 
            max_value=10, 
            step=1, 
            value=5, 
            help="The number of tasks (subsets of the dataset) simulated during each epoch of meta-training. More tasks improve generalization but increase computation."
        )
elif model_type == "GAN":
    # GAN Configuration Section
    with st.sidebar.expander("GAN Configuration", expanded=False):  # Collapsed by default
        gan_epochs = st.slider(
            "GAN Training Epochs:", 
            min_value=10, 
            max_value=500, 
            step=10, 
            value=100, 
            help="The number of epochs for training the GAN model."
        )

        gan_lr = st.number_input(
            "GAN Learning Rate:", 
            min_value=0.00001, 
            max_value=0.01, 
            step=0.00001, 
            value=0.0002, 
            help="Learning rate for training the GAN model."
        )

        latent_dim = st.slider(
            "Latent Dimension (GAN):", 
            min_value=2, 
            max_value=100, 
            step=1, 
            value=10, 
            help="The size of the random latent space for GAN input."
        )

        gan_batch_size = st.slider(
            "GAN Batch Size:", 
            min_value=8, 
            max_value=128, 
            step=8, 
            value=32, 
            help="Batch size for training the GAN model."
        )


# Sidebar: Restore Session
# Sidebar: Restore Session
st.sidebar.header("Restore Session")
uploaded_session = st.sidebar.file_uploader(
    "Upload Session File (JSON):",
    type=["json"],
    help="Upload a previously saved session file to restore your configuration."
)

if uploaded_session:
    try:
        # Load and parse the uploaded JSON session file
        session_data = json.load(uploaded_session)
        restored_session = restore_session(session_data)
        
        # Apply restored session values
        input_columns = restored_session["input_columns"]
        target_columns = restored_session["target_columns"]
        apriori_columns = restored_session["apriori_columns"]
        weights_targets = restored_session["weights_targets"]
        weights_apriori = restored_session["weights_apriori"]
        thresholds_targets = restored_session["thresholds_targets"]
        thresholds_apriori = restored_session["thresholds_apriori"]
        curiosity = restored_session["curiosity"]
        result_df = restored_session["results"]

        st.sidebar.success("Session restored successfully!")

        # Display restored dataset and results (optional)
        if not result_df.empty:
            st.write("### Restored Results Table")
            st.dataframe(result_df, use_container_width=True)

    except Exception as e:
        st.sidebar.error(f"Failed to restore session: {str(e)}")


# File upload
# Initialize input, target, and apriori columns globally
input_columns = []
target_columns = []
apriori_columns = []

# File upload
uploaded_file = st.file_uploader("Upload Dataset (CSV format):", type=["csv"])
if uploaded_file:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    data = pd.read_csv(file_path)
    st.success("Dataset uploaded successfully!")
    st.dataframe(data)

    # Feature selection
    st.header("Select Features")
    input_columns = st.multiselect(
        "Input Features:",
        options=data.columns.tolist(),
        default=input_columns  # Initialized as empty list above
    )

    remaining_columns = [col for col in data.columns if col not in input_columns]
    target_columns = st.multiselect(
        "Target Properties:",
        options=remaining_columns,
        default=target_columns  # Initialized as empty list above
    )

    remaining_columns_aprior = [col for col in remaining_columns if col not in target_columns]
    apriori_columns = st.multiselect(
        "A Priori Properties:",
        options=remaining_columns_aprior,
        default=apriori_columns  # Initialized as empty list above
    )
    st.write(f"Selected a priori columns: {apriori_columns}")



    # Target settings
    st.header("Target Settings")
    max_or_min_targets = []
    weights_targets = []
    thresholds_targets = []
    for col in target_columns:
        with st.expander(f"Target: {col}"):
            optimize_for = st.radio(f"Optimize {col} for:", ["Maximize", "Minimize"], index=0)
            weight = st.number_input(f"Weight for {col}:", value=1.0, step=0.1)
            threshold = st.text_input(f"Threshold (optional) for {col}:", value="")
            max_or_min_targets.append("max" if optimize_for == "Maximize" else "min")
            weights_targets.append(weight)
            thresholds_targets.append(float(threshold) if threshold else None)

    
    # Apriori Settings
    st.header("Apriori Settings")
    max_or_min_apriori = []
    weights_apriori = []
    thresholds_apriori = []

    for col in apriori_columns:
        with st.expander(f"A Priori: {col}"):
            optimize_for = st.radio(f"Optimize {col} for:", ["Maximize", "Minimize"], index=0)
            weight = st.number_input(f"Weight for {col}:", value=1.0, step=0.1)
            threshold = st.text_input(f"Threshold (optional) for {col}:", value="")
            max_or_min_apriori.append("max" if optimize_for == "Maximize" else "min")
            weights_apriori.append(weight)
            thresholds_apriori.append(float(threshold) if threshold else None)

   



    # Initialize session state for result_df and experiment flags
    if "result_df" not in st.session_state:
        st.session_state.result_df = pd.DataFrame()

    if "experiment_run" not in st.session_state:
        st.session_state.experiment_run = False

    if "tsne_generated" not in st.session_state:
        st.session_state.tsne_generated = False

    if "dropdown_option" not in st.session_state:
        st.session_state.dropdown_option = "None"

    # Dropdown menu for additional functionalities (always available)
    st.session_state.dropdown_option = st.selectbox(
        "Select an Analysis Option:",
        ["None", "Generate t-SNE Plot", "Radar Chart"],
        key="dropdown_menu",
    )

    # Experiment execution
    # Experiment execution
if st.button("Run Experiment", key="run_experiment"):
    st.session_state.experiment_run = True
    st.session_state.tsne_generated = False  # Reset t-SNE flag

    if not input_columns or not target_columns:
        st.error("Please select at least one input feature and one target property.")
    else:
        try:
            # Prepare data
            known_targets = ~data[target_columns[0]].isna()
            inputs_train = data.loc[known_targets, input_columns]
            targets_train = data.loc[known_targets, target_columns]
            inputs_infer = data.loc[~known_targets, input_columns]

            # Handle Idx_Sample (assign sequential IDs if not present)
            if "Idx_Sample" in data.columns:
                idx_samples = data.loc[~known_targets, "Idx_Sample"].reset_index(drop=True)
            else:
                idx_samples = pd.Series(range(1, len(inputs_infer) + 1), name="Idx_Sample")

            # Scale input data
            scaler_inputs = StandardScaler()
            inputs_train_scaled = scaler_inputs.fit_transform(inputs_train)
            inputs_infer_scaled = scaler_inputs.transform(inputs_infer)

            # Scale target data (only for MAML)
            if model_type == "MAML":
                scaler_targets = StandardScaler()
                targets_train_scaled = scaler_targets.fit_transform(targets_train)

            # Scale a priori data (if selected)
            if apriori_columns:
                apriori_data = data[apriori_columns]
                scaler_apriori = StandardScaler()
                apriori_scaled = scaler_apriori.fit_transform(apriori_data.loc[known_targets])
                apriori_infer_scaled = scaler_apriori.transform(apriori_data.loc[~known_targets])
            else:
                apriori_infer_scaled = np.zeros((inputs_infer.shape[0], 1))  # Default to zeros if no a priori data

            # Ensure default values for weights, max_or_min, and thresholds for a priori columns
            weights_apriori = weights_apriori if len(apriori_columns) > 0 else []
            max_or_min_apriori = max_or_min_apriori if len(apriori_columns) > 0 else []
            thresholds_apriori = thresholds_apriori if len(apriori_columns) > 0 else []

            # =============================
            # Model Execution (MAML or GAN)
            # =============================
            if model_type == "MAML":
                # =======================
                # Run MAML Experiment
                # =======================
                st.write("### Running MAML Model")
                meta_model = MAMLModel(len(input_columns), len(target_columns), hidden_size=hidden_size)
                meta_model = meta_train(
                    meta_model=meta_model,
                    data=data,
                    input_columns=input_columns,
                    target_columns=target_columns,
                    hidden_size=hidden_size,
                    epochs=meta_epochs,
                    inner_lr=inner_lr,
                    outer_lr=learning_rate,
                    num_tasks=num_tasks,
                )
                st.write("Meta-training completed!")

                # Perform inference
                meta_model.eval()
                inputs_infer_tensor = torch.tensor(inputs_infer_scaled, dtype=torch.float32)
                with torch.no_grad():
                    predictions_scaled = meta_model(inputs_infer_tensor).numpy()
                predictions = scaler_targets.inverse_transform(predictions_scaled)

                # Calculate uncertainty for MAML
                if len(target_columns) == 1:
                    uncertainty_scores = calculate_uncertainty(meta_model, inputs_infer_tensor)
                else:
                    uncertainty_scores = np.std(predictions_scaled, axis=1, keepdims=True)

                # Novelty calculation
                novelty_scores = calculate_novelty(inputs_infer_scaled, inputs_train_scaled)

                # Utility calculation
                if len(apriori_columns) > 0:
                    utility_scores = calculate_utility(
                        predictions=predictions,
                        uncertainties=uncertainty_scores,
                        apriori=apriori_infer_scaled,
                        curiosity=curiosity,
                        weights=np.array(weights_targets + weights_apriori),
                        max_or_min=max_or_min_targets + max_or_min_apriori,
                        thresholds=thresholds_targets + thresholds_apriori,
                    )
                else:
                    utility_scores = np.zeros(len(predictions))

            elif model_type == "GAN":
                # =======================
                # Run GAN Experiment
                # =======================
                st.write("### Running GAN Model")

                  # Wrap the training data into a DataLoader for GAN
                train_tensor = torch.tensor(inputs_train_scaled, dtype=torch.float32)
                train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=gan_batch_size, shuffle=True)

                # Initialize Generator and Discriminator
                generator = Generator(latent_dim=latent_dim, output_dim=len(input_columns))
                discriminator = Discriminator(input_dim=len(input_columns))

                # Train GAN
                generator = train_gan(
                    generator=generator,
                    discriminator=discriminator,
                    data_loader=train_loader,
                    epochs=gan_epochs,
                    latent_dim=latent_dim,
                    lr=gan_lr,
                )
                st.write("GAN training completed!")

                # Generate new material designs
                generated_samples = generate_material_designs(generator, latent_dim, num_samples=len(inputs_infer))
                predictions = scaler_inputs.inverse_transform(generated_samples)  # Transform back to original scale
                #st.write("Generated Material Designs:")
                #st.write(predictions)

                # Default values for novelty, uncertainty, and utility (GAN)
                novelty_scores = np.zeros(len(predictions))
                uncertainty_scores = np.zeros((len(predictions), 1))
                utility_scores = np.zeros(len(predictions))

            # ===========================
            # Post-processing and Outputs
            # ===========================
            # Ensure all arrays are of the same length
            num_samples = len(predictions)
            idx_samples = idx_samples[:num_samples]  # Adjust length if necessary

            # Create result dataframe
            result_df = pd.DataFrame({
                "Idx_Sample": idx_samples,
                "Utility": utility_scores,
                "Novelty": novelty_scores,
                "Uncertainty": uncertainty_scores.flatten(),
                **{col: predictions[:, i] for i, col in enumerate(target_columns)},
                **inputs_infer.reset_index(drop=True).to_dict(orient="list"),
            }).sort_values(by="Utility", ascending=False).reset_index(drop=True)

            st.session_state.result_df = result_df
            st.success("Experiment completed successfully!")

            # =======================
            # Visualization and Plots
            # =======================
            st.write("### Results Table")
            st.dataframe(result_df, use_container_width=True)

            # Scatter Plot
            if len(target_columns) > 1:
                scatter_fig = plot_scatter_matrix(result_df, target_columns, utility_scores)
                st.write("### Utility in Output Space (Scatter Matrix)")
                st.plotly_chart(scatter_fig)
            else:
                single_scatter_fig = px.scatter(
                    result_df,
                    x=target_columns[0],
                    y="Utility",
                    title=f"Utility vs {target_columns[0]}",
                    labels={target_columns[0]: target_columns[0], "Utility": "Utility"},
                    color="Utility",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(single_scatter_fig)

            # Radar Chart
            if not result_df.empty:
                categories = target_columns + ["Utility", "Novelty", "Uncertainty"]
                values = [result_df[col].mean() for col in categories]
                radar_fig = go.Figure()
                radar_fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself", name="Metrics"))
                radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
                st.plotly_chart(radar_fig)

            # t-SNE Plot (only for MAML)
            if model_type == "MAML" and st.session_state.dropdown_option == "Generate t-SNE Plot" and not st.session_state.result_df.empty:
                with st.spinner("Generating t-SNE plot..."):
                    try:
                        tsne_plot = create_tsne_plot(
                            data=st.session_state.result_df,
                            features=input_columns,
                            utility_col="Utility",
                        )
                        st.plotly_chart(tsne_plot)
                        st.session_state.tsne_generated = True
                    except Exception as e:
                        st.error(f"An error occurred while generating the t-SNE plot: {str(e)}")

            # Download Results
            csv = result_df.to_csv(index=False)
            st.download_button("Download Results as CSV", data=csv, file_name="results.csv", mime="text/csv")

            # Export Session Data
            session_data = {
                "input_columns": input_columns,
                "target_columns": target_columns,
                "apriori_columns": apriori_columns,
                "weights_targets": weights_targets,
                "weights_apriori": weights_apriori,
                "thresholds_targets": thresholds_targets,
                "thresholds_apriori": thresholds_apriori,
                "curiosity": curiosity,
                "results": result_df.to_dict(orient="records"),
            }
            session_json = json.dumps(session_data, indent=4)
            st.download_button(
                label="Download Session as JSON",
                data=session_json,
                file_name="session.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"An error occurred during the experiment: {str(e)}")
