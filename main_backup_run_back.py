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
    with st.sidebar.expander("GAN Configuration", expanded=True):
        latent_dim = st.slider(
            "Latent Dimension:", 
            min_value=5, 
            max_value=50, 
            step=5, 
            value=10, 
            help="The size of the latent space. Larger dimensions capture more variability in the generated material designs."
        )

        gan_epochs = st.slider(
            "Training Epochs (GAN):", 
            min_value=10, 
            max_value=100, 
            step=10, 
            value=50, 
            help="The number of epochs to train the GAN."
        )

        gan_lr = st.slider(
            "Learning Rate (GAN):", 
            min_value=0.0001, 
            max_value=0.01, 
            step=0.0001, 
            value=0.0002, 
            help="The learning rate for training the GAN."
        )

        num_samples = st.slider(
            "Generated Samples:", 
            min_value=10, 
            max_value=100, 
            step=10, 
            value=10, 
            help="The number of material designs to generate after training."
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

                # Scale target data
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
                


                # Ensure default values for weights, max_or_min, and thresholds for a priori columns if empty
                weights_apriori = weights_apriori if len(apriori_columns) > 0 else []
                max_or_min_apriori = max_or_min_apriori if len(apriori_columns) > 0 else []
                thresholds_apriori = thresholds_apriori if len(apriori_columns) > 0 else []


                # Meta-learning predictions and training
                meta_model = MAMLModel(len(input_columns), len(target_columns), hidden_size=hidden_size)
                # Perform meta-training
                st.write("### Meta-Training Phase")
                meta_model = meta_train(
                    meta_model=meta_model,
                    data=data,
                    input_columns=input_columns,
                    target_columns=target_columns,
                    hidden_size=hidden_size,  # Ensure this is passed
                    epochs=meta_epochs,  # Meta-training epochs
                    inner_lr=inner_lr,  # Learning rate for inner loop
                    outer_lr=learning_rate,  # Outer loop learning rate (from sidebar)
                    num_tasks=num_tasks,  # Number of simulated tasks
                )

                st.write("Meta-training completed!")

                # Initialize the optimizer for the MAML model
                optimizer = optim.Adam(meta_model.parameters(), lr=learning_rate)

                # Function to initialize the learning rate scheduler based on user selection
                def initialize_scheduler(optimizer, scheduler_type, **kwargs):
                    """
                    Initialize the learning rate scheduler based on the user selection.

                    Args:
                        optimizer (torch.optim.Optimizer): Optimizer for the model.
                        scheduler_type (str): The type of scheduler selected by the user.
                        kwargs: Additional parameters for the scheduler.

                    Returns:
                        Scheduler object or None.
                    """
                    if scheduler_type == "CosineAnnealing":
                        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
                    elif scheduler_type == "ReduceLROnPlateau":
                        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
                    else:
                        return None

                # Initialize the scheduler with user-selected parameters
                scheduler = initialize_scheduler(optimizer, scheduler_type, **scheduler_params)

                # Convert training data to PyTorch tensors
                inputs_train_tensor = torch.tensor(inputs_train_scaled, dtype=torch.float32)
                targets_train_tensor = torch.tensor(targets_train_scaled, dtype=torch.float32)

                # Training loop
                epochs = 50  # Number of training epochs
                loss_function = torch.nn.MSELoss()  # Define the loss function (can be adjusted)

                for epoch in range(epochs):  # Loop over the epochs
                    meta_model.train()  # Ensure the model is in training mode

                    # Forward pass: Predict outputs for training inputs
                    predictions_train = meta_model(inputs_train_tensor)

                    # Compute the loss between predictions and actual targets
                    loss = loss_function(predictions_train, targets_train_tensor)

                    # Backward pass and optimization
                    optimizer.zero_grad()  # Clear gradients from the previous step
                    loss.backward()        # Compute gradients via backpropagation
                    optimizer.step()       # Update model parameters using gradients

                    # Update the scheduler if applicable
                    if scheduler_type == "ReduceLROnPlateau":
                        scheduler.step(loss)  # Adjust learning rate based on the loss
                    elif scheduler_type == "CosineAnnealing":
                        scheduler.step()      # Adjust learning rate based on the schedule

                    # Log progress every 10 epochs
                    #if (epoch + 1) % 10 == 0:
                    #    st.write(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

                # After training, move to inference
                meta_model.eval()  # Set the model to evaluation mode
                inputs_infer_tensor = torch.tensor(inputs_infer_scaled, dtype=torch.float32)
                with torch.no_grad():
                    predictions_scaled = meta_model(inputs_infer_tensor).numpy()
                predictions = scaler_targets.inverse_transform(predictions_scaled)

                # Ensure predictions are always 2D (reshape for single target property)
                # Calculate uncertainty
                if len(target_columns) == 1:
                    # Perturbation-based uncertainty
                    num_perturbations = 20  # Number of perturbations
                    noise_scale = 0.1  # Adjust noise scale for exploration
                    perturbed_predictions = []

                    for _ in range(num_perturbations):
                        # Add noise to the input tensor
                        perturbed_input = inputs_infer_tensor + torch.normal(0, noise_scale, size=inputs_infer_tensor.shape)
                        perturbed_prediction = meta_model(perturbed_input).detach().numpy()
                        perturbed_predictions.append(perturbed_prediction)

                    # Stack all perturbed predictions and compute the variance
                    perturbed_predictions = np.stack(perturbed_predictions, axis=0)  # Shape: (num_perturbations, num_samples, target_dim)
                    uncertainty_scores = perturbed_predictions.std(axis=0).mean(axis=1, keepdims=True)  # Variance across perturbations
                else:
                    # For multiple target properties, compute uncertainty for each row
                    uncertainty_scores = np.std(predictions_scaled, axis=1, keepdims=True)

                # Novelty calculation
                novelty_scores = calculate_novelty(inputs_infer_scaled, inputs_train_scaled)

                # Ensure apriori_infer_scaled is in the correct shape if a priori columns are selected
                if len(apriori_columns) > 0:
                    if apriori_infer_scaled is not None:
                        # Reshape to 2D if needed
                        if apriori_infer_scaled.ndim == 1:
                            apriori_infer_scaled = apriori_infer_scaled.reshape(-1, 1)
                        
                        # Validate alignment with the number of a priori columns
                        if apriori_infer_scaled.shape[1] != len(apriori_columns):
                            raise ValueError(
                                f"A priori data has incorrect shape {apriori_infer_scaled.shape}. "
                                f"Expected columns: {len(apriori_columns)}."
                            )
                    else:
                        raise ValueError(
                            "A priori data is missing, but a priori columns were selected."
                        )
                else:
                    # No a priori columns selected; ensure apriori_infer_scaled is an empty array
                    apriori_infer_scaled = np.empty((predictions.shape[0], 0))


                # Validate and align inputs for utility calculation
                weights_combined = weights_targets[:len(target_columns)]  # Start with target-related weights
                max_or_min_combined = max_or_min_targets[:len(target_columns)]  # Start with target-related directions
                thresholds_combined = thresholds_targets[:len(target_columns)]  # Start with target-related thresholds

              

                # Add a priori contributions if applicable
                # Add a priori contributions if applicable
                if len(apriori_columns) > 0 and apriori_infer_scaled.shape[1] > 0:
                    # Ensure alignment of weights_apriori with a priori columns
                    if len(weights_apriori) < len(apriori_columns):
                        st.warning(
                            f"Insufficient weights provided for a priori columns. "
                            f"Expected {len(apriori_columns)}, but got {len(weights_apriori)}. Defaulting to zero for missing weights."
                        )
                        weights_apriori = weights_apriori + [0] * (len(apriori_columns) - len(weights_apriori))

                    # Extend weights_combined and other parameters
                    weights_combined.extend(weights_apriori[:len(apriori_columns)])
                    max_or_min_combined.extend(max_or_min_apriori[:len(apriori_columns)])
                    thresholds_combined.extend(thresholds_apriori[:len(apriori_columns)])
                    
                    # Combine predictions with a priori data
                    combined_data = np.hstack([predictions, apriori_infer_scaled])
                else:
                    # Use predictions only if no a priori data is present
                    combined_data = predictions
                  


                # Ensure combined_data aligns with weights, max_or_min, and thresholds
                if combined_data.shape[1] != len(weights_combined):
                    raise ValueError(
                        f"Combined data shape {combined_data.shape[1]} does not match the length of weights_combined {len(weights_combined)}. "
                        "Check that the number of selected targets and a priori columns align with the weights."
                    )

                # Ensure weights for predictions and a priori are correctly split
                prediction_weights = weights_combined[:predictions.shape[1]]
                apriori_weights = weights_combined[predictions.shape[1]:]

                # Validation for a priori weights alignment
                if apriori_infer_scaled.shape[1] > 0 and len(apriori_weights) != apriori_infer_scaled.shape[1]:
                    raise ValueError(
                        f"A priori data columns ({apriori_infer_scaled.shape[1]}) do not match the number of a priori weights ({len(apriori_weights)})."
                    )

                # Adjust the call to calculate_utility to include explicitly split weights
                utility_scores = calculate_utility(
                    predictions=predictions,                # Only prediction-related columns
                    uncertainties=uncertainty_scores,       # Uncertainty scores
                    apriori=apriori_infer_scaled,           # Explicitly pass a priori data
                    curiosity=curiosity,                    # Curiosity factor
                    weights=np.array(prediction_weights + apriori_weights),  # Full combined weights
                    max_or_min=max_or_min_combined,         # Combined optimization directions
                    thresholds=thresholds_combined          # Combined thresholds
                )






                # Ensure all arrays are of the same length
                num_samples = len(inputs_infer)
                idx_samples = idx_samples[:num_samples]  # Adjust length if necessary
                predictions = predictions[:num_samples]
                # Ensure predictions match the number of target columns
                if predictions.shape[1] != len(target_columns):
                    raise ValueError("Predictions shape does not match the number of target columns.")

                utility_scores = utility_scores[:num_samples]
                novelty_scores = novelty_scores[:num_samples]
                uncertainty_scores = uncertainty_scores[:num_samples]

                # Create result dataframe (exclude training samples)
                # global result_df
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
            except Exception as e:
                st.error(f"An error occurred during the experiment: {str(e)}")

                

                
            # Display Results Table and Scatter Plot
            if "result_df" in locals() and not result_df.empty:
                st.write("### Results Table")
                st.dataframe(result_df, use_container_width=True)

                # Scatter Plot
                if len(target_columns) > 1:
                    scatter_fig = plot_scatter_matrix(result_df, target_columns, utility_scores)
                    st.write("### Utility in Output Space (Scatter Matrix)")
                    st.plotly_chart(scatter_fig)
                else:
                    st.write("### Utility vs Target Property")
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

                # Generate t-SNE Plot
                # Handle Dropdown Selection
            if st.session_state.dropdown_option == "Generate t-SNE Plot" and not st.session_state.result_df.empty:
                with st.spinner("Generating t-SNE plot..."):
                    try:
                        tsne_plot = create_tsne_plot(
                            data=st.session_state.result_df,
                            features=input_columns,  # Ensure this matches valid columns
                            utility_col="Utility",
                        )
                        st.plotly_chart(tsne_plot)
                        st.session_state.tsne_generated = True
                    except Exception as e:
                        st.error(f"An error occurred while generating the t-SNE plot: {str(e)}")


            # Radar Chart
            if not st.session_state.result_df.empty:
                st.write("### Radar Chart: Performance Overview")

                # Ensure required columns exist
                if all(col in st.session_state.result_df.columns for col in target_columns + ["Utility", "Novelty", "Uncertainty"]):
                    # Define categories and values
                    categories = target_columns + ["Utility", "Novelty", "Uncertainty"]
                    values = [st.session_state.result_df[col].mean() for col in categories]

                    # Create radar chart
                    radar_fig = go.Figure()
                    radar_fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Metrics'))
                    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
                    st.plotly_chart(radar_fig)
                else:
                    missing_cols = [col for col in target_columns + ["Utility", "Novelty", "Uncertainty"] if col not in st.session_state.result_df.columns]
                    st.error(f"Radar chart cannot be generated. Missing columns: {', '.join(missing_cols)}")


               

                # Add a download button for predictions
                st.write("### Download Predictions")
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

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
                
                    "results": result_df.to_dict(orient="records")
                }
                session_json = json.dumps(session_data, indent=4)
                st.download_button(
                    label="Download Session as JSON",
                    data=session_json,
                    file_name="session.json",
                    mime="application/json"
                ) 


