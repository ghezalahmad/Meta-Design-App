import os
import json
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from app.reptile_model import ReptileModel, reptile_train
from app.models import MAMLModel, meta_train, evaluate_maml
from app.reptile_model import evaluate_reptile
# Import visualization function
from app.visualization import visualize_exploration_exploitation

from app.utils import (
    calculate_utility, calculate_novelty, calculate_uncertainty,
    set_seed, enforce_diversity
)
from app.visualization import (
    plot_scatter_matrix_with_uncertainty, create_tsne_plot_with_hover,
    create_parallel_coordinates, create_3d_scatter
)
from app.llm_suggestion import get_llm_suggestions
from app.session_management import restore_session
from app.bayesian_optimizer import bayesian_optimization


set_seed(42)

# Set up directories
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Streamlit UI Initialization
st.set_page_config(page_title="MetaDesign Dashboard", layout="wide")
st.title("MetaDesign Dashboard")
st.markdown("Optimize material properties using advanced meta-learning techniques like MAML and Reptile.")

# Sidebar: Model Selection
model_type = st.sidebar.selectbox(
    "Choose Model Type:", ["Reptile", "MAML"],
    help="Select the meta-learning model for material mix optimization. "
         "MAML adapts rapidly to new tasks, while Reptile is a simpler alternative."
)

# Add an option to enable/disable visualization in Streamlit
show_visualizations = st.sidebar.checkbox(
    "Show Exploration vs. Exploitation Visualizations", value=False,
    help="Enable to view detailed charts of how curiosity and acquisition functions influence candidate selection."
)



# ✅ Initialize defaults for both models
hidden_size = 128  # Default hidden size
inner_lr = 0.001
outer_lr = 0.00005  # Default learning rates
meta_epochs = 100  # Default for MAML
num_tasks = None  # ✅ Explicitly set number of tasks for MAML
reptile_learning_rate = 0.0005
reptile_epochs = 50
reptile_num_tasks = 5
curiosity = 0.0  # Default curiosity for both

# Sidebar: Model Configuration
if model_type == "MAML":
    st.sidebar.subheader("MAML Model Configuration")

    hidden_size = st.sidebar.slider(
        "Hidden Size:", 64, 256, 128, 16,
        help="Number of neurons in each hidden layer of the MAML model. "
             "Larger values can improve learning but increase computation time."
    )

    # ✅ Improve adaptive inner LR by using validation loss trend
    use_adaptive_inner_lr = st.sidebar.checkbox(
        "Enable Adaptive Inner LR", value=True,
        help="Dynamically adjust inner learning rate based on training performance."
    )

    if use_adaptive_inner_lr:
        inner_lr = 0.0001  # ✅ More stable default
        inner_lr_decay = 0.95  # ✅ Faster decay for stability
    else:
        inner_lr = st.sidebar.slider(
            "Inner Loop Learning Rate:", 0.00005, 0.01, 0.0005, 0.0001, format="%.6f",
            help="Controls how fast the model adapts to individual tasks."
        )

    # ✅ Adaptive Outer LR
    use_adaptive_outer_lr = st.sidebar.checkbox(
        "Enable Adaptive Outer LR", value=True,
        help="Dynamically adjust outer learning rate based on training progress."
    )

    if use_adaptive_outer_lr:
        outer_lr = 0.00002  # ✅ More stable learning rate
    else:
        outer_lr = st.sidebar.slider(
            "Outer Loop Learning Rate:", 0.00001, 0.1, 0.0005, 0.0001, format="%.6f",
            help="Rate at which the meta-model updates across tasks."
        )
    # ✅ Dynamically set meta-training epochs based on early stopping trend
    use_adaptive_epochs = st.sidebar.checkbox(
        "Enable Adaptive Meta-Training", value=True,
        help="Automatically adjust epochs based on training stability."
    )

    if use_adaptive_epochs:
        meta_epochs = 50  # ✅ More stable starting point
    else:
        meta_epochs = st.sidebar.slider(
            "Meta-Training Epochs:", 10, 200, 100, 10,
            help="Total training cycles for meta-learning."
        )

    num_tasks = st.sidebar.slider(
        "Number of Tasks:", 2, 10, 5, 1,
        help="The number of tasks sampled per meta-learning iteration. "
             "Higher values improve generalization but increase computation time."
    )
    acquisition = st.sidebar.selectbox(
        "Acquisition Function", ["EI", "UCB", "PI"], index=1,
        help="Choose the acquisition function for utility calculation."
    )


    curiosity = st.sidebar.slider(
        "Exploit (-2) vs Explore (+2)", -2.0, 2.0, 0.0, 0.1,
        help="Controls the balance between exploitation (using existing knowledge) "
             "and exploration (trying new configurations) in the model's learning process."
    )

    # ✅ Inner LR Decay
    inner_lr_decay = st.sidebar.slider(
        "Inner LR Decay Rate:", 0.95, 1.00, 0.98, 0.01,
        help="Decay rate for the inner learning rate over training. "
            "Slower decay (closer to 1.0) helps stability."
    )


elif model_type == "Reptile":
    st.sidebar.subheader("Reptile Model Configuration")

    reptile_hidden_size = st.sidebar.slider(
        "Hidden Size:", 64, 256, 128, 16,
        help="Number of neurons in each hidden layer of the Reptile model."
    )

    # ✅ Adaptive Learning Rate
    use_adaptive_reptile_lr = st.sidebar.checkbox(
        "Enable Adaptive Learning Rate", value=True,
        help="If enabled, the learning rate dynamically adjusts based on training progress."
    )

    if use_adaptive_reptile_lr:
        reptile_learning_rate = 0.01  # Initial value for adaptive tuning
    else:
        reptile_learning_rate = st.sidebar.slider(
            "Reptile Learning Rate:", 0.0001, 0.1, 0.01, 0.001,
            help="Controls how fast the model adapts to tasks. Higher values lead to faster learning "
                 "but may cause instability."
        )

    # ✅ Adaptive Batch Size
    use_adaptive_reptile_batch = st.sidebar.checkbox(
        "Enable Adaptive Batch Size", value=True,
        help="If enabled, batch size dynamically adjusts for stable training."
    )

    if use_adaptive_reptile_batch:
        reptile_batch_size = 16  # Initial batch size for adaptive tuning
    else:
        reptile_batch_size = st.sidebar.slider(
            "Batch Size:", 8, 128, 32, 8,
            help="Defines how many samples are used per training iteration. "
                 "Larger batch sizes stabilize training but require more memory."
        )

    # ✅ Reptile Training Epochs
    reptile_epochs = st.sidebar.slider(
        "Reptile Training Epochs:", 10, 200, 50, 10,
        help="Number of training cycles for optimizing the Reptile model."
    )

    # ✅ Number of Tasks
    reptile_num_tasks = st.sidebar.slider(
        "Number of Tasks:", 2, 10, 5, 1,
        help="Defines the number of tasks per training step. More tasks improve generalization "
             "but require more computation."
    )

    acquisition = st.sidebar.selectbox(
        "Acquisition Function", ["EI", "UCB", "PI"], index=1,
        help="Choose the acquisition function for utility calculation."
    )

    curiosity = st.sidebar.slider(
        "Exploit (-2) vs Explore (+2)", -2.0, 2.0, 0.0, 0.1,
        help="Adjusts the balance between utilizing known solutions (exploitation) "
             "and discovering new solutions (exploration)."
    )
# Load & Process Dataset
# Load & Process Dataset
def load_dataset():
    uploaded_file = st.file_uploader("Upload Dataset (CSV format):", type=["csv"])
    if uploaded_file:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(file_path)
        st.success("Dataset uploaded successfully!")
        return df
    return None

data = load_dataset()

if data is not None:
    st.dataframe(data, use_container_width=True)
    input_columns = st.multiselect("Input Features:", options=data.columns.tolist())
    target_columns = st.multiselect("Target Properties:", options=[col for col in data.columns if col not in input_columns])
    apriori_columns = st.multiselect("A Priori Properties:", options=[col for col in data.columns if col not in input_columns + target_columns])

    # Target Properties Configuration
    max_or_min_targets, weights_targets, thresholds_targets = [], [], []
    max_or_min_apriori, weights_apriori, thresholds_apriori = [], [], []

    for category, columns, max_or_min_list, weights_list, thresholds_list, key_prefix in [
        ("Target", target_columns, max_or_min_targets, weights_targets, thresholds_targets, "target"),
        ("A Priori", apriori_columns, max_or_min_apriori, weights_apriori, thresholds_apriori, "apriori")
    ]:
        for col in columns:
            with st.expander(f"{category}: {col}"):
                optimize_for = st.radio(
                    f"Optimize {col} for:",
                    ["Maximize", "Minimize"],
                    index=0,
                    key=f"{key_prefix}_opt_{col}"
                )
                weight = st.number_input(f"Weight for {col}:", value=1.0, step=0.1, key=f"{key_prefix}_weight_{col}")
                threshold = st.text_input(f"Threshold (optional) for {col}:", value="", key=f"{key_prefix}_threshold_{col}")
                max_or_min_list.append("max" if optimize_for == "Maximize" else "min")
                weights_list.append(weight)
                thresholds_list.append(float(threshold) if threshold else None)

    # ✅ Ensure session state variables persist before use
    if "plot_option" not in st.session_state:
        st.session_state["plot_option"] = "None"

    if "experiment_run" not in st.session_state:
        st.session_state["experiment_run"] = False

    # ✅ Store user-selected plot option
    plot_option = st.radio(
        "Select Visualization:", 
        ["None", "Scatter Matrix", "t-SNE", "Parallel Coordinates", "3D Scatter"], 
        horizontal=True, 
        key="plot_selection"
    )

    # ✅ Only update session state when the user explicitly selects a plot
    if plot_option != st.session_state["plot_option"]:
        st.session_state["plot_option"] = plot_option

    # ✅ Run Experiment - Ensure experiment state persists
    if st.button("Run Experiment"):
        st.session_state["experiment_run"] = True  # ✅ Experiment ran successfully
        st.session_state["plot_option"] = "None"   # ✅ Reset plot selection for new run

        # Define max_or_min to avoid None issues
        max_or_min = max_or_min_targets if max_or_min_targets else ['max'] * len(target_columns)

        # Ensure weights are properly formatted
        weights = np.clip(np.array(weights_targets) + 0.2, 0.1, 1.0) if weights_targets else np.ones(len(target_columns)) * 0.2

        if model_type == "MAML":
            model = MAMLModel(len(input_columns), len(target_columns), hidden_size)
            model, scaler_inputs, scaler_targets = meta_train(
                model, data, input_columns, target_columns, 
                meta_epochs, inner_lr, outer_lr, num_tasks, 
                inner_lr_decay, curiosity=curiosity
            )

            result_df = evaluate_maml(
                model, 
                data, 
                input_columns, 
                target_columns, 
                curiosity, 
                weights, 
                max_or_min, 
                acquisition=acquisition  # ✅ Explicitly set the acquisition function to "EI"
            )
            if show_visualizations and result_df is not None:
                visualize_exploration_exploitation(result_df, curiosity)

        elif model_type == "Reptile":
            model = ReptileModel(len(input_columns), len(target_columns))
            
<<<<<<< HEAD
            # ✅ Expect only 3 values from `reptile_train`
            model, scaler_x, scaler_y = reptile_train(
                model, data, input_columns, target_columns, 
                reptile_epochs, reptile_learning_rate, reptile_num_tasks
            )

            # ✅ Now call `evaluate_reptile` to generate `result_df`
            result_df = evaluate_reptile(
                model,
                data,
                input_columns,
                target_columns,
                curiosity,
                weights,
                max_or_min,
                acquisition=acquisition  # ✅ Set acquisition function for Reptile model
            )
            if show_visualizations and result_df is not None:
                visualize_exploration_exploitation(result_df, curiosity)



        st.markdown("### 🔍 **Experiment Summary:**")
        st.write(f"**Model Type:** {model_type}")
        st.write(f"**Acquisition Function:** {acquisition}")
        st.write(f"**Curiosity Level:** {curiosity}")
        #st.write(f"**Patience for Early Stopping:** {patience}")

        st.session_state["result_df"] = result_df
        st.dataframe(result_df, use_container_width=True)
        st.markdown("**Suggested Sample for Lab Testing:**")
        st.dataframe(result_df.iloc[0:1], use_container_width=True)

    # ✅ Ensure model has been executed before showing plots (Moved Outside Button Click)
    if st.session_state["experiment_run"] and "result_df" in st.session_state:  
        result_df = st.session_state["result_df"]

        if st.session_state["plot_option"] == "Scatter Matrix":
            fig = plot_scatter_matrix_with_uncertainty(result_df, target_columns)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid target properties selected for Scatter Matrix.")

        elif st.session_state["plot_option"] == "t-SNE":
            fig = create_tsne_plot_with_hover(result_df, input_columns, "Utility")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data for t-SNE.")

        elif st.session_state["plot_option"] == "Parallel Coordinates":
            fig = create_parallel_coordinates(result_df, target_columns)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid target properties selected for Parallel Coordinates.")

        elif st.session_state["plot_option"] == "3D Scatter":
            if len(target_columns) >= 3:
                st.plotly_chart(create_3d_scatter(result_df, target_columns[0], target_columns[1], target_columns[2]), use_container_width=True)
            else:
                st.warning("3D Scatter requires at least 3 target properties. Select more.")
=======

                # =============================
                # ✅ Step 3: Scale Data
                # =============================
                scaler_inputs = StandardScaler()
                inputs_train_scaled = scaler_inputs.fit_transform(inputs_train)
                inputs_infer_scaled = scaler_inputs.transform(inputs_infer)

                scaler_targets = StandardScaler()
                targets_train_scaled = scaler_targets.fit_transform(targets_train)
                


                # ✅ Scale apriori features if available
                if apriori_columns:
                    scaler_apriori = StandardScaler()
                    apriori_train_scaled = scaler_apriori.fit_transform(apriori_train)
                    apriori_infer_scaled = scaler_apriori.transform(apriori_infer)



                # =============================
                # ✅ Step 4: Ensure Index Samples
                # =============================
                if "Idx_Sample" in data.columns:
                    idx_samples = data.loc[~known_targets, "Idx_Sample"].reset_index(drop=True).values
                else:
                    idx_samples = np.arange(1, inputs_infer_scaled.shape[0] + 1)  # Fallback: Sequential numbering


                # =============================
                # ✅ Step 5: Compute Predictions (Fix High Values)
                # =============================

                with torch.no_grad():
                    predictions_scaled = np.random.rand(inputs_infer_scaled.shape[0], len(target_columns))  # Placeholder

                # ✅ Inverse Transform Predictions
                predictions = scaler_targets.inverse_transform(predictions_scaled)
               

                
                # =============================
                # ✅ Step 6: Compute Apriori Predictions
                # =============================
                if apriori_columns:
                    apriori_predictions_original_scale = scaler_apriori.inverse_transform(apriori_infer_scaled)
                    apriori_predictions_original_scale = apriori_predictions_original_scale.reshape(
                        inputs_infer_scaled.shape[0], len(apriori_columns)
                    )
                else:
                    apriori_predictions_original_scale = np.zeros((inputs_infer_scaled.shape[0], 1))


                # =============================
                # ✅ Step 7: Compute Utility, Novelty, and Uncertainty Scores
                # =============================
                novelty_scores = calculate_novelty(inputs_infer_scaled, inputs_train_scaled)
                uncertainty_scores = np.random.rand(inputs_infer_scaled.shape[0], 1)  # Placeholder

                # ✅ Compute Utility with Correct Scaling
                utility_scores = calculate_utility(
                    predictions, uncertainty_scores, apriori_predictions_original_scale,
                    curiosity=curiosity,
                    weights=weights_targets + (weights_apriori if apriori_columns else []),
                    max_or_min=max_or_min_targets + (max_or_min_apriori if apriori_columns else []),
                    thresholds=thresholds_targets + (thresholds_apriori if apriori_columns else []),
                )

                # ✅ Normalize Utility Scores to Fix Large Values
                utility_scores = np.mean(utility_scores, axis=1)
                utility_scores = (utility_scores - np.min(utility_scores)) / (np.max(utility_scores) - np.min(utility_scores)) * 10

  
                # =============================
                # ✅ Step 8: Ensure All Arrays Have the Same Length
                # =============================
                num_samples = inputs_infer_scaled.shape[0]  # Expected sample count

                # Ensure all arrays match num_samples
                idx_samples = np.array(idx_samples).flatten()[:num_samples]
                utility_scores = np.array(utility_scores).flatten()[:num_samples]
                novelty_scores = np.array(novelty_scores).flatten()[:num_samples]
                uncertainty_scores = np.array(uncertainty_scores).flatten()[:num_samples]
                predictions = np.array(predictions).reshape(num_samples, len(target_columns))

                if apriori_columns:
                    apriori_predictions_original_scale = np.array(apriori_predictions_original_scale).reshape(num_samples, len(apriori_columns))
                else:
                    apriori_predictions_original_scale = np.zeros((num_samples, 1))  # Default shape


                # =============================
                # ✅ Step 9: Create Result DataFrame
                # =============================
                result_df = pd.DataFrame({
                    "Idx_Sample": idx_samples,
                    "Utility": utility_scores,
                    "Novelty": novelty_scores,
                    "Uncertainty": uncertainty_scores,
                })

            

                if apriori_columns:
                    st.write(f"Apriori Predictions shape: {apriori_predictions_original_scale.shape}")  # Should match num_samples


                # ✅ Add Target Predictions
                for i, col in enumerate(target_columns):
                    result_df[col] = predictions[:, i]

                # ✅ Add Apriori Predictions (if applicable)
                if apriori_columns:
                    for i, col in enumerate(apriori_columns):
                        result_df[col] = apriori_predictions_original_scale[:, i]

                # ✅ Add Input Features
                result_df = pd.concat([result_df, inputs_infer.reset_index(drop=True)], axis=1)

                # ✅ Sort Results
                result_df = result_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)

                st.success("DataFrame creation successful! No mismatched array sizes.")
                # ✅ Ensure experiment flag is set before moving forward
                st.session_state["experiment_run"] = True
                st.session_state["result_df"] = result_df


                # ✅ Display Results Table
                #st.write("### Results Table")
                #st.dataframe(result_df, use_container_width=True)

                #st.success("DataFrame creation successful! No mismatched array sizes.")

               


        

                # =============================
                # Step 1: Check for Duplicate Samples
                # =============================

                num_unique_samples = pd.Series(inputs_train.apply(tuple, axis=1)).nunique()
                if num_unique_samples < len(inputs_train):
                    st.warning(f"Warning: {len(inputs_train) - num_unique_samples} duplicate samples detected in the selected training set.")
                else:
                    st.success("All selected training samples are unique.")

                # =============================
                # Step 2: Distance Matrix Analysis
                # =============================

                from scipy.spatial.distance import pdist, squareform

                # Compute pairwise Euclidean distances between selected training samples
                distance_matrix = squareform(pdist(inputs_train_scaled)) 

                # Convert distance matrix to DataFrame for easy visualization
                distance_df = pd.DataFrame(distance_matrix, index=inputs_train.index, columns=inputs_train.index)



                # Check if samples are clustered (small distances)
                mean_distance = np.mean(distance_matrix)
                st.write(f"Mean pairwise distance: {mean_distance:.4f}")
                if mean_distance < 5:  # Adjust threshold based on dataset scale
                    st.warning("The selected training samples are highly clustered! Consider adding more diverse samples.")
                else:
                    st.success("The selected training samples are well spread.")

                # Handle Idx_Sample (assign sequential IDs if not present)
                if "Idx_Sample" in data.columns:
                    idx_samples = data.loc[~known_targets, "Idx_Sample"].reset_index(drop=True)
                else:
                    idx_samples = pd.Series(range(1, len(inputs_infer) + 1), name="Idx_Sample")

                # Scale input data
                #scaler_inputs = StandardScaler()
                #inputs_train_scaled = scaler_inputs.fit_transform(inputs_train)
                inputs_infer_scaled = scaler_inputs.transform(inputs_infer)

                # Scale target data
                scaler_targets = StandardScaler()
                targets_train_scaled = scaler_targets.fit_transform(targets_train)

                # Scale a priori data (if selected)
                # Scale a priori data (if selected)
                if apriori_columns:
                    apriori_data = data[apriori_columns]
                    scaler_apriori = StandardScaler()
                    apriori_scaled = scaler_apriori.fit_transform(apriori_data.loc[known_targets])
                    apriori_infer_scaled = scaler_apriori.transform(apriori_data.loc[~known_targets])
                    weights_apriori = [1.0] * len(apriori_columns)  # Assign default weights if apriori columns are selected
                    thresholds_apriori = [None] * len(apriori_columns)  # Assign default thresholds as None for apriori columns
                    apriori_predictions_original_scale = scaler_apriori.inverse_transform(apriori_infer_scaled)

                else:
                    apriori_infer_scaled = np.zeros((inputs_infer.shape[0], 1))  # Default to zeros if no a priori data
                    weights_apriori = []  # Ensure weights_apriori is defined
                    thresholds_apriori = []  # Ensure thresholds_apriori is defined


            


                # =============================
                # Model Execution (MAML)
                # =============================
                if model_type == "MAML":
                    st.write("### Running MAML Model")
                    # Meta-learning predictions and training
                    meta_model = MAMLModel(len(input_columns), len(target_columns), hidden_size=hidden_size)
                    # Perform meta-training
                    # Perform meta-training
                    try:
                        st.write("### Meta-Training Phase")
                        meta_model = meta_train(
                            meta_model=meta_model,
                            data=data,
                            input_columns=input_columns,
                            target_columns=target_columns,
                            epochs=meta_epochs,  # Meta-training epochs (from sidebar)
                            inner_lr=inner_lr,  # Learning rate for inner loop (from sidebar)
                            outer_lr=learning_rate,  # Outer loop learning rate (from sidebar)
                            hidden_size=hidden_size,  # Updated hidden size
                            num_tasks=num_tasks  # Number of simulated tasks (from sidebar)
                        )
                        st.success("Meta-training completed successfully!")
                    except Exception as e:
                        st.error(f"An error occurred during meta-training: {str(e)}")

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

                    # Utility calculation
                    if apriori_infer_scaled.ndim == 1:
                        apriori_infer_scaled = apriori_infer_scaled.reshape(-1, 1)

                    utility_scores = calculate_utility(
                        predictions,
                        uncertainty_scores,
                        apriori_infer_scaled,
                        curiosity=curiosity,
                        weights=weights_targets + (weights_apriori if len(apriori_columns) > 0 else []),  # Combine weights
                        max_or_min=max_or_min_targets + (max_or_min_apriori if len(apriori_columns) > 0 else []),  # Combine min/max
                        thresholds=thresholds_targets + (thresholds_apriori if len(apriori_columns) > 0 else []),  # Combine thresholds
                    )


                   # Determine the expected number of samples
                    num_samples = inputs_infer_scaled.shape[0]  # Expected length based on inference dataset

                    # Ensure all arrays are 1D and match num_samples
                    idx_samples = np.array(idx_samples).flatten()[:num_samples]  # Slice to match length
                    utility_scores = np.array(utility_scores).flatten()[:num_samples]
                    novelty_scores = np.array(novelty_scores).flatten()[:num_samples]
                    uncertainty_scores = np.array(uncertainty_scores).flatten()[:num_samples]

                    # Ensure predictions are properly reshaped
                    predictions = np.array(predictions).reshape(num_samples, len(target_columns))

                    # Handle apriori predictions if they exist
                    if apriori_columns:
                        apriori_predictions_original_scale = np.array(apriori_predictions_original_scale).reshape(num_samples, len(apriori_columns))
                    else:
                        apriori_predictions_original_scale = np.zeros((num_samples, 1))  # Placeholder

                    # ✅ Now, all arrays have the same length


                    # Use Bayesian Optimization to select the next best sample to test
                    best_sample_idx = bayesian_optimization(inputs_train_scaled, targets_train_scaled, inputs_infer_scaled)

                    # Select the best new sample
                    best_new_sample = inputs_infer.iloc[best_sample_idx]

                    # Create result DataFrame with Bayesian-selected sample
                    # Create result DataFrame with ALL samples, but mark the Bayesian-opt selected one
                    result_df = pd.DataFrame({
                        "Idx_Sample": idx_samples,
                        "Utility": utility_scores,
                        "Novelty": novelty_scores,
                        "Uncertainty": uncertainty_scores.flatten(),
                    })

                    # Add target predictions
                    for i, col in enumerate(target_columns):
                        result_df[col] = predictions[:, i]

                    # Add apriori data
                    for i, col in enumerate(apriori_columns):
                        result_df[col] = apriori_predictions_original_scale[:, i]

                    # Add input features
                    result_df = pd.concat([result_df, inputs_infer.reset_index(drop=True)], axis=1)

                    # Sort results by Utility
                    result_df = result_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)

                    # Add a column to highlight the Bayesian-selected sample
                    result_df["Bayesian Selected"] = result_df["Idx_Sample"] == idx_samples[best_sample_idx]





                    st.session_state.result_df = result_df  # Store result in session state
                    st.session_state.experiment_run = True  # Set experiment flag

                    if "experiment_run" not in st.session_state or not st.session_state["experiment_run"]:
                        st.warning("Experiment has not run yet. Please execute the model first.")

                    elif "result_df" in st.session_state and not st.session_state.result_df.empty:
                        selected_option = st.selectbox(
                            "Select Plot to Generate",
                            ["Scatter Matrix", "t-SNE Plot", "Distribution of Selected Samples", "3D Scatter Plot", "Parallel Coordinate Plot", "Scatter Plot", "Pairwise Distance Heatmap"],
                            key="dropdown_menu",
                        )

                        # Add "Pairwise Distance Heatmap" to the dropdown options
                        if st.button("Generate Plot"):
                            if selected_option == "Scatter Plot":
                                scatter_fig = px.scatter(
                                    st.session_state.result_df,
                                    x=target_columns[0],
                                    y="Utility",
                                    color="Utility",
                                    title="Utility vs Target",
                                    labels={"Utility": "Utility", target_columns[0]: target_columns[0]},
                                    template="plotly_white",
                                )
                                st.write("### Scatter Plot")
                                st.plotly_chart(scatter_fig)

                            elif selected_option == "Distribution of Selected Samples":
                                # Get the target column dynamically (last column in dataset)
                                target_col = data.columns[-1]  # Ensures we always get the last column dynamically

                                # Debug: Check available columns in targets_train
                                st.write("Available columns in targets_train:", targets_train.columns.tolist())

                                # Ensure target_col exists in targets_train and data
                                if target_col not in data.columns:
                                    st.error(f"Column '{target_col}' not found in dataset.")
                                elif target_col not in targets_train.columns:
                                    st.error(f"Column '{target_col}' not found in selected training data (targets_train).")
                                else:
                                    # Extract selected training samples
                                    selected_samples = targets_train[target_col].dropna().values.tolist()

                                    # Extract full dataset distribution
                                    full_dataset = data[target_col].dropna().values.tolist()

                                    # Check if data exists
                                    if not selected_samples or not full_dataset:
                                        st.error("No valid data available for plotting. Ensure your dataset is correctly loaded.")
                                    else:
                                        # Create DataFrame for visualization
                                        df_samples = pd.DataFrame({
                                            "Sample Type": ["Selected"] * len(selected_samples) + ["Full Dataset"] * len(full_dataset),
                                            target_col: selected_samples + full_dataset
                                        })

                                        # Create Plotly histogram
                                        fig = px.histogram(
                                            df_samples,
                                            x=target_col,
                                            color="Sample Type",
                                            barmode="overlay",
                                            nbins=20,
                                            title=f"Distribution of Selected Samples vs Full Dataset ({target_col})",
                                            labels={target_col: target_col, "Sample Type": "Data Group"}
                                        )

                                        fig.update_traces(opacity=0.6)
                                        fig.update_layout(
                                            xaxis_title=target_col,
                                            yaxis_title="Frequency",
                                            legend_title="Sample Type"
                                        )

                                        st.plotly_chart(fig)


                            elif selected_option == "Pairwise Distance Heatmap":
                                if 'distance_df' in locals():
                                    # Create a larger interactive heatmap
                                    heatmap_fig = px.imshow(
                                        distance_df,
                                        labels=dict(color="Distance"),
                                        x=distance_df.columns,
                                        y=distance_df.index,
                                        title="Pairwise Distance Heatmap",
                                        color_continuous_scale="viridis",
                                    )

                                    # Adjust figure size and improve visibility
                                    heatmap_fig.update_layout(
                                        height=800,  # Increase height
                                        width=1100,   # Increase width
                                        margin=dict(l=70, r=70, t=70, b=70),  # Adjust margins
                                        xaxis=dict(title="Samples", tickangle=-45, showgrid=False),  # Improve x-axis labels
                                        yaxis=dict(title="Samples", showgrid=False),  # Improve y-axis labels
                                    )

                                    st.write("### Pairwise Distance Heatmap")
                                    st.plotly_chart(heatmap_fig, use_container_width=False)
                                else:
                                    st.error("Distance matrix is not available. Ensure you have computed it before generating this plot.")


                            elif selected_option == "Scatter Matrix":
                                if len(target_columns) < 2:
                                    st.error("Scatter Matrix requires at least two target properties. Please select more target properties.")
                                else:
                                    scatter_matrix_fig = plot_scatter_matrix_with_uncertainty(
                                        result_df=st.session_state.result_df,
                                        target_columns=target_columns,
                                        utility_scores=st.session_state.result_df["Utility"]
                                    )
                                    st.write("### Scatter Matrix of Target Properties")
                                    st.plotly_chart(scatter_matrix_fig)

                            elif selected_option == "t-SNE Plot":
                                if "Utility" in st.session_state.result_df.columns and len(input_columns) > 0:
                                    tsne_plot = create_tsne_plot_with_hover(
                                        data=st.session_state.result_df,
                                        feature_cols=input_columns,
                                        utility_col="Utility",
                                        row_number_col="Idx_Sample"
                                    )
                                    st.write("### t-SNE Plot")
                                    st.plotly_chart(tsne_plot)
                                else:
                                    st.error("Ensure the dataset has utility scores and selected input features.")

                            elif selected_option == "3D Scatter Plot":
                                if len(target_columns) >= 2:
                                    scatter_3d_fig = create_3d_scatter(
                                        result_df=st.session_state.result_df,
                                        x_column=target_columns[0],
                                        y_column=target_columns[1],
                                        z_column="Utility",
                                        color_column="Utility",
                                    )
                                    st.write("### 3D Scatter Plot")
                                    st.plotly_chart(scatter_3d_fig)
                                else:
                                    st.error("3D scatter plot requires at least two target columns.")

                            elif selected_option == "Parallel Coordinate Plot":
                                if len(target_columns) > 1:
                                    dimensions = target_columns + ["Utility", "Novelty", "Uncertainty"]
                                    parallel_fig = create_parallel_coordinates(
                                        result_df=st.session_state.result_df,
                                        dimensions=dimensions,
                                        color_column="Utility",
                                    )
                                    st.write("### Parallel Coordinate Plot")
                                    st.plotly_chart(parallel_fig)
                                else:
                                    st.error("Parallel Coordinate Plot requires at least two target columns.")


                
                
                    # Add a checkbox for highlighting maximum values
                    highlight_checkbox = st.checkbox("Highlight Maximum Values in Results Table")

                    # Display results based on checkbox selection
                    if highlight_checkbox:
                        st.write("### Results Table (Highlighted Maximum Values)")
                        st.dataframe(result_df.style.highlight_max(axis=0), use_container_width=True)
                    else:
                        st.write("### Results Table")
                        st.dataframe(result_df, use_container_width=True)



                    # Add a download button for predictions
                    st.write("### Download Predictions")
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

                # Reset weights function
                def reset_weights(model):
                    """
                    Resets the weights of all layers in the model to their initial state.
                    """
                    for layer in model.children():
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()


                if model_type == "Reptile":
                    st.write("### Running Reptile Model")
                    
                    # Prepare labeled and unlabeled data
                    labeled_data = data.dropna(subset=target_columns).sort_index()  # Data with targets
                    unlabeled_data = data[data[target_columns[0]].isna()].sort_index()  # Data without targets

                    reptile_model = ReptileModel(len(input_columns), len(target_columns), hidden_size=reptile_hidden_size)

                    # Reset model weights for reproducibility
                    reset_weights(reptile_model)

                    # Train Reptile model
                    reptile_model = reptile_train(
                        model=reptile_model,
                        data=labeled_data,  # Use only labeled data for training
                        input_columns=input_columns,
                        target_columns=target_columns,
                        epochs=reptile_epochs,
                        learning_rate=reptile_learning_rate,
                        num_tasks=reptile_num_tasks
                    )

                    st.write("Reptile training completed!")

                    # Perform inference
                    inputs_infer_scaled = scaler_inputs.transform(inputs_infer)
                    inputs_infer_tensor = torch.tensor(inputs_infer_scaled, dtype=torch.float32)

                    with torch.no_grad():
                        predictions_scaled = reptile_model(inputs_infer_tensor).numpy()

                    # Debugging: Check predictions shape
                    st.write(f"🔍 Reptile Predictions (Scaled) Shape: {predictions_scaled.shape}")

                    # Ensure predictions match the expected shape
                    num_samples = inputs_infer_scaled.shape[0]  # Expected sample count
                    if predictions_scaled.shape != (num_samples, len(target_columns)):
                        st.error(f"❌ Shape mismatch! Expected ({num_samples}, {len(target_columns)}), but got {predictions_scaled.shape}")
                        raise ValueError("Reptile model output shape does not match the expected dimensions.")

                    # Convert predictions back to original scale
                    try:
                        predictions = scaler_targets.inverse_transform(predictions_scaled)
                        st.write(f"✅ Final Predictions Shape: {predictions.shape}")  # Should be (num_samples, num_targets)
                    except Exception as e:
                        st.error(f"❌ Error in inverse transform: {str(e)}")
                        raise e

                    # Compute uncertainty
                    uncertainty_scores = calculate_uncertainty(
                        model=reptile_model,
                        inputs=inputs_infer_tensor,
                        num_perturbations=20,
                        noise_scale=0.1,
                    )

                    # Compute novelty
                    novelty_scores = calculate_novelty(inputs_infer_scaled, inputs_train_scaled)

                    # Compute utility
                    utility_scores = calculate_utility(
                        predictions=predictions,
                        uncertainties=uncertainty_scores,
                        apriori=apriori_infer_scaled if apriori_columns else None,
                        curiosity=curiosity,
                        weights=weights_targets + (weights_apriori if apriori_columns else []),
                        max_or_min=max_or_min_targets + (max_or_min_apriori if apriori_columns else []),
                        thresholds=thresholds_targets + (thresholds_apriori if apriori_columns else []),
                    )

                 

                    if apriori_columns:
                        st.write(f"- Apriori Predictions Shape: {apriori_infer_scaled.shape}")

                    # Ensure all arrays are correctly shaped
                    idx_samples = np.array(idx_samples).flatten()[:num_samples]
                    utility_scores = np.array(utility_scores).flatten()[:num_samples]
                    novelty_scores = np.array(novelty_scores).flatten()[:num_samples]
                    uncertainty_scores = np.array(uncertainty_scores).flatten()[:num_samples]
                    predictions = np.array(predictions).reshape(num_samples, len(target_columns))

                    if apriori_columns:
                        apriori_predictions_original_scale = np.array(apriori_infer_scaled).reshape(num_samples, len(apriori_columns))
                    else:
                        apriori_predictions_original_scale = None  # Avoid adding an unnecessary zero column

                    # Use Bayesian Optimization to select the best next sample
                    best_sample_idx = bayesian_optimization(inputs_train_scaled, targets_train_scaled, inputs_infer_scaled)

                    # ✅ Create the result DataFrame
                    result_df = pd.DataFrame({
                        "Idx_Sample": idx_samples,
                        "Utility": utility_scores,
                        "Novelty": novelty_scores,
                        "Uncertainty": uncertainty_scores,
                    })

                    # ✅ Add target predictions
                    for i, col in enumerate(target_columns):
                        result_df[col] = predictions[:, i]

                    # ✅ Add apriori predictions only if they exist
                    if apriori_columns and apriori_predictions_original_scale is not None:
                        apriori_df = pd.DataFrame(apriori_predictions_original_scale, columns=apriori_columns)
                        result_df = pd.concat([result_df, apriori_df], axis=1)

                    # ✅ Add input features
                    result_df = pd.concat([result_df, inputs_infer.reset_index(drop=True)], axis=1)

                    # ✅ Sort results by Utility
                    result_df = result_df.sort_values(by="Utility", ascending=False).reset_index(drop=True)

                    # ✅ Add a column to highlight the Bayesian-selected sample
                    result_df["Bayesian Selected"] = result_df["Idx_Sample"] == idx_samples[best_sample_idx]

                    # ✅ Store results in session state
                    st.session_state.result_df = result_df
                    st.session_state.experiment_run = True





                    # Display Results
                    #st.write("### Results Table")
                    #st.dataframe(result_df, use_container_width=True)

                     # Add a checkbox for highlighting maximum values
                    highlight_checkbox = st.checkbox("Highlight Maximum Values in Results Table")

                    # Display results based on checkbox selection
                    if highlight_checkbox:
                        st.write("### Results Table (Highlighted Maximum Values)")
                        st.dataframe(result_df.style.highlight_max(axis=0), use_container_width=True)
                    else:
                        st.write("### Results Table")
                        st.dataframe(result_df, use_container_width=True)


                    st.session_state.result_df = result_df  # Store result in session state
                    st.session_state.experiment_run = True  # Set experiment flag

                    if st.session_state.experiment_run and "result_df" in st.session_state and not st.session_state.result_df.empty:
                        selected_option = st.selectbox(
                            "Select Plot to Generate",
                            ["Scatter Matrix", "t-SNE Plot", "Distribution of Selected Samples","3D Scatter Plot", "Parallel Coordinate Plot", "Scatter Plot", "Pairwise Distance Heatmap"],
                            key="dropdown_menu",
                        )

                        # Add "Pairwise Distance Heatmap" to the dropdown options
                        if st.button("Generate Plot"):
                            if selected_option == "Scatter Plot":
                                scatter_fig = px.scatter(
                                    st.session_state.result_df,
                                    x=target_columns[0],
                                    y="Utility",
                                    color="Utility",
                                    title="Utility vs Target",
                                    labels={"Utility": "Utility", target_columns[0]: target_columns[0]},
                                    template="plotly_white",
                                )
                                st.write("### Scatter Plot")
                                st.plotly_chart(scatter_fig)
                            
                            elif selected_option == "Distribution of Selected Samples":
                                # Get the target column dynamically (last column in dataset)
                                target_col = data.columns[-1]  # Ensures we always get the last column dynamically

                                # Debug: Check available columns in targets_train
                                st.write("Available columns in targets_train:", targets_train.columns.tolist())

                                # Ensure target_col exists in targets_train and data
                                if target_col not in data.columns:
                                    st.error(f"Column '{target_col}' not found in dataset.")
                                elif target_col not in targets_train.columns:
                                    st.error(f"Column '{target_col}' not found in selected training data (targets_train).")
                                else:
                                    # Extract selected training samples
                                    selected_samples = targets_train[target_col].dropna().values.tolist()

                                    # Extract full dataset distribution
                                    full_dataset = data[target_col].dropna().values.tolist()

                                    # Check if data exists
                                    if not selected_samples or not full_dataset:
                                        st.error("No valid data available for plotting. Ensure your dataset is correctly loaded.")
                                    else:
                                        # Create DataFrame for visualization
                                        df_samples = pd.DataFrame({
                                            "Sample Type": ["Selected"] * len(selected_samples) + ["Full Dataset"] * len(full_dataset),
                                            target_col: selected_samples + full_dataset
                                        })

                                        # Create Plotly histogram
                                        fig = px.histogram(
                                            df_samples,
                                            x=target_col,
                                            color="Sample Type",
                                            barmode="overlay",
                                            nbins=20,
                                            title=f"Distribution of Selected Samples vs Full Dataset ({target_col})",
                                            labels={target_col: target_col, "Sample Type": "Data Group"}
                                        )

                                        fig.update_traces(opacity=0.6)
                                        fig.update_layout(
                                            xaxis_title=target_col,
                                            yaxis_title="Frequency",
                                            legend_title="Sample Type"
                                        )

                                        st.plotly_chart(fig)





                            elif selected_option == "Pairwise Distance Heatmap":
                                if 'distance_df' in locals():
                                    # Create a larger interactive heatmap
                                    heatmap_fig = px.imshow(
                                        distance_df,
                                        labels=dict(color="Distance"),
                                        x=distance_df.columns,
                                        y=distance_df.index,
                                        title="Pairwise Distance Heatmap",
                                        color_continuous_scale="viridis",
                                    )

                                    # Adjust figure size and improve visibility
                                    heatmap_fig.update_layout(
                                        height=800,  # Increase height
                                        width=1200,   # Increase width
                                        margin=dict(l=70, r=70, t=70, b=70),  # Adjust margins
                                        xaxis=dict(title="Samples", tickangle=-45, showgrid=False),  # Improve x-axis labels
                                        yaxis=dict(title="Samples", showgrid=False),  # Improve y-axis labels
                                    )

                                    st.write("### Pairwise Distance Heatmap")
                                    st.plotly_chart(heatmap_fig, use_container_width=False)
                                else:
                                    st.error("Distance matrix is not available. Ensure you have computed it before generating this plot.")


                            elif selected_option == "Scatter Matrix":
                                if len(target_columns) < 2:
                                    st.error("Scatter Matrix requires at least two target properties. Please select more target properties.")
                                else:
                                    scatter_matrix_fig = plot_scatter_matrix_with_uncertainty(
                                        result_df=st.session_state.result_df,
                                        target_columns=target_columns,
                                        utility_scores=st.session_state.result_df["Utility"]
                                    )
                                    st.write("### Scatter Matrix of Target Properties")
                                    st.plotly_chart(scatter_matrix_fig)

                            elif selected_option == "t-SNE Plot":
                                if "Utility" in st.session_state.result_df.columns and len(input_columns) > 0:
                                    tsne_plot = create_tsne_plot_with_hover(
                                        data=st.session_state.result_df,
                                        feature_cols=input_columns,
                                        utility_col="Utility",
                                        row_number_col="Idx_Sample"
                                    )
                                    st.write("### t-SNE Plot")
                                    st.plotly_chart(tsne_plot)
                                else:
                                    st.error("Ensure the dataset has utility scores and selected input features.")

                            elif selected_option == "3D Scatter Plot":
                                if len(target_columns) >= 2:
                                    scatter_3d_fig = create_3d_scatter(
                                        result_df=st.session_state.result_df,
                                        x_column=target_columns[0],
                                        y_column=target_columns[1],
                                        z_column="Utility",
                                        color_column="Utility",
                                    )
                                    st.write("### 3D Scatter Plot")
                                    st.plotly_chart(scatter_3d_fig)
                                else:
                                    st.error("3D scatter plot requires at least two target columns.")

                            elif selected_option == "Parallel Coordinate Plot":
                                if len(target_columns) > 1:
                                    dimensions = target_columns + ["Utility", "Novelty", "Uncertainty"]
                                    parallel_fig = create_parallel_coordinates(
                                        result_df=st.session_state.result_df,
                                        dimensions=dimensions,
                                        color_column="Utility",
                                    )
                                    st.write("### Parallel Coordinate Plot")
                                    st.plotly_chart(parallel_fig)
                                else:
                                    st.error("Parallel Coordinate Plot requires at least two target columns.")

                
                  

                        # Add a download button for predictions
                        st.write("### Download Predictions")
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="reptile_predictions.csv",
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

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

>>>>>>> 332b82c0b0f4a577a61e85a4ae791477265b04cd
