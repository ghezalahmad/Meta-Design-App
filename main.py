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


from app.reptile_model import ReptileModel, reptile_train
from app.models import MAMLModel, meta_train
from app.utils import calculate_utility, calculate_novelty
from app.visualization import plot_scatter_matrix, create_tsne_plot
from app.utils import calculate_uncertainty
from app.visualization import create_parallel_coordinates
from app.visualization import create_3d_scatter
from app.utils import set_seed  # Ensure the function is imported
from app.llm_suggestion import get_llm_suggestions
from app.session_management import restore_session



# Set up directories
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# Define callback function to update checkbox state
def toggle_llm_checkbox():
    st.session_state["llm_checkbox"] = not st.session_state["llm_checkbox"]


# Streamlit app setup
st.set_page_config(page_title="MetaDesign Dashboard", layout="wide")
#st.image("logo.png", width=150)  # Optional logo
st.title("MetaDesign Dashboard")
st.markdown("Optimize material properties using advanced meta-learning techniques like MAML and Reptile.")


# Sidebar: Model Selection
with st.sidebar.expander("Model Selection", expanded=True):  # Expanded by default
    model_type = st.selectbox(
        "Choose Model Type:",
        ["MAML", "Reptile"],  # Add available models here
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
            value=hidden_size if 'hidden_size' in locals() else 128,  # Use session data if available
            help="The number of neurons in the hidden layers of the MAML model. Larger sizes capture more complex patterns but increase training time."
        )

        learning_rate = st.slider(
            "Learning Rate:", 
            min_value=0.001, 
            max_value=0.1, 
            step=0.001, 
            value=learning_rate if 'learning_rate' in locals() else 0.01,  # Use session data if available
            help="The step size for updating model weights during optimization. Higher values accelerate training but may overshoot optimal solutions."
        )

        curiosity = st.slider(
            "Curiosity (Explore vs Exploit):", 
            min_value=-2.0, 
            max_value=2.0, 
            step=0.1, 
            value=curiosity if 'curiosity' in locals() else 0.0,  # Use session data if available
            help="Balances exploration and exploitation. Negative values focus on high-confidence predictions, while positive values prioritize exploring uncertain regions."
        )

    with st.sidebar.expander("Learning Rate Scheduler", expanded=False):
        scheduler_type = st.selectbox(
            "Scheduler Type:", 
            ["None", "CosineAnnealing", "ReduceLROnPlateau"], 
            index=0, 
            help="Choose the learning rate scheduler type:\n- None: Keeps the learning rate constant.\n- CosineAnnealing: Gradually reduces the learning rate in a cosine curve.\n- ReduceLROnPlateau: Lowers the learning rate when the loss plateaus."
        )
        scheduler_params = {}
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
elif model_type == "Reptile":
    with st.sidebar.expander("Reptile Configuration", expanded=True):
        reptile_hidden_size = st.slider(
            "Hidden Size (Reptile):", 
            min_value=64, 
            max_value=256, 
            step=16, 
            value=128, 
            help="The number of neurons in the hidden layers of the Reptile model."
        )

        reptile_learning_rate = st.slider(
            "Learning Rate:", 
            min_value=0.001, 
            max_value=0.1, 
            step=0.001, 
            value=0.01, 
            help="Learning rate for Reptile model updates."
        )

        reptile_epochs = st.slider(
            "Reptile Training Epochs:", 
            min_value=10, 
            max_value=100, 
            step=10, 
            value=50, 
            help="Number of iterations over tasks during Reptile training."
        )

        reptile_num_tasks = st.slider(
            "Number of Tasks:", 
            min_value=2, 
            max_value=10, 
            step=1, 
            value=5, 
            help="Number of tasks simulated in each epoch during Reptile training."
        )

        curiosity = st.slider(
            "Curiosity (Explore vs Exploit):", 
            min_value=-2.0, 
            max_value=2.0, 
            step=0.1, 
            value=0.0, 
            help="Balances exploration and exploitation. Negative values focus on high-confidence predictions, while positive values prioritize exploring uncertain regions."
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




# Initialize session state variables
if "experiment_run" not in st.session_state:
    st.session_state["experiment_run"] = False

if "result_df" not in st.session_state:
    st.session_state["result_df"] = None  # Set to None initially to differentiate

if "llm_suggestions" not in st.session_state:
    st.session_state["llm_suggestions"] = pd.DataFrame()


# File upload
# Initialize input, target, and apriori columns globally
input_columns = []
target_columns = []
apriori_columns = []

# Dataset Upload Section
st.markdown("---")
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



    # Target settings
    if target_columns:
        st.markdown("#### Target Settings")
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
    if apriori_columns:
        st.markdown("#### Apriori Settings")
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

    if "dropdown_option" not in st.session_state:
        st.session_state.dropdown_option = "None"





    # Add "Parallel Coordinate Plot" to the dropdown menu
    st.session_state.dropdown_option = st.selectbox(
        "Select a Plot to Generate:",
        ["None", "Scatter Plot", "Radar Chart", "Generate t-SNE Plot", "3D Scatter Plot", "Parallel Coordinate Plot"],
        key="dropdown_menu",
    )



    # Experiment execution
    if st.button("Run Experiment"):
        st.session_state["experiment_run"] = True
        # Simulate generating results (replace with actual results processing)
        st.session_state["result_df"] = data.head(5)

    # =============================
    # LLM
    # =============================
    # Show Results After Running Experiment
    if st.session_state["experiment_run"] and st.session_state["result_df"] is not None:
        #st.write("### Results Table")
        #st.dataframe(st.session_state["result_df"])  # Display the result table

        # LLM Suggestions Section
        llm_checkbox = st.checkbox("Let LLM suggest the best samples to test in the lab")

        if llm_checkbox:
            api_key = st.text_input("Enter your API Key:", type="password")
            num_samples = st.slider("Number of samples to suggest:", min_value=1, max_value=10, value=3)

            if st.button("Get LLM Suggestions"):
                if not api_key:
                    st.error("Please enter your API key!")
                else:
                    # Simulate LLM suggestion logic
                    suggestions = st.session_state["result_df"].head(num_samples)
                    st.session_state["llm_suggestions"] = suggestions
                    st.success("LLM Suggestions Generated!")

            # Display LLM Suggestions
            if not st.session_state["llm_suggestions"].empty:
                st.write("### Suggested Samples by LLM")
                st.dataframe(st.session_state["llm_suggestions"])
    

        set_seed(42)  # Use a fixed seed for reproducibility

        #st.session_state.experiment_run = True
        #st.session_state.tsne_generated = False  # Reset t-SNE flag
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
                # Scale a priori data (if selected)
                if apriori_columns:
                    apriori_data = data[apriori_columns]
                    scaler_apriori = StandardScaler()
                    apriori_scaled = scaler_apriori.fit_transform(apriori_data.loc[known_targets])
                    apriori_infer_scaled = scaler_apriori.transform(apriori_data.loc[~known_targets])
                    weights_apriori = [1.0] * len(apriori_columns)  # Assign default weights if apriori columns are selected
                    thresholds_apriori = [None] * len(apriori_columns)  # Assign default thresholds as None for apriori columns
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


                    # Ensure all arrays are of the same length
                    num_samples = len(inputs_infer)
                    idx_samples = idx_samples[:num_samples]  # Adjust length if necessary
                    predictions = predictions[:num_samples]
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




                    st.session_state.result_df = result_df  # Store result in session state
                    st.session_state.experiment_run = True  # Set experiment flag
                    # After training or inference
                    # Generate the selected plot
                    # Dropdown menu logic
                    if st.session_state.experiment_run and "result_df" in st.session_state and not st.session_state.result_df.empty:
                        selected_option = st.session_state.dropdown_option

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

                        elif selected_option == "Radar Chart":
                            categories = target_columns + ["Utility", "Novelty", "Uncertainty"]
                            values = [st.session_state.result_df[col].mean() for col in categories]
                            radar_fig = go.Figure()
                            radar_fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself"))
                            st.write("### Radar Chart")
                            st.plotly_chart(radar_fig)

                        elif selected_option == "Generate t-SNE Plot":
                            if "Utility" in st.session_state.result_df.columns and len(input_columns) > 0:
                                tsne_features = input_columns + ["Utility"]
                                tsne_plot = create_tsne_plot(
                                    data=st.session_state.result_df,
                                    features=tsne_features,
                                    utility_col="Utility",
                                )
                                st.write("### t-SNE Plot")
                                st.plotly_chart(tsne_plot)
                            else:
                                st.error("Ensure the dataset has utility scores and selected input features.")

                        elif selected_option == "3D Scatter Plot":
                            if len(target_columns) >= 2:
                                st.write("### 3D Scatter Plot")
                                scatter_3d_fig = create_3d_scatter(
                                    result_df=st.session_state.result_df,
                                    x_column=target_columns[0],
                                    y_column=target_columns[1],
                                    z_column="Utility",
                                    color_column="Utility",
                                )
                                st.plotly_chart(scatter_3d_fig)
                            else:
                                st.error("3D scatter plot requires at least two target properties.")

                        elif selected_option == "Parallel Coordinate Plot":
                            st.write("### Parallel Coordinate Plot")
                            dimensions = target_columns + ["Utility", "Novelty", "Uncertainty"]
                            parallel_fig = create_parallel_coordinates(
                                result_df=st.session_state.result_df,
                                dimensions=dimensions,
                                color_column="Utility",
                            )
                            st.plotly_chart(parallel_fig)


                
                
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
                    # Prepare labeled and unlabeled data
                    labeled_data = data.dropna(subset=target_columns).sort_index()  # Data with targets
                    unlabeled_data = data[data[target_columns[0]].isna()].sort_index()  # Data without targets

                    reptile_model = ReptileModel(len(input_columns), len(target_columns), hidden_size=reptile_hidden_size)

                    # Reset model weights for reproducibility
                    reset_weights(reptile_model)

                    # Train Reptile model
                    # Train Reptile model on labeled data
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
                        #st.write("Predictions (Scaled):", predictions_scaled)

                    # Transform predictions back to original scale
                    predictions = scaler_targets.inverse_transform(predictions_scaled)
                    #st.write("Predictions (Original Scale):", predictions)

                    # Calculate uncertainty
                    uncertainty_scores = calculate_uncertainty(
                        model=reptile_model,
                        inputs=inputs_infer_tensor,
                        num_perturbations=20,
                        noise_scale=0.1,
                    )
                    #st.write("Uncertainty Scores:", uncertainty_scores)

                    # Calculate novelty
                    novelty_scores = calculate_novelty(inputs_infer_scaled, inputs_train_scaled)
                    #st.write("Novelty Scores:", novelty_scores)

                    # Calculate utility
                    utility_scores = calculate_utility(
                        predictions=predictions,
                        uncertainties=uncertainty_scores,
                        apriori=apriori_infer_scaled,
                        curiosity=curiosity,
                        weights=weights_targets + (weights_apriori if len(apriori_columns) > 0 else []),
                        max_or_min=max_or_min_targets + (max_or_min_apriori if len(apriori_columns) > 0 else []),
                        thresholds=thresholds_targets + (thresholds_apriori if len(apriori_columns) > 0 else []),
                    )
                    #st.write("Utility Scores:", utility_scores)

                    # Combine Results
                    result_df = pd.DataFrame({
                        "Idx_Sample": idx_samples,
                        "Utility": utility_scores,
                        "Novelty": novelty_scores,
                        "Uncertainty": uncertainty_scores.flatten(),
                        **{col: predictions[:, i] for i, col in enumerate(target_columns)},
                        **inputs_infer.reset_index(drop=True).to_dict(orient="list"),
                    }).sort_values(by="Utility", ascending=False).reset_index(drop=True)

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
                    # After training or inference
                    # Generate the selected plot
                    # Dropdown menu logic
                    if st.session_state.experiment_run and "result_df" in st.session_state and not st.session_state.result_df.empty:
                        selected_option = st.session_state.dropdown_option

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

                        elif selected_option == "Radar Chart":
                            categories = target_columns + ["Utility", "Novelty", "Uncertainty"]
                            values = [st.session_state.result_df[col].mean() for col in categories]
                            radar_fig = go.Figure()
                            radar_fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself"))
                            st.write("### Radar Chart")
                            st.plotly_chart(radar_fig)

                        elif selected_option == "Generate t-SNE Plot":
                            if "Utility" in st.session_state.result_df.columns and len(input_columns) > 0:
                                tsne_features = input_columns + ["Utility"]
                                tsne_plot = create_tsne_plot(
                                    data=st.session_state.result_df,
                                    features=tsne_features,
                                    utility_col="Utility",
                                )
                                st.write("### t-SNE Plot")
                                st.plotly_chart(tsne_plot)
                            else:
                                st.error("Ensure the dataset has utility scores and selected input features.")

                        elif selected_option == "3D Scatter Plot":
                            if len(target_columns) >= 2:
                                st.write("### 3D Scatter Plot")
                                scatter_3d_fig = create_3d_scatter(
                                    result_df=st.session_state.result_df,
                                    x_column=target_columns[0],
                                    y_column=target_columns[1],
                                    z_column="Utility",
                                    color_column="Utility",
                                )
                                st.plotly_chart(scatter_3d_fig)
                            else:
                                st.error("3D scatter plot requires at least two target properties.")

                        elif selected_option == "Parallel Coordinate Plot":
                            st.write("### Parallel Coordinate Plot")
                            dimensions = target_columns + ["Utility", "Novelty", "Uncertainty"]
                            parallel_fig = create_parallel_coordinates(
                                result_df=st.session_state.result_df,
                                dimensions=dimensions,
                                color_column="Utility",
                            )
                            st.plotly_chart(parallel_fig)



                   

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

