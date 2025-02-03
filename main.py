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
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.spatial import distance


from app.reptile_model import ReptileModel, reptile_train
from app.models import MAMLModel, meta_train
from app.utils import calculate_utility, calculate_novelty
from app.visualization import plot_scatter_matrix_with_uncertainty, create_tsne_plot_with_hover
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



def enforce_diversity(candidate_inputs, selected_inputs, min_distance=5):
    """
    Filters candidate samples to ensure diversity.
    Ensures that selected samples are sufficiently different from existing ones.
    """
    diverse_candidates = []
    for candidate in candidate_inputs:
        distances = [np.linalg.norm(candidate - existing) for existing in selected_inputs]
        if all(d > min_distance for d in distances):  # Keep only if it's sufficiently different
            diverse_candidates.append(candidate)
    
    return np.array(diverse_candidates) if diverse_candidates else candidate_inputs  # Fallback to all if too strict

def bayesian_optimization(train_inputs, train_targets, candidate_inputs, n_calls=None):
    """
    Uses Bayesian Optimization to select the most promising sample from candidate inputs.
    Ensures that the search space does not include degenerate features.
    """
    kernel = Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    # Fit GPR model on known data
    gpr.fit(train_inputs, train_targets)

    # Define acquisition function (UCB)
    def objective_function(sample):
        sample = np.array(sample).reshape(1, -1)

        # Ensure predictions are obtained safely
        try:
            mean, std = gpr.predict(sample, return_std=True)

            kappa = 1.96  # Adjust exploration-exploitation balance

            # ✅ Ensure it returns a single scalar
            if isinstance(mean, np.ndarray) and mean.ndim > 0:
                mean = float(mean.mean())  # Take mean of all target predictions
            if isinstance(std, np.ndarray) and std.ndim > 0:
                std = float(std.mean())  # Take mean of uncertainties

            return -(mean + kappa * std)  # Minimize the negative value for optimization
        
        except Exception as e:
            print(f"Error in objective function: {e}")
            return np.inf  # Return a large penalty value to prevent optimizer from breaking



    # =============================
    # ✅ Debugging: Check if Any Feature Has a Constant Value
    # =============================
    valid_indices = []
    for i in range(candidate_inputs.shape[1]):
        min_val, max_val = min(candidate_inputs[:, i]), max(candidate_inputs[:, i])
        if min_val != max_val:  # Include only valid features
            valid_indices.append(i)

    if not valid_indices:
        raise ValueError("All features have constant values, Bayesian Optimization is not possible.")

    # Filter candidate inputs to exclude constant features
    candidate_inputs_filtered = candidate_inputs[:, valid_indices]

    # Define the search space with valid features only
    space = [Real(min(candidate_inputs_filtered[:, i]), max(candidate_inputs_filtered[:, i])) for i in range(candidate_inputs_filtered.shape[1])]

    # Dynamic Bayesian Optimization iterations
    if n_calls is None:
        n_calls = min(len(candidate_inputs_filtered), 20)  # Auto-adjust iterations

    # Run Bayesian Optimization
    result = gp_minimize(objective_function, space, n_calls=n_calls, random_state=42)

    # Select the best sample based on Mahalanobis distance
    best_sample = np.array(result.x).reshape(1, -1)
    covariance = np.cov(candidate_inputs_filtered.T)  # Covariance matrix for Mahalanobis
    inv_covariance = np.linalg.pinv(covariance)  # Inverse covariance
    distances = np.array([distance.mahalanobis(x, best_sample[0], inv_covariance) for x in candidate_inputs_filtered])

    # Enforce diversity
    candidate_inputs_diverse = enforce_diversity(candidate_inputs_filtered, train_inputs)
    
    if len(candidate_inputs_diverse) > 0:
        best_sample_idx = np.argmin(distances)  # Select the most "distant" candidate
    else:
        best_sample_idx = np.random.randint(0, len(candidate_inputs_filtered))  # Fallback to a random sample
    
    return best_sample_idx





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
        ["Reptile", "MAML"],  # Add available models here
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

        if not input_columns or not target_columns:
            st.error("Please select at least one input feature and one target property.")
        else:
            try:
                # =============================
                # ✅ Step 1: Data Preparation
                # =============================
                known_targets = ~data[target_columns[0]].isna()
                inputs_train = data.loc[known_targets, input_columns]
                targets_train = data.loc[known_targets, target_columns]
                inputs_infer = data.loc[~known_targets, input_columns]

                # ✅ Handle Apriori Data if Available
                if apriori_columns:
                    apriori_data = data[apriori_columns]
                    apriori_train = apriori_data.loc[known_targets]
                    apriori_infer = apriori_data.loc[~known_targets]
                else:
                    apriori_train, apriori_infer = None, None  # No apriori data

                # =============================
                # ✅ Step 2: Ensure No Constant Features
                # =============================
                valid_columns = [col for col in inputs_infer.columns if inputs_infer[col].nunique() > 1]
                inputs_train = inputs_train[valid_columns]
                inputs_infer = inputs_infer[valid_columns]

            

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

                # Debugging array lengths before DataFrame creation
                st.write(f"idx_samples: {len(idx_samples)}")
                st.write(f"Utility Scores: {len(utility_scores)}")
                st.write(f"Novelty Scores: {len(novelty_scores)}")
                st.write(f"Uncertainty Scores: {len(uncertainty_scores)}")
                st.write(f"Predictions shape: {predictions.shape}")  # Should be (num_samples, num_targets)

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

                    # Debugging: Check all array lengths before creating DataFrame
                    st.write(f"📊 Final Check Before DataFrame Creation:")
                    st.write(f"- idx_samples: {len(idx_samples)}")
                    st.write(f"- Utility Scores: {len(utility_scores)}")
                    st.write(f"- Novelty Scores: {len(novelty_scores)}")
                    st.write(f"- Uncertainty Scores: {len(uncertainty_scores)}")
                    st.write(f"- Predictions Shape: {predictions.shape}")

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

