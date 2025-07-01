
import streamlit as st
st.set_page_config(page_title="MetaDesign Dashboard", layout="wide")
import os
import json
import time
import numpy as np
import pandas as pd
from plotly import graph_objects as go

import torch
import plotly.express as px
from sklearn.preprocessing import StandardScaler, RobustScaler
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode, JsCode

# Import model implementations
from app.models import MAMLModel, meta_train, evaluate_maml, calculate_utility_with_acquisition
from app.reptile_model import ReptileModel, reptile_train, evaluate_reptile
from app.protonet_model import ProtoNetModel, protonet_train, evaluate_protonet
from app.rf_model import RFModel, train_rf_model, evaluate_rf_model

# Import utility functions
from app.utils import (
    calculate_utility, calculate_novelty, calculate_uncertainty,
    set_seed, enforce_diversity, identify_pareto_front,
    balance_exploration_exploitation
)

# Import visualization functions
from app.visualization import (
    visualize_exploration_exploitation,
    plot_scatter_matrix_with_uncertainty, 
    create_tsne_plot_with_hover,
    create_parallel_coordinates, 
    create_3d_scatter,
    create_pareto_front_visualization,
    create_acquisition_function_visualization,
    visualize_exploration_exploitation_tradeoff,
    highlight_optimal_regions,
    visualize_property_distributions,
    visualize_model_comparison,
    _select_error_col_if_available
    
)

from app.bayesian_optimizer import (
    bayesian_optimization, 
    multi_objective_bayesian_optimization,
    BayesianOptimizer
)
# At the top of your Streamlit app
# Set random seed for reproducibility
set_seed(42)

# Set up directories
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Streamlit UI Initialization
st.title("MetaDesign Dashboard")
st.markdown("""
    Optimize material properties using advanced meta-learning techniques. 
    This application helps discover optimal material compositions with minimal experiments.
""")

# --- Experiment State Management ---
st.sidebar.header("Experiment Management")
# Load Experiment State
uploaded_state_file = st.sidebar.file_uploader("Load Experiment State (ZIP)", type=["zip"], key="load_state_uploader")
if uploaded_state_file is not None:
    import io
    import zipfile
    import json
    try:
        with zipfile.ZipFile(uploaded_state_file, "r") as zip_ref:
            # Load dataset.csv
            if "dataset.csv" in zip_ref.namelist():
                with zip_ref.open("dataset.csv") as df_file:
                    st.session_state.dataset = pd.read_csv(df_file)
                    st.success("Dataset loaded from state file.")
                    # To refresh AgGrid and other dependent UI, a rerun might be needed if not immediate.
                    # Forcing selection of columns based on loaded data might be needed if they were empty.
            else:
                st.warning("dataset.csv not found in the uploaded state file.")

            # Load results.csv
            if "results.csv" in zip_ref.namelist():
                with zip_ref.open("results.csv") as results_file:
                    st.session_state.result_df = pd.read_csv(results_file)
                    st.session_state["experiment_run"] = True # Assume experiment was run if results exist
                    st.success("Results loaded from state file.")

            # TODO: Load ui_settings.json and repopulate UI elements
            # TODO: Load model state

        st.sidebar.success("Experiment state partially loaded. Please verify settings.")
        # Clear the uploader to allow re-upload of same filename if needed, after processing
        # This can sometimes be tricky with Streamlit's default file_uploader behavior.
        # A common workaround is to use a button to trigger processing and then clear via a callback or rerun.
        # For now, we'll rely on user uploading a new file if they want to change.

    except Exception as e:
        st.sidebar.error(f"Error loading experiment state: {e}")

st.sidebar.header("Model & Data Configuration")
# Sidebar: Model Selection with improved descriptions
model_type = st.sidebar.selectbox(
    "Choose Model Type:", 
    ["MAML", "Reptile", "ProtoNet", "Random Forest"],
    help="""
    Select the meta-learning model for material mix optimization:
    - MAML (Model-Agnostic Meta-Learning): Adapts rapidly to new tasks with gradient-based adaptation.
    - Reptile: Simpler alternative to MAML with competitive performance and lower computational cost.
    - ProtoNet: Excels in few-shot scenarios by learning an embedding space and prototype-based prediction.
    - Random Forest: Robust ensemble model, good for tabular data and provides uncertainty.
    """
)

# Visualization settings
visualization_expander = st.sidebar.expander("Visualization Settings", expanded=False)
with visualization_expander:
    show_visualizations = st.checkbox(
        "Show Exploration vs. Exploitation Visualizations", value=False,
        help="Enable to view detailed charts of how curiosity and acquisition functions influence candidate selection."
    )
    
    show_acquisition_viz = st.checkbox(
        "Show Acquisition Function Analysis", value=False,
        help="Enable to visualize how the acquisition function balances exploration and exploitation."
    )
    
    show_pareto_front = st.checkbox(
        "Show Pareto Front for Multi-Objective Optimization", value=False,
        help="Enable to visualize the Pareto front for multi-objective optimization."
    )

# Initialize defaults for all models
defaults = {
    "hidden_size": 128,
    "inner_lr": 0.001,
    "outer_lr": 0.00005,
    "meta_epochs": 100,
    "num_tasks": 5,
    "reptile_learning_rate": 0.001,
    "reptile_epochs": 50,
    "reptile_num_tasks": 5,
    "protonet_learning_rate": 0.001,
    "protonet_epochs": 50,
    "protonet_num_tasks": 5,
    "curiosity": 0.0
}

# Model-specific configuration
if model_type == "MAML":
    st.sidebar.subheader("MAML Model Configuration")
    
    with st.sidebar.expander("Network Architecture", expanded=False):
        hidden_size = st.slider("Hidden Size:", 64, 512, defaults["hidden_size"], 32)
        num_layers = st.slider("Number of Layers:", 2, 5, 3, 1)
        dropout_rate = st.slider("Dropout Rate:", 0.0, 0.5, 0.2, 0.05)
    
    with st.sidebar.expander("Meta-Learning Parameters", expanded=True):
        use_adaptive_inner_lr = st.checkbox("Enable Adaptive Inner LR", value=True)
        inner_lr = 0.0001 if use_adaptive_inner_lr else st.slider("Inner Loop Learning Rate:", 0.00005, 0.01, defaults["inner_lr"], 0.0001)
        inner_lr_decay = 0.95 if use_adaptive_inner_lr else st.slider("Inner LR Decay Rate:", 0.8, 1.00, 0.98, 0.01)
        
        use_adaptive_outer_lr = st.checkbox("Enable Adaptive Outer LR", value=True)
        outer_lr = 0.00002 if use_adaptive_outer_lr else st.slider("Outer Loop Learning Rate:", 0.00001, 0.01, defaults["outer_lr"], 0.00001)
        
        use_adaptive_epochs = st.checkbox("Enable Adaptive Training Length", value=True)
        meta_epochs = 100 if use_adaptive_epochs else st.slider("Meta-Training Epochs:", 10, 300, defaults["meta_epochs"], 10)
        
        num_tasks = st.slider("Number of Tasks:", 2, 10, defaults["num_tasks"], 1)
    
    curiosity = st.slider("Exploit (-2) vs Explore (+2)", -2.0, 2.0, defaults["curiosity"], 0.1)
    if curiosity < -1.0:
        st.info("Current strategy: Strong exploitation - focusing on known good regions")
    elif curiosity < 0:
        st.info("Current strategy: Moderate exploitation - slight preference for known regions")
    elif curiosity < 1.0:
        st.info("Current strategy: Balanced approach - considering both known and novel regions")
    else:
        st.info("Current strategy: Strong exploration - actively seeking novel regions")


elif model_type == "Reptile":
    st.sidebar.subheader("Reptile Model Configuration")
    
    with st.sidebar.expander("Network Architecture", expanded=False):
        hidden_size = st.slider("Hidden Size:", 64, 512, defaults["hidden_size"], 32)
        num_layers = st.slider("Number of Layers:", 2, 5, 3, 1)
        dropout_rate = st.slider("Dropout Rate:", 0.0, 0.5, 0.3, 0.05)
    
    with st.sidebar.expander("Training Parameters", expanded=True):
        use_adaptive_reptile_lr = st.checkbox("Enable Adaptive Learning Rate", value=True)
        reptile_learning_rate = 0.005 if use_adaptive_reptile_lr else st.slider("Learning Rate:", 0.0001, 0.1, defaults["reptile_learning_rate"], 0.001)
        
        use_adaptive_batch = st.checkbox("Enable Adaptive Batch Size", value=True)
        batch_size = 16 if use_adaptive_batch else st.slider("Batch Size:", 4, 128, 16, 4)
        
        reptile_epochs = st.slider("Training Epochs:", 10, 300, defaults["reptile_epochs"], 10)
        reptile_num_tasks = st.slider("Number of Tasks:", 2, 10, defaults["reptile_num_tasks"], 1)
        
        curiosity = st.slider("Exploit (-2) vs Explore (+2)", -2.0, 2.0, defaults["curiosity"], 0.1)
        strict_optimization = st.checkbox("Strict Optimization Direction", value=True)
        
        if curiosity < -1.0:
            st.info("Current strategy: Strong exploitation - focusing on known good regions")
        elif curiosity < 0:
            st.info("Current strategy: Moderate exploitation - slight preference for known regions")
        elif curiosity < 1.0:
            st.info("Current strategy: Balanced approach - considering both known and novel regions")
        else:
            st.info("Current strategy: Strong exploration - actively seeking novel regions")

elif model_type == "ProtoNet":
    st.sidebar.subheader("ProtoNet Model Configuration")
    
    with st.sidebar.expander("Network Architecture", expanded=False):
        embedding_size = st.slider("Embedding Size:", 64, 512, 128, 32)
        num_layers = st.slider("Number of Layers:", 2, 5, 3, 1)
        dropout_rate = st.slider("Dropout Rate:", 0.0, 0.5, 0.3, 0.05)
    
    with st.sidebar.expander("Training Parameters", expanded=True):
        if "protonet_learning_rate" not in st.session_state:
            st.session_state["protonet_learning_rate"] = defaults["protonet_learning_rate"]
        
        protonet_learning_rate = st.slider("Learning Rate:", 0.0001, 0.01, st.session_state["protonet_learning_rate"], 0.0001)
        st.session_state["protonet_learning_rate"] = protonet_learning_rate
        
        protonet_epochs = st.slider("Training Epochs:", 10, 300, defaults["protonet_epochs"], 10)
        protonet_num_tasks = st.slider("Number of Tasks:", 2, 10, defaults["protonet_num_tasks"], 1)
        num_shot = st.slider("Support Samples (N-shot):", 1, 8, 4, 1)
        num_query = st.slider("Query Samples:", 1, 8, 4, 1)
    
    curiosity = st.slider("Exploit (-2) vs Explore (+2)", -2.0, 2.0, defaults["curiosity"], 0.1)
    if curiosity < -1.0:
        st.info("Current strategy: Strong exploitation - focusing on known good regions")
    elif curiosity < 0:
        st.info("Current strategy: Moderate exploitation - slight preference for known regions")
    elif curiosity < 1.0:
        st.info("Current strategy: Balanced approach - considering both known and novel regions")
    else:
        st.info("Current strategy: Strong exploration - actively seeking novel regions")

elif model_type == "Random Forest":
    st.sidebar.subheader("Random Forest Configuration")
    with st.sidebar.expander("Model Parameters", expanded=True):
        rf_n_estimators = st.slider("Number of Estimators (Trees):", 50, 500, 100, 10, key="rf_n_estimators")
        # max_depth, min_samples_split, min_samples_leaf could be added if more control is needed
        # For now, keeping it simple.
        rf_perform_grid_search = st.checkbox("Perform GridSearchCV (slower)", value=False, key="rf_grid_search")

    curiosity = st.slider("Exploit (-2) vs Explore (+2)", -2.0, 2.0, defaults["curiosity"], 0.1, key="rf_curiosity")
    if curiosity < -1.0:
        st.info("Current strategy: Strong exploitation - focusing on known good regions")
    elif curiosity < 0:
        st.info("Current strategy: Moderate exploitation - slight preference for known regions")
    elif curiosity < 1.0:
        st.info("Current strategy: Balanced approach - considering both known and novel regions")
    else:
        st.info("Current strategy: Strong exploration - actively seeking novel regions")


# Add ensemble option after model selection
st.subheader("Ensemble Settings")
use_ensemble = st.checkbox(
    "Use Model Ensemble", 
    value=False,
    help="Combine multiple models for more robust predictions. Requires extra computation time."
)

if use_ensemble:
    ensemble_col1, ensemble_col2 = st.columns(2)
    
    with ensemble_col1:
        ensemble_models = st.multiselect(
            "Models to Include:",
            ["MAML", "Reptile", "ProtoNet"],
            default=[model_type],  # Start with currently selected model
            help="Select which models to include in the ensemble"
        )
    
    with ensemble_col2:
        weighting_method = st.selectbox(
            "Ensemble Weighting:",
            ["Equal", "Performance-based", "Uncertainty-based"],
            index=1,
            help="Method to weight models in the ensemble"
        )



# Dataset Management
st.header("Dataset Management")




# Load & Display Dataset with Editing Capability
def load_and_edit_dataset(upload_folder="uploads"):
    uploaded_file = st.file_uploader("Upload Dataset (CSV format):", type=["csv"])
    
    if "dataset" not in st.session_state:
        st.session_state["dataset"] = None

    if uploaded_file:
        file_path = os.path.join(upload_folder, uploaded_file.name)
        os.makedirs(upload_folder, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(file_path)
        st.session_state["dataset"] = df
        st.success("Dataset uploaded successfully!")

    # Display the dataset if available
    if st.session_state["dataset"] is not None:
        st.markdown("### Editable Dataset")
        df = st.session_state["dataset"].copy()
        
        # Make the table editable
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True)
        gb.configure_selection(selection_mode="single", use_checkbox=True)
        grid_options = gb.build()

        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            fit_columns_on_grid_load=True,
            theme="streamlit",
        )

        # Update session state with the modified dataframe
        df_edited = pd.DataFrame(grid_response["data"])
        
        # ✅ Ensure numeric columns stay numeric
        for col in df_edited.columns:
            if df[col].dtype in ["int64", "float64"]:  # Only convert originally numeric columns
                df_edited[col] = pd.to_numeric(df_edited[col], errors="coerce")

        st.session_state["dataset"] = df_edited  # Save back to session state

    return st.session_state["dataset"]





# Run the dataset loader in the Streamlit app
data = load_and_edit_dataset()


# Define feature selection and targets
if data is not None:    
    # Data columns selection with improved UI
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_columns = st.multiselect(
            "Input Features:", 
            options=data.columns.tolist(),
            help="Select the material composition variables"
        )
    
    with col2:
        remaining_cols = [col for col in data.columns if col not in input_columns]
        target_columns = st.multiselect(
            "Target Properties:", 
            options=remaining_cols,
            help="Select the material properties you want to optimize"
        )
    
    with col3:
        remaining_cols = [col for col in data.columns if col not in input_columns + target_columns]
        apriori_columns = st.multiselect(
            "A Priori Properties:", 
            options=remaining_cols,
            help="Select properties with known values that constrain the optimization"
        )

    # ADD FEATURE SELECTION CODE HERE - after user selections but before model initialization
    if len(input_columns) > 0 and len(target_columns) > 0:
        st.subheader("Feature Selection")
        use_feature_selection = st.checkbox(
            "Use Automated Feature Selection", 
            value=False,
            help="Automatically select the most important features to improve model performance"
        )
        
        if use_feature_selection:
            from app.feature_selection import meta_feature_selection
            
            fs_col1, fs_col2 = st.columns(2)
            with fs_col1:
                min_features = st.slider(
                    "Minimum Features", 3, 10, 3, 1,
                    help="Minimum number of features to select"
                )
            with fs_col2:
                max_features = st.slider(
                    "Maximum Features", 5, min(20, len(input_columns)), min(10, len(input_columns)), 1,
                    help="Maximum number of features to select"
                )
            
            with st.spinner("Running feature selection..."):
                selected_features = meta_feature_selection(
                    data, input_columns, target_columns,
                    min_features=min_features,
                    max_features=max_features
                )
                
                # Update the input columns to use only the selected features
                input_columns = selected_features
    
    # Input Feature Constraints Configuration
    if input_columns:
        st.subheader("Input Feature Constraints (Optional)")
        if "constraints" not in st.session_state:
            st.session_state.constraints = {col: {"min": None, "max": None} for col in data.columns}

        expander_constraints = st.expander("Define Min/Max Constraints for Input Features", expanded=False)
        with expander_constraints:
            for col in input_columns:
                # Ensure new columns are added to session state if input_columns change
                if col not in st.session_state.constraints:
                     st.session_state.constraints[col] = {"min": None, "max": None}

                c1, c2 = st.columns(2)
                current_min = st.session_state.constraints[col]["min"]
                current_max = st.session_state.constraints[col]["max"]

                # Use number_input to allow None or float
                new_min = c1.number_input(f"Min for {col}:", value=current_min if current_min is not None else np.nan, format="%g", key=f"min_{col}")
                new_max = c2.number_input(f"Max for {col}:", value=current_max if current_max is not None else np.nan, format="%g", key=f"max_{col}")

                st.session_state.constraints[col]["min"] = None if np.isnan(new_min) else float(new_min)
                st.session_state.constraints[col]["max"] = None if np.isnan(new_max) else float(new_max)

                # Basic validation: min <= max
                if st.session_state.constraints[col]["min"] is not None and \
                   st.session_state.constraints[col]["max"] is not None and \
                   st.session_state.constraints[col]["min"] > st.session_state.constraints[col]["max"]:
                    st.warning(f"For {col}, min value cannot be greater than max value. Adjusting max to be equal to min.")
                    st.session_state.constraints[col]["max"] = st.session_state.constraints[col]["min"]
                    # Rerun to update the UI element value for max if we corrected it (may not be immediate in Streamlit)

            st.markdown("---") # Separator
            st.markdown("##### Define Sum Constraint (e.g., for compositional data sum to 1 or 100)")

            if "sum_constraint_cols" not in st.session_state:
                st.session_state.sum_constraint_cols = []
            if "sum_constraint_target" not in st.session_state:
                st.session_state.sum_constraint_target = 1.0 # Default to 1.0, common for proportions
            if "sum_constraint_tolerance" not in st.session_state:
                st.session_state.sum_constraint_tolerance = 0.01 # Default tolerance

            selected_sum_cols = st.multiselect(
                "Select features for sum constraint:",
                options=input_columns,
                default=st.session_state.sum_constraint_cols,
                key="sum_constraint_features_multiselect"
            )
            st.session_state.sum_constraint_cols = selected_sum_cols

            target_sum_val = st.number_input(
                "Target sum for selected features:",
                value=st.session_state.sum_constraint_target,
                format="%g",
                key="sum_constraint_target_value"
            )
            st.session_state.sum_constraint_target = float(target_sum_val) if target_sum_val is not None else None

            target_sum_tolerance = st.number_input(
                "Tolerance for sum constraint (e.g., +/- 0.01):",
                value=st.session_state.sum_constraint_tolerance,
                min_value=0.0,
                format="%g",
                key="sum_constraint_tolerance_value"
            )
            st.session_state.sum_constraint_tolerance = float(target_sum_tolerance) if target_sum_tolerance is not None else 0.0


    # Target Properties Configuration
    if target_columns:
        st.subheader("Properties Configuration")

        # Multi-objective strategy selection
        mobo_strategy = "weighted_sum" # Default
        if len(target_columns) > 1:
            mobo_strategy = st.selectbox(
                "Multi-Objective Strategy:",
                options=["weighted_sum", "parego"],
                index=0,
                format_func=lambda x: "Weighted Sum (Fixed Weights)" if x == "weighted_sum" else "ParEGO (Randomized Weights)",
                help="Choose how to handle multiple objectives. ParEGO can help explore non-convex Pareto fronts.",
                key="mobo_strategy_selector"
            )
        
        max_or_min_targets, weights_targets, thresholds_targets = [], [], []
        max_or_min_apriori, weights_apriori, thresholds_apriori = [], [], []

        for category, columns, max_or_min_list, weights_list, thresholds_list, key_prefix in [
            ("Target", target_columns, max_or_min_targets, weights_targets, thresholds_targets, "target"),
            ("A Priori", apriori_columns, max_or_min_apriori, weights_apriori, thresholds_apriori, "apriori")
        ]:
            if columns:
                st.markdown(f"#### {category} Properties")
                
                # Create multiple columns for more compact UI
                col_per_property = 3
                properties_per_row = 3
                
                for i in range(0, len(columns), properties_per_row):
                    property_group = columns[i:i+properties_per_row]
                    property_cols = st.columns(len(property_group))
                    
                    for j, col in enumerate(property_group):
                        with property_cols[j]:
                            st.markdown(f"**{col}**")
                            
                            optimize_for = st.radio(
                                f"Optimize for:",
                                ["Maximize", "Minimize"],
                                index=0,
                                key=f"{key_prefix}_opt_{col}",
                                horizontal=True
                            )
                            
                            weight = st.number_input(
                                f"Weight:",
                                value=1.0, 
                                step=0.1, 
                                min_value=0.1,
                                max_value=10.0,
                                key=f"{key_prefix}_weight_{col}"
                            )
                            
                            threshold = st.text_input(
                                f"Threshold:",
                                value="", 
                                key=f"{key_prefix}_threshold_{col}"
                            )
                            
                            max_or_min_list.append("max" if optimize_for == "Maximize" else "min")
                            weights_list.append(weight)
                            thresholds_list.append(float(threshold) if threshold and threshold.strip() else None)

    # Ensure session state variables persist
    if "plot_option" not in st.session_state:
        st.session_state["plot_option"] = "None"

    if "experiment_run" not in st.session_state:
        st.session_state["experiment_run"] = False
    
    if "model_history" not in st.session_state:
        st.session_state["model_history"] = {}

    # Visualization options
    st.subheader("Visualization")
    
    visualization_options = ["None", "Scatter Matrix", "t-SNE", "Parallel Coordinates", "3D Scatter", 
                           "Pareto Front", "Property Distributions", "Acquisition Function Analysis"]
    
    if len(st.session_state.get("model_history", {})) > 1:
        visualization_options.append("Model Comparison")
    
    plot_option = st.radio(
        "Select Visualization:", 
        visualization_options, 
        horizontal=True, 
        key="plot_selection"
    )
    
    # Only update session state when the user explicitly selects a plot
    if plot_option != st.session_state["plot_option"]:
        st.session_state["plot_option"] = plot_option

    # Run Experiment button
    run_col1, run_col2 = st.columns([3, 1])
    run_col1 = st.columns([1])[0]  # A single full-width column

    with run_col1:
        if st.button("Run Experiment", key="run_experiment", use_container_width=True):
            # Validate input
            if not input_columns:
                st.error("Please select at least one input feature.")
            elif not target_columns:
                st.error("Please select at least one target property.")
            else:
                # Apply constraints to the data before further processing
                current_data = data.copy() # Work on a copy
                if "constraints" in st.session_state and st.session_state.constraints:
                    constrained_cols_applied = []
                    for col, bounds in st.session_state.constraints.items():
                        if col in current_data.columns and col in input_columns: # Apply only to selected input features
                            if bounds["min"] is not None:
                                current_data = current_data[current_data[col] >= bounds["min"]]
                                constrained_cols_applied.append(f"{col} >= {bounds['min']}")
                            if bounds["max"] is not None:
                                current_data = current_data[current_data[col] <= bounds["max"]]
                                constrained_cols_applied.append(f"{col} <= {bounds['max']}")
                    if constrained_cols_applied:
                        st.info(f"Applied constraints: {'; '.join(constrained_cols_applied)}. Filtered data from {len(data)} to {len(current_data)} rows.")
                        if len(current_data) == 0:
                            st.error("No data remains after applying bound constraints. Please adjust constraints or data.")
                            st.stop() # Stop execution if no data left

                # Apply sum constraint if defined and data still exists
                if len(current_data) > 0 and \
                   "sum_constraint_cols" in st.session_state and \
                   st.session_state.sum_constraint_cols and \
                   st.session_state.sum_constraint_target is not None:

                    sum_cols = st.session_state.sum_constraint_cols
                    target_sum = st.session_state.sum_constraint_target
                    tolerance = st.session_state.sum_constraint_tolerance

                    # Ensure all selected sum_cols are actually in current_data (they should be if they are input_columns)
                    valid_sum_cols = [col for col in sum_cols if col in current_data.columns]
                    if len(valid_sum_cols) == len(sum_cols) and len(valid_sum_cols) > 0: # All selected cols are valid
                        actual_sums = current_data[valid_sum_cols].sum(axis=1)
                        lower_bound = target_sum - tolerance
                        upper_bound = target_sum + tolerance

                        original_len_before_sum_filter = len(current_data)
                        current_data = current_data[
                            (actual_sums >= lower_bound) & (actual_sums <= upper_bound)
                        ]

                        if len(current_data) < original_len_before_sum_filter:
                             st.info(f"Applied sum constraint ({', '.join(valid_sum_cols)} sum to be approx {target_sum} +/- {tolerance}). Filtered data from {original_len_before_sum_filter} to {len(current_data)} rows.")

                        if len(current_data) == 0:
                            st.error("No data remains after applying sum constraint. Please adjust constraints or data.")
                            st.stop()
                    elif len(sum_cols) > 0 : # Some selected columns for sum are not valid / not found
                        st.warning(f"Could not apply sum constraint: Not all selected features ({', '.join(sum_cols)}) are available in the data for summing.")

                # The 'current_data' DataFrame now respects defined bound and sum constraints.
                # This filtered data will be passed to the training/evaluation functions.

                st.session_state["experiment_run"] = True
                
                # Define max_or_min to avoid None issues
                max_or_min = max_or_min_targets if max_or_min_targets else ['max'] * len(target_columns)
                
                # Ensure weights are properly formatted
                weights = np.array(weights_targets) if weights_targets else np.ones(len(target_columns))
                
                # Import ensemble methods if needed
                if use_ensemble:
                    from app.ensemble import weighted_uncertainty_ensemble
                    
                    # Default parameters for all models to avoid undefined errors
                    if 'meta_epochs' not in locals():
                        meta_epochs = defaults["meta_epochs"]
                    if 'inner_lr' not in locals():
                        inner_lr = defaults["inner_lr"]
                    if 'outer_lr' not in locals():
                        outer_lr = defaults["outer_lr"]
                    if 'inner_lr_decay' not in locals():
                        inner_lr_decay = 0.95
                    if 'num_tasks' not in locals():
                        num_tasks = defaults["num_tasks"]
                        
                    if 'reptile_epochs' not in locals():
                        reptile_epochs = defaults["reptile_epochs"]
                    if 'reptile_learning_rate' not in locals():
                        reptile_learning_rate = defaults["reptile_learning_rate"]
                    if 'reptile_num_tasks' not in locals():
                        reptile_num_tasks = defaults["reptile_num_tasks"]
                        
                    if 'protonet_epochs' not in locals():
                        protonet_epochs = defaults["protonet_epochs"]
                    if 'protonet_learning_rate' not in locals():
                        protonet_learning_rate = defaults["protonet_learning_rate"]
                    if 'protonet_num_tasks' not in locals():
                        protonet_num_tasks = defaults["protonet_num_tasks"]
                    if 'num_shot' not in locals():
                        num_shot = 4
                    if 'num_query' not in locals():
                        num_query = 4
            
                # Display a spinner during computation
                with st.spinner(f"Running {'ensemble' if use_ensemble else model_type} model..."):
                    # Initialize a dictionary to store models if using ensemble
                    models_dict = {}
                    
                    # Train and evaluate single model or models for ensemble
                    if use_ensemble:
                        for model_name in ensemble_models:
                            st.info(f"Training {model_name} model for ensemble...")
                            
                            if model_name == "MAML":
                                # Initialize and train MAML model
                                model = MAMLModel(
                                    input_size=len(input_columns),
                                    output_size=len(target_columns),
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate
                                )
                                
                                model, scaler_inputs, scaler_targets = meta_train(
                                    meta_model=model, 
                                    data=current_data,  # Use constrained data
                                    input_columns=input_columns, 
                                    target_columns=target_columns, 
                                    epochs=meta_epochs // 2,  # Reduce epochs for faster ensemble training
                                    inner_lr=inner_lr, 
                                    outer_lr=outer_lr, 
                                    num_tasks=num_tasks, 
                                    inner_lr_decay=inner_lr_decay, 
                                    curiosity=curiosity,
                                    min_samples_per_task=3,
                                    early_stopping_patience=5  # Reduce patience for faster ensemble training
                                )
                                
                                # Store the model
                                models_dict["MAML"] = (model, scaler_inputs, scaler_targets)
                            
                            elif model_name == "Reptile":
                                # Initialize and train Reptile model
                                model = ReptileModel(
                                    input_size=len(input_columns),
                                    output_size=len(target_columns),
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate
                                )
                                
                                model, scaler_x, scaler_y = reptile_train(
                                    model, 
                                    data, 
                                    input_columns, 
                                    target_columns, 
                                    reptile_epochs // 2,  # Reduce epochs for faster ensemble training
                                    reptile_learning_rate, 
                                    reptile_num_tasks,
                                    batch_size=batch_size if 'batch_size' in locals() else 16
                                )
                                
                                # Store the model
                                models_dict["Reptile"] = (model, scaler_x, scaler_y)
                            
                            elif model_name == "ProtoNet":
                                # Initialize and train ProtoNet model
                                model = ProtoNetModel(
                                    input_size=len(input_columns),
                                    output_size=len(target_columns),
                                    embedding_size=embedding_size if 'embedding_size' in locals() else 256,
                                    num_layers=num_layers if 'num_layers' in locals() else 3,
                                    dropout_rate=dropout_rate if 'dropout_rate' in locals() else 0.3
                                )
                                
                                model, scaler_x, scaler_y = protonet_train(
                                    model, 
                                    data, 
                                    input_columns, 
                                    target_columns, 
                                    protonet_epochs // 2,  # Reduce epochs for faster ensemble training
                                    protonet_learning_rate, 
                                    protonet_num_tasks, 
                                    num_shot=num_shot if 'num_shot' in locals() else 4, 
                                    num_query=num_query if 'num_query' in locals() else 4
                                )
                                
                                # Store the model
                                models_dict["ProtoNet"] = (model, scaler_x, scaler_y)
                        
                        # Generate model weights based on weighting method
                        model_weights = None
                        if weighting_method == "Equal":
                            model_weights = {k: 1.0 / len(models_dict) for k in models_dict.keys()}
                        # Performance-based and Uncertainty-based are handled within the ensemble function
                        
                        # Run ensemble predictions
                        result_df, ensemble_info = weighted_uncertainty_ensemble(
                            models_dict, 
                            current_data, # Use constrained data
                            input_columns, 
                            target_columns, 
                            acquisition_function=None, 
                            curiosity=curiosity, 
                            weights=model_weights
                        )
                        
                        # Store ensemble info
                        st.session_state["ensemble_info"] = ensemble_info
                    
                    else:
                        # Display a spinner during computation
                        with st.spinner(f"Running {model_type} model..."):
                            if model_type == "MAML":
                                # Initialize the MAML model
                                model = MAMLModel(
                                    input_size=len(input_columns),
                                    output_size=len(target_columns),
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate
                                )
                                
                                # Train the model
                                model, scaler_inputs, scaler_targets = meta_train(
                                    meta_model=model, 
                                    data=current_data, # Use constrained data
                                    input_columns=input_columns, 
                                    target_columns=target_columns, 
                                    epochs=meta_epochs, 
                                    inner_lr=inner_lr, 
                                    outer_lr=outer_lr, 
                                    num_tasks=num_tasks, 
                                    inner_lr_decay=inner_lr_decay, 
                                    curiosity=curiosity,
                                    min_samples_per_task=3,
                                    early_stopping_patience=10
                                )
                                
                                # Evaluate the model
                                result_df = evaluate_maml(
                                    meta_model=model, 
                                    data=current_data, # Use constrained data
                                    input_columns=input_columns, 
                                    target_columns=target_columns, 
                                    curiosity=curiosity, 
                                    weights=weights, 
                                    max_or_min=max_or_min, 
                                    acquisition=None,
                                    dynamic_acquisition=True,
                                    min_labeled_samples=5
                                )

                        
                            elif model_type == "Reptile":
                                # Initialize the Reptile model
                                model = ReptileModel(
                                    input_size=len(input_columns),
                                    output_size=len(target_columns),
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate
                                )
                                
                                # Train the model
                                model, scaler_x, scaler_y = reptile_train(
                                    model, 
                                    current_data, # Use constrained data
                                    input_columns, 
                                    target_columns, 
                                    reptile_epochs, 
                                    reptile_learning_rate, 
                                    reptile_num_tasks,
                                    batch_size=batch_size if 'batch_size' in locals() else 16
                                )
                                
                                # Evaluate the model
                                result_df = evaluate_reptile(
                                    model,
                                    current_data, # Use constrained data
                                    input_columns,
                                    target_columns,
                                    curiosity,
                                    weights,
                                    max_or_min,
                                    acquisition=None,
                                    strict_optimization=strict_optimization
                                )

                            elif model_type == "ProtoNet":
                                # Initialize the ProtoNet model
                                # Update the ProtoNet initialization to match your model
                                model = ProtoNetModel(
                                    input_size=len(input_columns),
                                    output_size=len(target_columns),
                                    embedding_size=embedding_size if 'embedding_size' in locals() else 256,
                                    num_layers=num_layers if 'num_layers' in locals() else 3,
                                    dropout_rate=dropout_rate if 'dropout_rate' in locals() else 0.3
                                )
                                                        
                                # Train the model
                                model, scaler_x, scaler_y = protonet_train(
                                    model, 
                                    current_data, # Use constrained data
                                    input_columns, 
                                    target_columns, 
                                    protonet_epochs, 
                                    protonet_learning_rate, 
                                    protonet_num_tasks, 
                                    num_shot=num_shot, 
                                    num_query=num_query
                                )
                                
                                # Evaluate the model
                                result_df = evaluate_protonet(
                                    model,
                                    current_data, # Use constrained data
                                    input_columns,
                                    target_columns,
                                    curiosity,
                                    weights,
                                    max_or_min,
                                    acquisition=None
                                )
                                if result_df is not None:
                                    st.session_state["result_df"] = result_df
                                else:
                                    st.error("No results generated by ProtoNet.")

                            elif model_type == "Random Forest":
                                # Train the Random Forest model
                                model, scaler_inputs, scaler_targets = train_rf_model(
                                    data=current_data, # Use constrained data
                                    input_columns=input_columns,
                                    target_columns=target_columns,
                                    n_estimators=rf_n_estimators, # from sidebar
                                    perform_grid_search=rf_perform_grid_search, # from sidebar
                                    random_state=42
                                )
                                st.session_state["rf_model"] = model # Store the RF model object
                                st.session_state["rf_scaler_inputs"] = scaler_inputs
                                st.session_state["rf_scaler_targets"] = scaler_targets

                                # Evaluate the model
                                result_df = evaluate_rf_model(
                                    rf_model=model,
                                    data=current_data, # Use constrained data
                                    input_columns=input_columns,
                                    target_columns=target_columns,
                                    curiosity=curiosity, # from sidebar for RF
                                    weights=weights,
                                    max_or_min=max_or_min,
                                    acquisition=None # Let evaluate_rf_model pick or use a default
                                )
                                if result_df is not None:
                                    st.session_state["result_df"] = result_df
                                else:
                                    st.error("No results generated by Random Forest.")

                        
                        # Store model in session state for comparison
                        if "model_history" not in st.session_state:
                            st.session_state["model_history"] = {}
                        
                        timestamp = int(time.time())  # Current Unix timestamp for uniqueness
                        model_id = f"{model_type}_{timestamp}"
                        st.session_state["model_history"][model_id] = {
                            "model": model,
                            "model_type": model_type,
                            "acquisition": None,
                            "curiosity": curiosity
                        }
                        
                        # Store results
                        st.session_state["result_df"] = result_df
                        
                        # Show experiment summary
                        st.success("Experiment completed successfully!")
                        
                        st.markdown("### 🔍 **Experiment Summary:**")
                        
                        # Create summary in columns
                        sum_col1, sum_col2, sum_col3 = st.columns(3)
                        
                        with sum_col1:
                            st.metric("Model Type", model_type)
                            
                        with sum_col2:
                            st.metric("Acquisition Function", None)
                            
                        with sum_col3:
                            st.metric("Curiosity Level", f"{curiosity:.1f}")
                        
                        # Show exploration-exploitation visualization if requested
                        if show_visualizations and result_df is not None:
                            st.subheader("Exploration vs. Exploitation Analysis")
                            visualize_exploration_exploitation(result_df, curiosity)
                        
                        # Show acquisition function analysis if requested
                        if show_acquisition_viz and result_df is not None:
                            st.subheader("Acquisition Function Analysis")
                            fig = create_acquisition_function_visualization(result_df, None, curiosity)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display results table
                        st.subheader("Experiment Results")
                        st.dataframe(result_df, use_container_width=True)

                        # Show suggested sample
                        st.markdown("### 🔬 **Suggested Sample for Lab Testing:**")
                        st.dataframe(result_df.iloc[0:1], use_container_width=True)

                    # Display ensemble info if available
                    if "ensemble_info" in st.session_state and st.session_state["ensemble_info"]:
                        with st.expander("Ensemble Information", expanded=False):
                            # Get ensemble info
                            ensemble_info = st.session_state["ensemble_info"]
                            
                            # Display model weights
                            st.subheader("Model Weights")
                            weight_df = pd.DataFrame({
                                'Model': list(ensemble_info["model_weights"].keys()),
                                'Weight': list(ensemble_info["model_weights"].values())
                            })
                            st.dataframe(weight_df)
                            
                            # Display model contribution visualization
                            st.subheader("Model Contributions")
                            # Create a bar chart of average model contributions
                            model_names = list(ensemble_info["model_contributions"].keys())
                            avg_contributions = [np.mean(contribs) for contribs in ensemble_info["model_contributions"].values()]
                            
                            contrib_df = pd.DataFrame({
                                'Model': model_names,
                                'Average Contribution': avg_contributions
                            })
                            
                            st.bar_chart(contrib_df.set_index('Model'))
    with run_col2:
        if st.button("Reset", key="reset_experiment", use_container_width=True):
            st.session_state["experiment_run"] = False
            st.session_state["plot_option"] = "None"
            st.session_state.pop("result_df", None)
            st.experimental_rerun()
            
        # Add a clear model history button
        if st.button("Clear Model History", key="clear_history", use_container_width=True):
            st.session_state["model_history"] = {} # Note: model_history itself is not part of the saved state yet
            st.success("Model history cleared!")
            st.experimental_rerun()

        # Save Experiment State Button
        if st.button("Save Experiment State", key="save_experiment_state", use_container_width=True):
            if "dataset" in st.session_state and st.session_state.dataset is not None:
                import io
                import zipfile
                import json

                try:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                        # 1. Save dataset
                        csv_buffer = io.StringIO()
                        st.session_state.dataset.to_csv(csv_buffer, index=False)
                        zip_file.writestr("dataset.csv", csv_buffer.getvalue())

                        # 2. Save UI settings
                        ui_settings = {
                            "input_columns": input_columns, # These are from current UI state, not necessarily session_state directly
                            "target_columns": target_columns,
                            "apriori_columns": apriori_columns,
                            "model_type": model_type,
                            "curiosity": curiosity,
                            "constraints": st.session_state.get("constraints"),
                            "sum_constraint_cols": st.session_state.get("sum_constraint_cols"),
                            "sum_constraint_target": st.session_state.get("sum_constraint_target"),
                            "sum_constraint_tolerance": st.session_state.get("sum_constraint_tolerance"),
                            "mobo_strategy": mobo_strategy if len(target_columns) > 1 else "weighted_sum",
                            # Add model-specific hyperparameters by checking model_type
                        }
                        if model_type == "MAML":
                            ui_settings["maml_hidden_size"] = hidden_size
                            ui_settings["maml_num_layers"] = num_layers
                            ui_settings["maml_dropout_rate"] = dropout_rate
                            ui_settings["maml_inner_lr"] = inner_lr
                            ui_settings["maml_inner_lr_decay"] = inner_lr_decay
                            ui_settings["maml_outer_lr"] = outer_lr
                            ui_settings["maml_meta_epochs"] = meta_epochs
                            ui_settings["maml_num_tasks"] = num_tasks
                        elif model_type == "Reptile":
                            ui_settings["reptile_hidden_size"] = hidden_size
                            ui_settings["reptile_num_layers"] = num_layers
                            ui_settings["reptile_dropout_rate"] = dropout_rate
                            ui_settings["reptile_learning_rate"] = reptile_learning_rate
                            ui_settings["reptile_epochs"] = reptile_epochs
                            ui_settings["reptile_num_tasks"] = reptile_num_tasks
                            # ui_settings["reptile_batch_size"] = batch_size # if defined
                        elif model_type == "ProtoNet":
                            ui_settings["protonet_embedding_size"] = embedding_size
                            ui_settings["protonet_num_layers"] = num_layers
                            ui_settings["protonet_dropout_rate"] = dropout_rate
                            ui_settings["protonet_learning_rate"] = protonet_learning_rate
                            ui_settings["protonet_epochs"] = protonet_epochs
                            ui_settings["protonet_num_tasks"] = protonet_num_tasks
                            ui_settings["protonet_num_shot"] = num_shot
                            ui_settings["protonet_num_query"] = num_query
                        elif model_type == "Random Forest":
                            ui_settings["rf_n_estimators"] = rf_n_estimators
                            ui_settings["rf_perform_grid_search"] = rf_perform_grid_search

                        zip_file.writestr("ui_settings.json", json.dumps(ui_settings, indent=4))

                        # 3. Save last result_df
                        if "result_df" in st.session_state and st.session_state.result_df is not None:
                            csv_buffer_results = io.StringIO()
                            st.session_state.result_df.to_csv(csv_buffer_results, index=False)
                            zip_file.writestr("results.csv", csv_buffer_results.getvalue())

                        # TODO: Add model state saving here later (model weights, scalers)

                    zip_buffer.seek(0)
                    st.download_button(
                        label="Download Experiment State (ZIP)",
                        data=zip_buffer,
                        file_name="metadesign_experiment_state.zip",
                        mime="application/zip"
                    )
                    st.success("Experiment state prepared for download.")
                except Exception as e:
                    st.error(f"Error saving experiment state: {e}")
            else:
                st.warning("No dataset loaded to save state.")


    # Show visualization based on selection
    if st.session_state["experiment_run"] and "result_df" in st.session_state:  
        result_df = st.session_state["result_df"]
        
        if plot_option != "None":
            st.subheader(f"{plot_option} Visualization")
            
            if plot_option == "Scatter Matrix":
                
                if result_df is not None:
                    # Ensure Row number is present
                    if "Row number" not in result_df.columns:
                        result_df.insert(0, "Row number", range(1, len(result_df) + 1))

                    # Select numeric and non-NaN target columns only
                    valid_dimensions = [
                        col for col in target_columns
                        if col in result_df.columns and pd.api.types.is_numeric_dtype(result_df[col])
                    ]

                    if len(valid_dimensions) == 1:
                        target_name = valid_dimensions[0]
                        result_df = result_df.sort_values(by=target_name).reset_index(drop=True)

                        # ✅ Use raw utility for visualization
                        raw_utility = calculate_utility(
                            predictions=result_df[target_columns].values,
                            uncertainties=np.stack([
                                result_df.get(f"Uncertainty ({col})", pd.Series(np.zeros(len(result_df))))
                                for col in target_columns
                            ], axis=1),
                            novelty=result_df["Novelty"].values if "Novelty" in result_df else None,
                            curiosity=curiosity,
                            weights=weights_targets,
                            max_or_min=max_or_min_targets,
                            thresholds=thresholds_targets if thresholds_targets else None,
                            acquisition="UCB",  # Or dynamic if you're selecting
                            for_visualization=True
                        )
                        result_df["Raw Utility"] = raw_utility

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=result_df[target_name],
                            y=result_df["Raw Utility"],
                            mode='markers',
                            marker=dict(
                                size=7,
                                color=result_df["Raw Utility"],
                                colorscale="Plasma",
                                colorbar=dict(title="Utility"),
                            ),
                            customdata=result_df["Row number"],
                            error_x=dict(
                                type='data',
                                array=_select_error_col_if_available(result_df, target_name),
                                color='lightgray',
                                thickness=1,
                            ),
                            hovertemplate="Row number: %{customdata}, X: %{x:.2f}, Y: %{y:.2f}, Utility: %{marker.color:.2f}",
                            hoverlabel=dict(bgcolor="black"),
                            name=''
                        ))
                        fig.update_layout(
                            title="Scatter plot of target properties",
                            height=1000
                        )
                        fig.update_xaxes(title_text=target_name)
                        fig.update_yaxes(title_text="Utility")
                        st.plotly_chart(fig, use_container_width=True)



            
            elif plot_option == "t-SNE":
                fig = create_tsne_plot_with_hover(result_df, input_columns, "Utility")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data for t-SNE visualization.")
            
            elif plot_option == "Parallel Coordinates":
                fig = create_parallel_coordinates(result_df, target_columns)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid target properties selected for Parallel Coordinates.")
            
            elif plot_option == "3D Scatter":
                if len(target_columns) >= 3:
                    # Let user select which properties to visualize in 3D
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_prop = st.selectbox("X-axis property:", target_columns, index=0)
                    with col2:
                        y_prop = st.selectbox("Y-axis property:", target_columns, index=min(1, len(target_columns)-1))
                    with col3:
                        z_prop = st.selectbox("Z-axis property:", target_columns, index=min(2, len(target_columns)-1))
                    
                    color_by = st.radio(
                        "Color by:", 
                        ["Utility", "Uncertainty", "Novelty"] + target_columns,
                        horizontal=True
                    )
                    
                    fig = create_3d_scatter(result_df, x_prop, y_prop, z_prop, color_by=color_by)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("3D Scatter requires at least 3 target properties. Please select more target properties.")
            
            elif plot_option == "Pareto Front":
                if len(target_columns) >= 2:
                    # Let user select which objectives to visualize
                    col1, col2 = st.columns(2)
                    with col1:
                        obj1 = st.selectbox("First objective:", target_columns, index=0)
                    with col2:
                        obj2 = st.selectbox("Second objective:", target_columns, index=min(1, len(target_columns)-1))
                    
                    # Get max_or_min directions for selected objectives
                    obj1_idx = target_columns.index(obj1)
                    obj2_idx = target_columns.index(obj2)
                    obj_directions = [max_or_min_targets[obj1_idx], max_or_min_targets[obj2_idx]]
                    
                    fig = create_pareto_front_visualization(
                        result_df, 
                        [obj1, obj2], 
                        obj_directions
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Pareto front visualization requires at least 2 target properties.")
            
            elif plot_option == "Property Distributions":
                fig = visualize_property_distributions(result_df, target_columns)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid target properties for distribution visualization.")
            
            elif plot_option == "Acquisition Function Analysis":
                fig = create_acquisition_function_visualization(result_df, None, curiosity)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show exploration-exploitation tradeoff with different curiosity values
                st.subheader("Exploration-Exploitation Tradeoff")
                st.markdown("""
                    This visualization shows how different curiosity values would affect the balance
                    between exploration and exploitation for the selected sample.
                """)
                
                fig2 = visualize_exploration_exploitation_tradeoff(result_df)
                st.plotly_chart(fig2, use_container_width=True)
                
            elif plot_option == "Model Comparison" and len(st.session_state.get("model_history", {})) > 1:
                st.subheader("Model Comparison")
                
                # Let user select which models to compare
                models_to_compare = st.multiselect(
                    "Select models to compare:",
                    options=list(st.session_state["model_history"].keys()),
                    default=list(st.session_state["model_history"].keys())
                )
                
                if models_to_compare:
                    # Prepare models dictionary for comparison
                    models_dict = {
                        model_id: st.session_state["model_history"][model_id]["model"] 
                        for model_id in models_to_compare
                    }
                    
                    # Select metric for comparison
                    metric = st.selectbox(
                        "Select comparison metric:",
                        ["MSE", "MAE", "R2", "all"],
                        index=0
                    )
                    
                    # Create model comparison visualization
                    fig = visualize_model_comparison(
                        models_dict, 
                        current_data, # Use constrained data for comparison
                        input_columns, 
                        target_columns, 
                        metric=metric
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one model for comparison.")

        # Add additional sections for advanced analysis
        if st.session_state["experiment_run"] and "result_df" in st.session_state:
            with st.expander("Advanced Analysis", expanded=False):
                st.subheader("Optimal Regions Analysis")
                
                if len(target_columns) >= 2:
                    # Let user select which properties to analyze
                    col1, col2 = st.columns(2)
                    with col1:
                        prop1 = st.selectbox("First property:", input_columns, index=0, key="opt_prop1")
                    with col2:
                        prop2 = st.selectbox("Second property:", input_columns, index=min(1, len(input_columns)-1), key="opt_prop2")
                    
                    # Percentile threshold for highlighting
                    percentile = st.slider("Percentile threshold:", 50, 95, 75, 5)
                    
                    # Create optimal regions visualization
                    highlight_df = pd.concat([result_df[target_columns], result_df[[prop1, prop2]]], axis=1)
                    
                    fig = highlight_optimal_regions(
                        highlight_df, 
                        target_columns, 
                        max_or_min_targets,
                        percentile=percentile
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Optimal regions analysis requires at least 2 input features and 2 target properties.")

            # Add experiment history section
            with st.expander("Experiment History", expanded=False):
                if "model_history" in st.session_state and st.session_state["model_history"]:
                    st.subheader("Previous Experiments")
                    
                    # Create a dataframe of experiment history
                    history_data = []
                    for model_id, model_info in st.session_state["model_history"].items():
                        history_data.append({
                            "Model ID": model_id,
                            "Model Type": model_info["model_type"],
                            "Acquisition": model_info.get("acquisition", "N/A"),
                            "Curiosity": model_info.get("curiosity", "N/A")
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    st.dataframe(history_df, use_container_width=True)
                else:
                    st.info("No previous experiments recorded.")


# Add footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>MetaDesign Dashboard - Advanced Meta-Learning for Materials Discovery</p>
    </div>
""", unsafe_allow_html=True)