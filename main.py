
import streamlit as st
st.set_page_config(page_title="MetaDesign Dashboard", layout="wide")
import os
import json
import time
import numpy as np
import pandas as pd
from plotly import graph_objects as go
import joblib # Added for saving/loading sklearn models and scalers

import torch
import plotly.express as px
from sklearn.preprocessing import StandardScaler, RobustScaler
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode, JsCode

# Import model implementations
from app.models import MAMLModel, meta_train, evaluate_maml
from app.reptile_model import ReptileModel, reptile_train, evaluate_reptile
from app.protonet_model import ProtoNetModel, protonet_train, evaluate_protonet
from app.rf_model import RFModel, train_rf_model, evaluate_rf_model
from app.pinn_model import PINNModel, pinn_train, evaluate_pinn

# Import utility functions
from app.utils import (
    calculate_utility, calculate_novelty, # calculate_uncertainty was removed
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

# Process loaded UI settings if available (after a state file upload and rerun)
if "loaded_ui_settings" in st.session_state and st.session_state.loaded_ui_settings is not None:
    loaded_settings = st.session_state.loaded_ui_settings.copy() # Work with a copy
    st.info("Applying loaded UI settings...")

    # These keys will be used to set defaults for column selection multiselects later
    st.session_state.input_columns_loaded_from_file = loaded_settings.get("input_columns", [])
    st.session_state.target_columns_loaded_from_file = loaded_settings.get("target_columns", [])
    st.session_state.apriori_columns_loaded_from_file = loaded_settings.get("apriori_columns", [])

    # Model type (widget key is 'model_type_selector' - defined later by the selectbox itself)
    # So, we set the value that the selectbox will use for its 'index' or initial state.
    # The selectbox will be: model_type = st.sidebar.selectbox(..., key="model_type_selector")
    # We need to ensure the widget is defined such that it can pick this up or we set its state.
    # For selectbox, it's often easier to set its default list variable.
    # Let's assume the selectbox for model_type will use st.session_state.model_type_selector if it exists.
    st.session_state.model_type_selector_loaded_value = loaded_settings.get("model_type", "MAML")


    # Constraints - these session_state keys are directly used by the constraint UI
    st.session_state.constraints = loaded_settings.get("constraints", {})
    st.session_state.sum_constraint_cols = loaded_settings.get("sum_constraint_cols", [])
    st.session_state.sum_constraint_target = loaded_settings.get("sum_constraint_target", 1.0)
    st.session_state.sum_constraint_tolerance = loaded_settings.get("sum_constraint_tolerance", 0.01)

    # MOBO Strategy (widget key is 'mobo_strategy_selector')
    st.session_state.mobo_strategy_selector_loaded_value = loaded_settings.get("mobo_strategy", "weighted_sum")

    # Model Hyperparameters & Curiosity (using their widget keys directly)
    # This relies on the keys in ui_settings.json matching the keys used in st.slider/st.checkbox etc.
    # Generic curiosity is saved as "curiosity". Model specific sliders use keys like "maml_curiosity", "rf_curiosity" etc.
    # We also save model specific params like "maml_hidden_size", "rf_n_estimators"

    # Store all loaded settings that are not column selections or complex ones like curiosity
    # These will be picked up by widgets if their keys match.
    for key, value in loaded_settings.items():
        if key not in ["input_columns", "target_columns", "apriori_columns", "model_type", "mobo_strategy", "constraints",
                       "sum_constraint_cols", "sum_constraint_target", "sum_constraint_tolerance", "curiosity"]:
            st.session_state[key] = value

    # Handle curiosity separately as its key is dynamic in the UI
    # The actual curiosity value used by the app will be set when the specific model's UI is rendered
    st.session_state.curiosity_loaded_value = loaded_settings.get("curiosity", 0.0)

    # Load Model State (after UI settings like model_type are known)
    # This requires the ZIP file to be accessible here, which it isn't directly after a rerun.
    # The model loading needs to happen INSIDE the `if uploaded_state_file is not None:` block,
    # before the rerun, using the zip_ref. This was a flaw in my previous reasoning.

    # Let's correct the flow: Model loading must happen within the ZIP processing block.
    # The `loaded_settings` dictionary is ALREADY available there if `ui_settings.json` was read.
    # The `st.experimental_rerun()` should happen AFTER ALL loading (dataset, results, ui_settings, model).

    # Clear the main loaded_settings dict and the specific 'loaded_value' flags
    # to prevent them from interfering with subsequent user interactions or reruns.
    del st.session_state.loaded_ui_settings
    # Specific loaded flags will be cleared by widgets themselves or we can clear them here if widgets don't automatically.
    # For now, relying on widgets to use these values for their current run and then operate normally.
    # If a widget's default is continuously overridden, then explicit deletion of the specific _loaded_value key is needed
    # after the widget definition. Given the current setup, most widgets should be okay.
    # Let's explicitly clear the ones we created for selectbox defaults to be safe.
    if "model_type_selector_loaded_value" in st.session_state:
        del st.session_state.model_type_selector_loaded_value
    if "mobo_strategy_selector_loaded_value" in st.session_state:
        del st.session_state.mobo_strategy_selector_loaded_value
    # input/target/apriori columns loaded_from_file are cleared after their multiselects are defined.
    # curiosity_loaded_value will be used by sliders and then normal session_state takes over for the slider's key.

    st.info("Loaded UI settings have been applied.")


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

            ui_settings_loaded_for_model = None
            if "ui_settings.json" in zip_ref.namelist():
                with zip_ref.open("ui_settings.json") as settings_file:
                    ui_settings_loaded_for_model = json.load(settings_file) # Load into a local var first
                    st.session_state.loaded_ui_settings = ui_settings_loaded_for_model # Then into session_state for rerun UI population
                st.success("UI settings parsed from state file.")

            # Load Model State (e.g., Random Forest)
            if ui_settings_loaded_for_model: # Check if we have settings to know model type
                model_type_to_load = ui_settings_loaded_for_model.get("model_type")
                if model_type_to_load == "Random Forest":
                    try:
                        if "model_state/rf_model.joblib" in zip_ref.namelist() and \
                           "model_state/rf_scaler_x.joblib" in zip_ref.namelist() and \
                           "model_state/rf_scaler_y.joblib" in zip_ref.namelist():

                            with zip_ref.open("model_state/rf_model.joblib") as mf:
                                loaded_rf_sklearn_model = joblib.load(mf)
                            with zip_ref.open("model_state/rf_scaler_x.joblib") as sf_x:
                                loaded_scaler_x = joblib.load(sf_x)
                            with zip_ref.open("model_state/rf_scaler_y.joblib") as sf_y:
                                loaded_scaler_y = joblib.load(sf_y)

                            from app.rf_model import RFModel # Ensure import
                            reconstructed_rf_model = RFModel() # Use default params from class
                            # Apply loaded hyperparameters if they were saved and are part of ui_settings_loaded_for_model
                            reconstructed_rf_model.model.set_params(
                                n_estimators=ui_settings_loaded_for_model.get("rf_n_estimators", reconstructed_rf_model.model.get_params()["n_estimators"])
                                # Add other relevant RF params here if they were saved in ui_settings
                            )
                            reconstructed_rf_model.model = loaded_rf_sklearn_model # The actual fitted model
                            reconstructed_rf_model.scaler_x = loaded_scaler_x
                            reconstructed_rf_model.scaler_y = loaded_scaler_y
                            reconstructed_rf_model.is_trained = True

                            st.session_state.rf_model = reconstructed_rf_model
                            st.session_state.model = reconstructed_rf_model # Generic holder
                            # These might not be strictly necessary if rf_model holds them
                            st.session_state.rf_scaler_inputs = loaded_scaler_x
                            st.session_state.rf_scaler_targets = loaded_scaler_y
                            st.success("Random Forest model state loaded successfully.")
                        else:
                            st.warning("Random Forest model state files not found in ZIP, though model type was RF.")
                    except Exception as e:
                        st.error(f"Error loading Random Forest model state: {e}")

                elif model_type_to_load in ["MAML", "Reptile", "ProtoNet"]:
                    try:
                        model_file = f"model_state/{model_type_to_load.lower()}_model_statedict.pt"
                        scaler_x_file = f"model_state/{model_type_to_load.lower()}_scaler_x.joblib"
                        scaler_y_file = f"model_state/{model_type_to_load.lower()}_scaler_y.joblib"

                        if model_file in zip_ref.namelist() and \
                           scaler_x_file in zip_ref.namelist() and \
                           scaler_y_file in zip_ref.namelist():

                            # Determine input_size and output_size (crucial for model instantiation)
                            # These should be available from the loaded dataset and selected columns in ui_settings
                            loaded_input_cols = ui_settings_loaded_for_model.get("input_columns", [])
                            loaded_target_cols = ui_settings_loaded_for_model.get("target_columns", [])

                            if not loaded_input_cols or not loaded_target_cols:
                                st.error(f"Cannot load {model_type_to_load} model: input/target column information missing in saved settings.")
                                raise ValueError("Missing column info for model instantiation.")

                            input_size = len(loaded_input_cols)
                            output_size = len(loaded_target_cols)

                            # Instantiate model based on type and loaded hyperparameters
                            if model_type_to_load == "MAML":
                                model_instance = MAMLModel(
                                    input_size=input_size, output_size=output_size,
                                    hidden_size=ui_settings_loaded_for_model.get("maml_hidden_size", defaults["hidden_size"]),
                                    num_layers=ui_settings_loaded_for_model.get("maml_num_layers", 3),
                                    dropout_rate=ui_settings_loaded_for_model.get("maml_dropout_rate", 0.3)
                                )
                            elif model_type_to_load == "Reptile":
                                model_instance = ReptileModel(
                                    input_size=input_size, output_size=output_size,
                                    hidden_size=ui_settings_loaded_for_model.get("reptile_hidden_size", defaults["hidden_size"]),
                                    num_layers=ui_settings_loaded_for_model.get("reptile_num_layers", 3),
                                    dropout_rate=ui_settings_loaded_for_model.get("reptile_dropout_rate", 0.3)
                                )
                            elif model_type_to_load == "ProtoNet":
                                model_instance = ProtoNetModel(
                                    input_size=input_size, output_size=output_size,
                                    embedding_size=ui_settings_loaded_for_model.get("protonet_embedding_size", 256),
                                    num_layers=ui_settings_loaded_for_model.get("protonet_num_layers", 3),
                                    dropout_rate=ui_settings_loaded_for_model.get("protonet_dropout_rate", 0.3)
                                )
                            else: # Should not happen due to outer if
                                raise ValueError(f"Unknown PyTorch model type for loading: {model_type_to_load}")

                            with zip_ref.open(model_file) as mf:
                                model_instance.load_state_dict(torch.load(mf))
                            with zip_ref.open(scaler_x_file) as sf_x:
                                loaded_scaler_x = joblib.load(sf_x)
                            with zip_ref.open(scaler_y_file) as sf_y:
                                loaded_scaler_y = joblib.load(sf_y)

                            model_instance.eval() # Set to eval mode

                            st.session_state.model = model_instance
                            st.session_state.scaler_inputs = loaded_scaler_x
                            st.session_state.scaler_targets = loaded_scaler_y
                            st.success(f"{model_type_to_load} model state loaded successfully.")
                        else:
                            st.warning(f"{model_type_to_load} model state files not found in ZIP.")
                    except Exception as e:
                        st.error(f"Error loading {model_type_to_load} model state: {e}")

            if "loaded_ui_settings" in st.session_state: # If UI settings were loaded (implies model type is known)
                st.experimental_rerun()

        # This message might appear before rerun fully updates UI
        st.sidebar.success("Experiment state files processed. UI should update on rerun if settings were loaded.")
        # Clear the uploader to allow re-upload of same filename if needed, after processing
        # This can sometimes be tricky with Streamlit's default file_uploader behavior.
        # A common workaround is to use a button to trigger processing and then clear via a callback or rerun.
        # For now, we'll rely on user uploading a new file if they want to change.

    except Exception as e:
        st.sidebar.error(f"Error loading experiment state: {e}")

st.sidebar.header("Model & Data Configuration")
# Sidebar: Model Selection with improved descriptions
# Determine initial index for model_type selectbox if loaded from state
model_options = ["MAML", "Reptile", "ProtoNet", "Random Forest", "PINN"]
default_model_type = "MAML"
if "model_type_selector_loaded_value" in st.session_state:
    default_model_type = st.session_state.model_type_selector_loaded_value
    # Clear it after use to prevent it from always overriding user selection on normal reruns
    # del st.session_state.model_type_selector_loaded_value # Let's not delete, subsequent widgets might need it as context
model_idx = model_options.index(default_model_type) if default_model_type in model_options else 0

model_type = st.sidebar.selectbox(
    "Choose Model Type:", 
    options=model_options,
    index=model_idx,
    key="model_type_selector", # Ensure a key is set
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
        hidden_size = st.slider("Hidden Size:", 64, 512, defaults["hidden_size"], 32, help="Number of units in hidden layers.")
        num_layers = st.slider("Number of Layers:", 2, 5, 3, 1, help="Number of hidden layers in the network.")
        dropout_rate = st.slider("Dropout Rate:", 0.0, 0.5, 0.2, 0.05, help="Dropout probability for regularization during training.")
    
    with st.sidebar.expander("Meta-Learning Parameters", expanded=True):
        use_adaptive_inner_lr = st.checkbox("Enable Adaptive Inner LR", value=True, help="Automatically adjust inner loop learning rate based on heuristics.")
        inner_lr_help_text = "Learning rate for task-specific adaptation (inner loop). Ignored if adaptive LR is enabled."
        inner_lr = 0.0001 if use_adaptive_inner_lr else st.slider("Inner Loop Learning Rate:", 0.00005, 0.01, defaults["inner_lr"], 0.0001, help=inner_lr_help_text)
        
        inner_lr_decay_help_text = "Decay rate for inner loop learning rate per epoch. Ignored if adaptive LR is enabled."
        inner_lr_decay = 0.95 if use_adaptive_inner_lr else st.slider("Inner LR Decay Rate:", 0.8, 1.00, 0.98, 0.01, help=inner_lr_decay_help_text)
        
        use_adaptive_outer_lr = st.checkbox("Enable Adaptive Outer LR", value=True, help="Automatically adjust outer loop learning rate based on heuristics.")
        outer_lr_help_text = "Learning rate for meta-model updates (outer loop). Ignored if adaptive LR is enabled."
        outer_lr = 0.00002 if use_adaptive_outer_lr else st.slider("Outer Loop Learning Rate:", 0.00001, 0.01, defaults["outer_lr"], 0.00001, help=outer_lr_help_text)
        
        use_adaptive_epochs = st.checkbox("Enable Adaptive Training Length", value=True, help="Automatically determine number of meta-training epochs based on heuristics.")
        meta_epochs_help_text = "Number of epochs for meta-training. Ignored if adaptive training length is enabled."
        meta_epochs = 100 if use_adaptive_epochs else st.slider("Meta-Training Epochs:", 10, 300, defaults["meta_epochs"], 10, help=meta_epochs_help_text)

        num_tasks = st.slider("Number of Tasks:", 2, 10, defaults["num_tasks"], 1, help="Number of tasks to sample per meta-training epoch.")
    
    default_curiosity = defaults["curiosity"]
    if "curiosity_loaded_value" in st.session_state:
        default_curiosity = st.session_state.curiosity_loaded_value

    curiosity_help = "Adjusts the exploration-exploitation balance. Higher values favor exploration of novel/uncertain regions, lower values favor exploitation of known high-performing regions."
    curiosity = st.slider("Exploit (-2) vs Explore (+2)", -2.0, 2.0, default_curiosity, 0.1, key="maml_curiosity", help=curiosity_help)
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
        hidden_size = st.slider("Hidden Size:", 64, 512, defaults["hidden_size"], 32, help="Number of units in hidden layers for Reptile.")
        num_layers = st.slider("Number of Layers:", 2, 5, 3, 1, help="Number of hidden layers in the Reptile network.")
        dropout_rate = st.slider("Dropout Rate:", 0.0, 0.5, 0.3, 0.05, help="Dropout probability for Reptile.")
    
    with st.sidebar.expander("Training Parameters", expanded=True):
        use_adaptive_reptile_lr = st.checkbox("Enable Adaptive Learning Rate", value=True, help="Automatically adjust Reptile learning rate.")
        reptile_lr_help = "Learning rate for Reptile model updates. Ignored if adaptive LR is enabled."
        reptile_learning_rate = 0.005 if use_adaptive_reptile_lr else st.slider("Learning Rate:", 0.0001, 0.1, defaults["reptile_learning_rate"], 0.001, help=reptile_lr_help)
        
        use_adaptive_batch = st.checkbox("Enable Adaptive Batch Size", value=True, help="Automatically adjust batch size for Reptile.")
        batch_size_help = "Number of samples per gradient update. Ignored if adaptive batch size is enabled."
        batch_size = 16 if use_adaptive_batch else st.slider("Batch Size:", 4, 128, 16, 4, help=batch_size_help)
        
        reptile_epochs = st.slider("Training Epochs:", 10, 300, defaults["reptile_epochs"], 10, help="Number of epochs for Reptile training.")
        reptile_num_tasks = st.slider("Number of Tasks:", 2, 10, defaults["reptile_num_tasks"], 1, help="Number of tasks to sample per Reptile training epoch.")
        
        default_curiosity = defaults["curiosity"]
        if "curiosity_loaded_value" in st.session_state:
            default_curiosity = st.session_state.curiosity_loaded_value
        curiosity_help = "Adjusts the exploration-exploitation balance for Reptile."
        curiosity = st.slider("Exploit (-2) vs Explore (+2)", -2.0, 2.0, default_curiosity, 0.1, key="reptile_curiosity", help=curiosity_help)
        strict_optimization = st.checkbox("Strict Optimization Direction", value=True, help="If checked, ensures Reptile strictly follows the optimization direction (max/min) for the primary target, potentially overriding utility for highest/lowest predicted values.")
        
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
        embedding_size = st.slider("Embedding Size:", 64, 512, 128, 32, help="Size of the embedding space for ProtoNet.")
        num_layers = st.slider("Number of Layers (Encoder):", 2, 5, 3, 1, help="Number of layers in the ProtoNet encoder.")
        dropout_rate = st.slider("Dropout Rate (Encoder):", 0.0, 0.5, 0.3, 0.05, help="Dropout probability for ProtoNet's encoder.")
    
    with st.sidebar.expander("Training Parameters", expanded=True):
        if "protonet_learning_rate" not in st.session_state: # Retain session state for this specific slider
            st.session_state["protonet_learning_rate"] = defaults["protonet_learning_rate"]
        
        protonet_learning_rate = st.slider("Learning Rate:", 0.0001, 0.01, st.session_state["protonet_learning_rate"], 0.0001, help="Learning rate for ProtoNet training.")
        st.session_state["protonet_learning_rate"] = protonet_learning_rate # Update session state
        
        protonet_epochs = st.slider("Training Epochs:", 10, 300, defaults["protonet_epochs"], 10, help="Number of epochs for ProtoNet training.")
        protonet_num_tasks = st.slider("Number of Tasks (Episodes):", 2, 10, defaults["protonet_num_tasks"], 1, help="Number of episodes per ProtoNet training epoch.")
        num_shot = st.slider("Support Samples (N-shot):", 1, 8, 4, 1, help="Number of support samples per class in each episode.")
        num_query = st.slider("Query Samples:", 1, 8, 4, 1, help="Number of query samples per class in each episode.")

    default_curiosity = defaults["curiosity"]
    if "curiosity_loaded_value" in st.session_state:
        default_curiosity = st.session_state.curiosity_loaded_value
    curiosity_help = "Adjusts the exploration-exploitation balance for ProtoNet."
    curiosity = st.slider("Exploit (-2) vs Explore (+2)", -2.0, 2.0, default_curiosity, 0.1, key="protonet_curiosity", help=curiosity_help)
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
        rf_n_estimators = st.slider("Number of Estimators (Trees):", 50, 500, 100, 10, key="rf_n_estimators", help="The number of trees in the forest.")
        # max_depth, min_samples_split, min_samples_leaf could be added if more control is needed
        # For now, keeping it simple.
        rf_perform_grid_search = st.checkbox("Perform GridSearchCV (slower)", value=False, key="rf_grid_search", help="If checked, performs a grid search to find optimal hyperparameters for Random Forest. This can be significantly slower.")

    default_curiosity = defaults["curiosity"]
    if "curiosity_loaded_value" in st.session_state:
        default_curiosity = st.session_state.curiosity_loaded_value
    curiosity_help = "Adjusts the exploration-exploitation balance for Random Forest."
    curiosity = st.slider("Exploit (-2) vs Explore (+2)", -2.0, 2.0, default_curiosity, 0.1, key="rf_curiosity", help=curiosity_help)
    if curiosity < -1.0:
        st.info("Current strategy: Strong exploitation - focusing on known good regions")
    elif curiosity < 0:
        st.info("Current strategy: Moderate exploitation - slight preference for known regions")
    elif curiosity < 1.0:
        st.info("Current strategy: Balanced approach - considering both known and novel regions")
    else:
        st.info("Current strategy: Strong exploration - actively seeking novel regions")

elif model_type == "PINN":
    st.sidebar.subheader("PINN Model Configuration")

    with st.sidebar.expander("Network Architecture", expanded=False):
        hidden_size = st.slider("Hidden Size:", 64, 512, defaults["hidden_size"], 32, help="Number of units in hidden layers for PINN.")
        num_layers = st.slider("Number of Layers:", 2, 5, 3, 1, help="Number of hidden layers in the PINN network.")
        dropout_rate = st.slider("Dropout Rate:", 0.0, 0.5, 0.3, 0.05, help="Dropout probability for PINN.")

    with st.sidebar.expander("Training Parameters", expanded=True):
        pinn_learning_rate = st.slider("Learning Rate:", 0.0001, 0.1, 0.001, 0.001, help="Learning rate for PINN model updates.")
        pinn_epochs = st.slider("Training Epochs:", 10, 300, 50, 10, help="Number of epochs for PINN training.")
        pinn_batch_size = st.slider("Batch Size:", 4, 128, 16, 4, help="Number of samples per gradient update.")
        physics_loss_weight = st.slider("Physics Loss Weight:", 0.0, 1.0, 0.1, 0.01, help="Weight for the physics-informed loss term.")

        default_curiosity = defaults["curiosity"]
        if "curiosity_loaded_value" in st.session_state:
            default_curiosity = st.session_state.curiosity_loaded_value
        curiosity_help = "Adjusts the exploration-exploitation balance for PINN."
        curiosity = st.slider("Exploit (-2) vs Explore (+2)", -2.0, 2.0, default_curiosity, 0.1, key="pinn_curiosity", help=curiosity_help)

        if curiosity < -1.0:
            st.info("Current strategy: Strong exploitation - focusing on known good regions")
        elif curiosity < 0:
            st.info("Current strategy: Moderate exploitation - slight preference for known regions")
        elif curiosity < 1.0:
            st.info("Current strategy: Balanced approach - considering both known and novel regions")
        else:
            st.info("Current strategy: Strong exploration - actively seeking novel regions")

# Optimization Strategy Section
if model_type != "PINN":
    st.sidebar.subheader("Optimization Strategy")
    use_bayesian_optimizer_for_suggestion = st.sidebar.checkbox(
        "Use Bayesian Optimizer for Next Suggestion",
        value=False,
        help="If checked, the selected trained model (e.g., MAML, RF) will be used as a surrogate within a Bayesian Optimization loop. The BO will then use its own acquisition function (selected below) to score and rank candidate materials. If unchecked, the ranking is based on the model's direct utility/uncertainty estimates combined with the main curiosity slider."
    )
    acquisition_function_bo = "UCB"
    if use_bayesian_optimizer_for_suggestion:
        bo_acq_help = """
        Select the acquisition function for the Bayesian Optimizer:
        - UCB (Upper Confidence Bound): Balances exploration and exploitation. Good default.
        - EI (Expected Improvement): Focuses on improving over the best observed point. Good for exploitation.
        - PI (Probability of Improvement): Similar to EI, but focuses on probability rather than magnitude of improvement.
        """
        acquisition_function_bo = st.sidebar.selectbox(
            "BO Acquisition Function:",
            options=["UCB", "EI", "PI"],
            index=0,
            format_func=lambda x: {"UCB": "Upper Confidence Bound", "EI": "Expected Improvement", "PI": "Probability of Improvement"}.get(x, x),
            help=bo_acq_help
        )
else:
    use_bayesian_optimizer_for_suggestion = False


# Add ensemble option after model selection
if model_type != "PINN":
    st.subheader("Ensemble Settings")
    use_ensemble = st.checkbox(
        "Use Model Ensemble",
        value=False,
        help="Combines predictions from multiple models (MAML, Reptile, ProtoNet) for potentially more robust and accurate suggestions. This will take longer as each selected model needs to be trained."
    )

    if use_ensemble:
        ensemble_col1, ensemble_col2 = st.columns(2)

        with ensemble_col1:
            ensemble_models = st.multiselect(
                "Models to Include in Ensemble:",
                options=["MAML", "Reptile", "ProtoNet"],
                default=[model_type] if model_type in ["MAML", "Reptile", "ProtoNet"] else ["MAML"], # Start with currently selected model if compatible
                help="Select the meta-learning models to include in the ensemble. Random Forest is not available for ensembling in the current setup."
            )

        with ensemble_col2:
            weighting_help = """
            Method to combine predictions from ensemble members:
            - Equal: All models contribute equally.
            - Performance-based: Models are weighted based on their performance on validation data (if available).
            - Uncertainty-based: Models that are more confident (lower uncertainty) are given higher weight. (Note: May require specific uncertainty calibration).
            """
            weighting_method = st.selectbox(
                "Ensemble Weighting Method:",
                options=["Equal", "Performance-based", "Uncertainty-based"],
                index=1, # Default to Performance-based
                help=weighting_help
            )
else:
    use_ensemble = False



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
if data is not None:
    # Define feature selection and targets
    # Data columns selection with improved UI
    col1, col2, col3 = st.columns(3)

    # Initialize loaded column selections in session_state if not already there (e.g., on first run without loading)
    if "input_columns_loaded_from_file" not in st.session_state:
        st.session_state.input_columns_loaded_from_file = []
    if "target_columns_loaded_from_file" not in st.session_state:
        st.session_state.target_columns_loaded_from_file = []
    if "apriori_columns_loaded_from_file" not in st.session_state:
        st.session_state.apriori_columns_loaded_from_file = []
    
    with col1:
        # Ensure defaults are valid options from the current dataset
        valid_default_input_cols = [col for col in st.session_state.input_columns_loaded_from_file if col in data.columns]
        input_columns = st.multiselect(
            "Input Features:", 
            options=data.columns.tolist(),
            default=valid_default_input_cols, # Use loaded value
            key="input_columns_multiselect",
            help="Select the material composition variables"
        )
    
    with col2:
        remaining_cols_for_target = [col for col in data.columns if col not in input_columns]
        valid_default_target_cols = [col for col in st.session_state.target_columns_loaded_from_file if col in remaining_cols_for_target]
        target_columns = st.multiselect(
            "Target Properties:", 
            options=remaining_cols_for_target,
            default=valid_default_target_cols, # Use loaded value
            key="target_columns_multiselect",
            help="Select the material properties you want to optimize"
        )
    
    with col3:
        remaining_cols_for_apriori = [col for col in data.columns if col not in input_columns + target_columns]
        valid_default_apriori_cols = [col for col in st.session_state.apriori_columns_loaded_from_file if col in remaining_cols_for_apriori]
        apriori_columns = st.multiselect(
            "A Priori Properties:", 
            options=remaining_cols_for_apriori,
            default=valid_default_apriori_cols, # Use loaded value
            key="apriori_columns_multiselect",
            help="Select properties with known values that constrain the optimization"
        )

    # Clear loaded column selections after first use to allow user to change them without being overridden on next rerun
    if "input_columns_loaded_from_file" in st.session_state and st.session_state.input_columns_loaded_from_file:
        st.session_state.input_columns_loaded_from_file = [] # Clear after use
    if "target_columns_loaded_from_file" in st.session_state and st.session_state.target_columns_loaded_from_file:
        st.session_state.target_columns_loaded_from_file = []
    if "apriori_columns_loaded_from_file" in st.session_state and st.session_state.apriori_columns_loaded_from_file:
        st.session_state.apriori_columns_loaded_from_file = []


    # ADD FEATURE SELECTION CODE HERE - after user selections but before model initialization
    if len(input_columns) > 0 and len(target_columns) > 0:
        st.subheader("Feature Selection")
        use_feature_selection = st.checkbox(
            "Use Automated Feature Selection",
            value=False,
            help="Automatically select a subset of the most important input features using statistical methods and correlation analysis. This can improve model performance and interpretability, especially with high-dimensional data."
        )
        
        if use_feature_selection:
            from app.feature_selection import meta_feature_selection
            
            fs_col1, fs_col2 = st.columns(2)
            with fs_col1:
                min_features_help = "Minimum number of features to retain after selection."
                min_features = st.slider(
                    "Minimum Features to Select", 3, 10, 3, 1,
                    help=min_features_help
                )
            with fs_col2:
                max_features_help = "Maximum number of features to retain after selection. Cannot exceed total available features."
                max_features = st.slider(
                    "Maximum Features to Select", 5, min(20, len(input_columns)), min(10, len(input_columns)), 1,
                    help=max_features_help
                )
            
            with st.spinner("Running automated feature selection... This may take a moment."):
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

        constraints_help = "Define minimum and/or maximum allowable values for each input feature. This will filter the dataset *before* training and evaluation to only include samples respecting these bounds. Also used by some suggestion algorithms."
        expander_constraints = st.expander("Define Min/Max Constraints for Input Features", expanded=False)
        if expander_constraints: # Only show help if expander is manually opened by user
            expander_constraints.info(constraints_help)

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

            st.markdown("---")
            st.markdown("##### Define Sum Constraint")
            st.info("Useful for compositional data where selected features should sum to a specific value (e.g., 1.0 for fractions, 100 for percentages). This filters the dataset before training/evaluation.")

            if "sum_constraint_cols" not in st.session_state:
                st.session_state.sum_constraint_cols = []
            if "sum_constraint_target" not in st.session_state:
                st.session_state.sum_constraint_target = 1.0
            if "sum_constraint_tolerance" not in st.session_state:
                st.session_state.sum_constraint_tolerance = 0.01

            selected_sum_cols = st.multiselect(
                "Select features for sum constraint:",
                options=input_columns,
                default=st.session_state.sum_constraint_cols,
                key="sum_constraint_features_multiselect",
                help="Choose two or more input features whose sum should meet the target value."
            )
            st.session_state.sum_constraint_cols = selected_sum_cols

            target_sum_val = st.number_input(
                "Target sum for selected features:",
                value=st.session_state.sum_constraint_target,
                format="%g",
                key="sum_constraint_target_value",
                help="The value the selected features should sum to (e.g., 1.0 or 100.0)."
            )
            st.session_state.sum_constraint_target = float(target_sum_val) if target_sum_val is not None else None

            target_sum_tolerance = st.number_input(
                "Tolerance for sum constraint (e.g., +/- 0.01):",
                value=st.session_state.sum_constraint_tolerance,
                min_value=0.0,
                format="%g",
                key="sum_constraint_tolerance_value",
                help="Allowable deviation from the target sum (e.g., a tolerance of 0.01 for a target sum of 1.0 means sums between 0.99 and 1.01 are accepted)."
            )
            st.session_state.sum_constraint_tolerance = float(target_sum_tolerance) if target_sum_tolerance is not None else 0.0


    # Target Properties Configuration
    if target_columns:
        st.subheader("Properties Configuration")

        # Multi-objective strategy selection
        mobo_strategy = "weighted_sum" # Default
        default_mobo_strategy = "weighted_sum"
        if "mobo_strategy_selector_loaded_value" in st.session_state:
            default_mobo_strategy = st.session_state.mobo_strategy_selector_loaded_value
            # del st.session_state.mobo_strategy_selector_loaded_value # Clear after use

        if len(target_columns) > 1:
            mobo_options = ["weighted_sum", "parego"]
            mobo_idx = mobo_options.index(default_mobo_strategy) if default_mobo_strategy in mobo_options else 0
            mobo_strategy_help = """
            Strategy for handling multiple target properties:
            - Weighted Sum: Combines objectives into a single score using fixed weights defined below. Simple and effective if weights are well-chosen.
            - ParEGO: A Bayesian Optimization strategy that uses random scalarizations. Good for exploring non-convex Pareto fronts but can be computationally more intensive if BO is used for suggestions.
            """
            mobo_strategy = st.selectbox(
                "Multi-Objective Strategy:",
                options=mobo_options,
                index=mobo_idx,
                format_func=lambda x: "Weighted Sum (Fixed Weights)" if x == "weighted_sum" else "ParEGO (Randomized Weights)",
                help=mobo_strategy_help,
                key="mobo_strategy_selector"
            )
        
        max_or_min_targets, weights_targets, thresholds_targets = [], [], []
        max_or_min_apriori, weights_apriori, thresholds_apriori = [], [], []

        for category, columns, max_or_min_list, weights_list, thresholds_list, key_prefix in [
            ("Target", target_columns, max_or_min_targets, weights_targets, thresholds_targets, "target"),
            ("A Priori", apriori_columns, max_or_min_apriori, weights_apriori, thresholds_apriori, "apriori")
        ]:
            if columns:
                st.markdown(f"#### {category} Properties Configuration")
                st.info(f"Define optimization direction, importance (weight), and optional thresholds for each {category.lower()} property.")
                
                col_per_property = 3
                properties_per_row = 3
                
                for i in range(0, len(columns), properties_per_row):
                    property_group = columns[i:i+properties_per_row]
                    property_cols = st.columns(len(property_group))
                    
                    for j, col_name in enumerate(property_group):
                        with property_cols[j]:
                            st.markdown(f"**{col_name}**")
                            
                            optimize_for = st.radio(
                                "Optimize for:", # Label made generic, context is col_name
                                ["Maximize", "Minimize"],
                                index=0, # Default to Maximize
                                key=f"{key_prefix}_opt_{col_name}",
                                horizontal=True,
                                help=f"Should the model aim to maximize or minimize '{col_name}'?"
                            )
                            
                            weight = st.number_input(
                                "Weight:", # Label made generic
                                value=1.0, 
                                step=0.1, 
                                min_value=0.1,
                                max_value=10.0,
                                key=f"{key_prefix}_weight_{col_name}",
                                help=f"Relative importance of '{col_name}' in utility calculations (higher means more important)."
                            )
                            
                            threshold = st.text_input(
                                "Threshold:", # Label made generic
                                value="", 
                                key=f"{key_prefix}_threshold_{col_name}",
                                help=f"Optional. If set, samples not meeting this threshold for '{col_name}' will be penalized or filtered in utility calculations. E.g., '>10' or '<5'."
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
                                st.session_state.model = model # Store generic model
                                st.session_state.scaler_inputs = scaler_inputs
                                st.session_state.scaler_targets = scaler_targets
                                
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
                                st.session_state.model = model # Store generic model
                                st.session_state.scaler_inputs = scaler_x # reptile_train returns scaler_x, scaler_y
                                st.session_state.scaler_targets = scaler_y
                                
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
                                st.session_state.model = model # Store generic model
                                st.session_state.scaler_inputs = scaler_x # protonet_train returns scaler_x, scaler_y
                                st.session_state.scaler_targets = scaler_y
                                
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

                            elif model_type == "PINN":
                                # Initialize the PINN model
                                model = PINNModel(
                                    input_size=len(input_columns),
                                    output_size=len(target_columns),
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    dropout_rate=dropout_rate
                                )

                                # Train the model
                                model, scaler_x, scaler_y = pinn_train(
                                    model,
                                    current_data, # Use constrained data
                                    input_columns,
                                    target_columns,
                                    pinn_epochs,
                                    pinn_learning_rate,
                                    physics_loss_weight,
                                    pinn_batch_size
                                )
                                st.session_state.model = model # Store generic model
                                st.session_state.scaler_inputs = scaler_x
                                st.session_state.scaler_targets = scaler_y

                                # Evaluate the model
                                result_df = evaluate_pinn(
                                    model,
                                    current_data, # Use constrained data
                                    input_columns,
                                    target_columns,
                                    curiosity,
                                    weights,
                                    max_or_min,
                                )
                                if result_df is not None:
                                    st.session_state["result_df"] = result_df
                                else:
                                    st.error("No results generated by PINN.")

                        # After model training (either single or last model in ensemble if that path was taken)
                        # Determine the primary trained model to be used as surrogate if BO is selected
                        active_model_object = None
                        if model_type == "Random Forest":
                            active_model_object = st.session_state.get("rf_model")
                        else: # MAML, Reptile, ProtoNet
                            active_model_object = st.session_state.get("model")

                        if use_bayesian_optimizer_for_suggestion and active_model_object and getattr(active_model_object, 'is_trained', False):
                            st.info(f"Using Bayesian Optimizer with {model_type} as surrogate to rank candidates.")

                            # Prepare data for BO
                            # current_data is already filtered by constraints
                            labeled_bo_data = current_data.dropna(subset=target_columns)
                            unlabeled_bo_data = current_data[current_data[target_columns].isna().any(axis=1)].copy()

                            if labeled_bo_data.empty or unlabeled_bo_data.empty:
                                st.error("Not enough labeled or unlabeled data for Bayesian Optimization after filtering.")
                            else:
                                X_labeled_bo = labeled_bo_data[input_columns]
                                if len(target_columns) > 1:
                                    st.info(f"Using Multi-Objective Bayesian Optimization with strategy: {mobo_strategy}")
                                    # Prepare data for MOBO
                                    # train_inputs and candidate_inputs can be DataFrames
                                    # train_targets should be NumPy array
                                    # input_columns list is also needed by the MOBO function if surrogate_model is used

                                    acq_scores = multi_objective_bayesian_optimization(
                                        train_inputs=X_labeled_bo, # DataFrame
                                        train_targets=labeled_bo_data[target_columns].values, # NumPy array
                                        candidate_inputs=unlabeled_bo_data[input_columns], # DataFrame
                                        weights=weights, # User-defined weights for objectives
                                        max_or_min=max_or_min, # User-defined directions
                                        curiosity=curiosity,
                                        acquisition=acquisition_function_bo,
                                        strategy=mobo_strategy, # From UI
                                        surrogate_model=active_model_object,
                                        input_columns=input_columns
                                    )
                                    if acq_scores is None or len(acq_scores) != len(unlabeled_bo_data):
                                        st.error("Multi-objective Bayesian optimization did not return valid acquisition scores. Falling back to standard evaluation.")
                                        # Fallback logic or stop - for now, let's just error out or let it be handled by evaluate_*
                                        # This means result_df would not be overridden by BO results here.
                                        # To prevent this, ensure evaluate_* is called if BO fails.
                                        # For now, let's assume acq_scores is valid. If not, result_df won't be set by BO.
                                        # A more robust fallback would be needed if this path is critical.
                                        # Let's structure it so that if this fails, the original result_df from evaluate_* is used.
                                        # This requires evaluate_* to run first, or a flag.
                                        # For now, if BO is selected, it's the primary source of Utility.
                                        pass # Will be handled if acq_scores is None
                                else: # Single objective BO
                                    st.info(f"Using Single-Objective Bayesian Optimization with acquisition: {acquisition_function_bo}")
                                    # BO acquisition functions are typically single-objective. Using first target.
                                    y_labeled_bo = labeled_bo_data[target_columns[0]].values

                                    bo = BayesianOptimizer(surrogate_model=active_model_object)
                                    bo.fit(X_labeled_bo, y_labeled_bo) # Pass DataFrame X to fit

                                    # Get acquisition scores for unlabeled candidates
                                    X_unlabeled_bo_np = unlabeled_bo_data[input_columns].values
                                    acq_scores = bo.acquisition_function(
                                        X_unlabeled_bo_np,
                                        acquisition=acquisition_function_bo, # From UI
                                        curiosity=curiosity # Use the curiosity from the active model's UI section
                                    )

                                result_df_bo = unlabeled_bo_data.copy()
                                if acq_scores is not None and len(acq_scores) == len(result_df_bo):
                                    result_df_bo["Utility"] = acq_scores
                                else:
                                    st.error("Failed to get acquisition scores from Bayesian Optimizer. Utility will not be based on BO.")
                                    # If acq_scores failed, result_df from model's own evaluate_* should be used.
                                    # This means we should not overwrite result_df if BO path fails to produce scores.
                                    # This logic needs evaluate_* to run first if BO is an overlay.
                                    # For now: if use_bayesian_optimizer_for_suggestion is true, we *try* to set result_df.
                                    # If it fails to get acq_scores, result_df might be from a previous step or incomplete.
                                    # Let's ensure result_df is only assigned if acq_scores are valid.
                                    # The `result_df = result_df_bo[bo_new_col_order]` line later will handle this.
                                    # So if acq_scores is None, "Utility" won't be there, and it might break.
                                    # Better: only proceed with BO result_df if acq_scores are valid.
                                    result_df = st.session_state.get("result_df") # Fallback to potentially existing result_df
                                    # This is getting complicated. Let's simplify:
                                    # The BO path will *replace* the result_df. If it fails, it errors.

                                if acq_scores is None:
                                     st.error("Acquisition score calculation failed in Bayesian Optimization. Cannot proceed with BO-based utility.")
                                     # To prevent erroring out later, we should not try to use result_df_bo
                                     # Instead, ensure the original result_df from evaluate_* is used.
                                     # This means the BO block should be self-contained in setting result_df or erroring.
                                     # For now, if acq_scores is None, we will skip overriding result_df.
                                     # This means the `result_df` from the non-BO path (evaluate_*) should have already run.
                                     # This implies a structural change:
                                     # 1. Always run evaluate_*
                                     # 2. If BO is checked, then *override* Utility (and possibly other things)
                                     # This is safer.

                                     # For now, let's assume if acq_scores is None, the original result_df remains.
                                     # The current structure IS that evaluate_* runs first, then this BO block overrides result_df.
                                     # So, if acq_scores is None here, the original result_df should persist.
                                     # The only issue is if `result_df = result_df_bo[bo_new_col_order]` runs with an incomplete `result_df_bo`.
                                     # Let's ensure `result_df_bo` is fully populated before assigning to `result_df`.
                                     # And if `acq_scores` is None, this whole BO-specific population of `result_df` will be skipped.
                                     # This is handled by the `if acq_scores is not None and len(acq_scores) == len(result_df_bo):` block.

                                # Get mean predictions and uncertainties directly from surrogate for display
                                # The surrogate's predict_with_uncertainty method should handle input_columns
                                pred_means, pred_stds = active_model_object.predict_with_uncertainty(unlabeled_bo_data, input_columns)

                                for i, col_name in enumerate(target_columns):
                                    if pred_means.ndim > 1 and pred_means.shape[1] > i:
                                        result_df_bo[col_name] = np.maximum(pred_means[:, i], 0)
                                    elif pred_means.ndim == 1 and i == 0 : # single target model
                                         result_df_bo[col_name] = np.maximum(pred_means, 0)
                                    else: # Fallback if shapes don't align as expected
                                        result_df_bo[col_name] = 0

                                # Assuming pred_stds is already (n_samples, 1) or needs to be processed if multi-target
                                result_df_bo["Uncertainty"] = np.clip(pred_stds, 1e-9, None).flatten()

                                # Novelty
                                if active_model_object.scaler_x:
                                    scaled_labeled_inputs = active_model_object.scaler_x.transform(X_labeled_bo)
                                    scaled_unlabeled_inputs = active_model_object.scaler_x.transform(unlabeled_bo_data[input_columns])
                                    novelty_scores_bo = calculate_novelty(scaled_unlabeled_inputs, scaled_labeled_inputs)
                                else:
                                    novelty_scores_bo = np.zeros(len(unlabeled_bo_data))
                                result_df_bo["Novelty"] = novelty_scores_bo

                                result_df_bo["Exploration"] = result_df_bo["Uncertainty"] * result_df_bo["Novelty"]
                                result_df_bo["Exploitation"] = 1.0 - result_df_bo["Uncertainty"]

                                result_df_bo = result_df_bo.sort_values(by="Utility", ascending=False)
                                result_df_bo["Selected for Testing"] = False
                                if not result_df_bo.empty:
                                    result_df_bo.iloc[0, result_df_bo.columns.get_loc("Selected for Testing")] = True
                                result_df_bo.reset_index(drop=True, inplace=True)

                                # Reorder columns
                                bo_cols_to_front = ["Idx_Sample"] if "Idx_Sample" in result_df_bo.columns else []
                                bo_metrics_cols = ["Utility", "Exploration", "Exploitation", "Novelty", "Uncertainty"]
                                bo_remaining_cols = [col for col in result_df_bo.columns if col not in bo_cols_to_front + bo_metrics_cols + target_columns + ["Selected for Testing"]]
                                bo_new_col_order = bo_cols_to_front + bo_metrics_cols + target_columns + bo_remaining_cols + ["Selected for Testing"]
                                bo_new_col_order = [col for col in bo_new_col_order if col in result_df_bo.columns]
                                result_df = result_df_bo[bo_new_col_order] # Override result_df with BO results
                                st.session_state["result_df"] = result_df
                        
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

                        # 4. Save Model State (starting with RF)
                        import joblib # Ensure joblib is imported (should be at top of file)
                        model_state_saved = False
                        if model_type == "Random Forest" and "rf_model" in st.session_state:
                            if st.session_state.rf_model and getattr(st.session_state.rf_model, 'is_trained', False):
                                try:
                                    # Save RF model
                                    model_buffer = io.BytesIO()
                                    joblib.dump(st.session_state.rf_model.model, model_buffer) # Save the internal sklearn RF
                                    zip_file.writestr("model_state/rf_model.joblib", model_buffer.getvalue())

                                    # Save scalers associated with RFModel
                                    if st.session_state.rf_model.scaler_x:
                                        scaler_x_buffer = io.BytesIO()
                                        joblib.dump(st.session_state.rf_model.scaler_x, scaler_x_buffer)
                                        zip_file.writestr("model_state/rf_scaler_x.joblib", scaler_x_buffer.getvalue())

                                    if st.session_state.rf_model.scaler_y:
                                        scaler_y_buffer = io.BytesIO()
                                        joblib.dump(st.session_state.rf_model.scaler_y, scaler_y_buffer)
                                        zip_file.writestr("model_state/rf_scaler_y.joblib", scaler_y_buffer.getvalue())
                                    model_state_saved = True
                                    st.info("Random Forest model state and scalers added to save file.")
                                except Exception as e:
                                    st.warning(f"Could not save Random Forest model state: {e}")

                        elif model_type in ["MAML", "Reptile", "ProtoNet"] and "model" in st.session_state:
                            if st.session_state.model: # Check if model object exists
                                try:
                                    # Save PyTorch model state_dict
                                    model_buffer = io.BytesIO()
                                    torch.save(st.session_state.model.state_dict(), model_buffer)
                                    zip_file.writestr(f"model_state/{model_type.lower()}_model_statedict.pt", model_buffer.getvalue())

                                    # Save scalers (assuming they are stored in session_state after training these models)
                                    if "scaler_inputs" in st.session_state and st.session_state.scaler_inputs:
                                        scaler_x_buffer = io.BytesIO()
                                        joblib.dump(st.session_state.scaler_inputs, scaler_x_buffer)
                                        zip_file.writestr(f"model_state/{model_type.lower()}_scaler_x.joblib", scaler_x_buffer.getvalue())

                                    if "scaler_targets" in st.session_state and st.session_state.scaler_targets:
                                        scaler_y_buffer = io.BytesIO()
                                        joblib.dump(st.session_state.scaler_targets, scaler_y_buffer)
                                        zip_file.writestr(f"model_state/{model_type.lower()}_scaler_y.joblib", scaler_y_buffer.getvalue())

                                    model_state_saved = True
                                    st.info(f"{model_type} model state and scalers added to save file.")
                                except Exception as e:
                                    st.warning(f"Could not save {model_type} model state: {e}")

                        # TODO: Add saving for GPR fallback model if it was used/is in session_state

                        if not model_state_saved:
                            st.info("Model state not saved (model not trained, or not yet supported for saving type).")

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
