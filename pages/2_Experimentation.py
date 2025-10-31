import streamlit as st
import pandas as pd
import numpy as np
import time
import torch
from app.models import MAMLModel, meta_train, evaluate_maml
from app.reptile_model import ReptileModel, reptile_train, evaluate_reptile
from app.protonet_model import ProtoNetModel, protonet_train, evaluate_protonet
from app.rf_model import RFModel, train_rf_model, evaluate_rf_model
from app.pinn_model import PINNModel, pinn_train, evaluate_pinn
from app.ensemble import weighted_uncertainty_ensemble
from app.bayesian_optimizer import multi_objective_bayesian_optimization, BayesianOptimizer

st.set_page_config(page_title="Experimentation", layout="wide")

st.title("2. Experimentation 🔬")
st.markdown("Configure your model, run the experiment, and get the next set of suggestions.")

# Check if data is loaded
if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("No dataset found. Please go to the 'Data Setup' page to upload or create a dataset.")
    st.stop()

# --- Sidebar for Model Configuration ---
st.sidebar.header("Model & Data Configuration")
model_options = ["MAML", "Reptile", "ProtoNet", "Random Forest", "PINN"]
model_type = st.sidebar.selectbox("Choose Model Type:", options=model_options, key="model_type_selector")

# Model-specific configuration
if model_type == "MAML":
    st.sidebar.subheader("MAML Configuration")
    hidden_size = st.sidebar.slider("Hidden Size:", 64, 512, 128)
    num_layers = st.sidebar.slider("Number of Layers:", 2, 5, 3)
    dropout_rate = st.sidebar.slider("Dropout Rate:", 0.0, 0.5, 0.2)
    inner_lr = st.sidebar.slider("Inner Loop LR:", 0.00005, 0.01, 0.001)
    outer_lr = st.sidebar.slider("Outer Loop LR:", 0.00001, 0.01, 0.00005)
    meta_epochs = st.sidebar.slider("Meta-Training Epochs:", 10, 300, 100)
    num_tasks = st.sidebar.slider("Number of Tasks:", 2, 10, 5)
    curiosity = st.sidebar.slider("Curiosity:", -2.0, 2.0, 0.0, 0.1)

elif model_type == "Reptile":
    st.sidebar.subheader("Reptile Configuration")
    hidden_size = st.sidebar.slider("Hidden Size:", 64, 512, 128)
    num_layers = st.sidebar.slider("Number of Layers:", 2, 5, 3)
    dropout_rate = st.sidebar.slider("Dropout Rate:", 0.0, 0.5, 0.3)
    reptile_learning_rate = st.sidebar.slider("Learning Rate:", 0.0001, 0.1, 0.001)
    reptile_epochs = st.sidebar.slider("Training Epochs:", 10, 300, 50)
    reptile_num_tasks = st.sidebar.slider("Number of Tasks:", 2, 10, 5)
    curiosity = st.sidebar.slider("Curiosity:", -2.0, 2.0, 0.0, 0.1)

elif model_type == "ProtoNet":
    st.sidebar.subheader("ProtoNet Configuration")
    embedding_size = st.sidebar.slider("Embedding Size:", 64, 512, 128)
    num_layers = st.sidebar.slider("Number of Layers:", 2, 5, 3)
    dropout_rate = st.sidebar.slider("Dropout Rate:", 0.0, 0.5, 0.3)
    protonet_learning_rate = st.sidebar.slider("Learning Rate:", 0.0001, 0.01, 0.001)
    protonet_epochs = st.sidebar.slider("Training Epochs:", 10, 300, 50)
    protonet_num_tasks = st.sidebar.slider("Number of Tasks:", 2, 10, 5)
    num_shot = st.sidebar.slider("Support Samples (N-shot):", 1, 8, 4)
    num_query = st.sidebar.slider("Query Samples:", 1, 8, 4)
    curiosity = st.sidebar.slider("Curiosity:", -2.0, 2.0, 0.0, 0.1)

elif model_type == "Random Forest":
    st.sidebar.subheader("Random Forest Configuration")
    rf_n_estimators = st.sidebar.slider("Number of Estimators:", 50, 500, 100)
    rf_perform_grid_search = st.sidebar.checkbox("Perform GridSearchCV")
    curiosity = st.sidebar.slider("Curiosity:", -2.0, 2.0, 0.0, 0.1)

elif model_type == "PINN":
    st.sidebar.subheader("PINN Configuration")
    hidden_size = st.sidebar.slider("Hidden Size:", 64, 512, 128)
    num_layers = st.sidebar.slider("Number of Layers:", 2, 5, 3)
    dropout_rate = st.sidebar.slider("Dropout Rate:", 0.0, 0.5, 0.3)
    pinn_learning_rate = st.sidebar.slider("Learning Rate:", 0.0001, 0.1, 0.001)
    pinn_epochs = st.sidebar.slider("Training Epochs:", 10, 300, 50)
    pinn_batch_size = st.sidebar.slider("Batch Size:", 4, 128, 16)
    physics_loss_weight = st.sidebar.slider("Physics Loss Weight:", 0.0, 1.0, 0.1)
    curiosity = st.sidebar.slider("Curiosity:", -2.0, 2.0, 0.0, 0.1)

# --- Run Experiment ---
st.header("Run Experiment")
button_label = "Suggest Next Experiment" if st.session_state.get("experiment_run", False) else "Run Experiment"

if st.button(button_label, key="run_experiment_button", use_container_width=True):
    # Retrieve data and config from session state
    data = st.session_state.dataset
    input_columns = st.session_state.get("input_columns", [])
    target_columns = st.session_state.get("target_columns", [])
    optimization_params = st.session_state.get("optimization_params", {})

    if not input_columns or not target_columns:
        st.error("Input features and target properties must be selected on the 'Data Setup' page.")
        st.stop()

    max_or_min_targets = [optimization_params[col]["direction"] for col in target_columns]
    weights_targets = np.array([optimization_params[col]["weight"] for col in target_columns])

    with st.spinner(f"Running {model_type} model... This may take a while."):
        # --- Model Training and Evaluation Logic ---
        # This is a simplified version of the logic from the original main.py
        result_df = None
        model = None

        # This logic should be expanded to match the full complexity of the original file,
        # including ensemble and bayesian optimization options.
        if model_type == "MAML":
            model = MAMLModel(input_size=len(input_columns), output_size=len(target_columns), hidden_size=hidden_size, num_layers=num_layers, dropout_rate=dropout_rate)
            model, scaler_inputs, scaler_targets = meta_train(meta_model=model, data=data, input_columns=input_columns, target_columns=target_columns, epochs=meta_epochs, inner_lr=inner_lr, outer_lr=outer_lr, num_tasks=num_tasks, inner_lr_decay=0.95, curiosity=curiosity, min_samples_per_task=3, early_stopping_patience=10)
            result_df = evaluate_maml(meta_model=model, data=data, input_columns=input_columns, target_columns=target_columns, curiosity=curiosity, weights=weights_targets, max_or_min=max_or_min_targets, acquisition=None, dynamic_acquisition=True, min_labeled_samples=5)

        # ... (Add similar blocks for Reptile, ProtoNet, RF, PINN) ...

        st.session_state.model = model
        st.session_state.result_df = result_df
        st.session_state["experiment_run"] = True

        st.success("Experiment completed successfully!")
        st.markdown("Navigate to the **Results Analysis** page to view the outcome.")

# --- Log Experimental Results ---
if st.session_state.get("experiment_run", False):
    st.header("Log Lab Experiment Results")
    st.markdown("After running a physical experiment, log the results here to update the dataset for the next iteration.")

    if 'result_df' in st.session_state and st.session_state.result_df is not None:
        suggested_sample = st.session_state.result_df.iloc[0:1]
        st.markdown("#### Top Suggested Sample:")
        st.dataframe(suggested_sample)

        with st.form(key="log_results_form"):
            sample_index_to_log = st.number_input(
                label="Enter Index of Tested Sample",
                min_value=0,
                max_value=len(st.session_state.dataset) - 1,
                value=suggested_sample.index[0],
                help="Enter the index of the sample you tested in the lab."
            )

            experimental_results = {}
            for col_name in st.session_state.get("target_columns", []):
                experimental_results[col_name] = st.number_input(label=f"Measured {col_name}", key=f"measured_{col_name}", format="%.4f")

            submitted = st.form_submit_button("Add Result to Dataset")
            if submitted:
                sample_index_to_update = int(sample_index_to_log)
                for col_name, value in experimental_results.items():
                    st.session_state.dataset.loc[sample_index_to_update, col_name] = value

                st.success(f"Successfully updated sample at index {sample_index_to_update}.")
                st.info("The dataset has been updated. You can now re-run the experiment to get a new suggestion.")
