
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
    visualize_model_comparison
)

st.set_page_config(page_title="Experimentation", layout="wide")

st.title("2. Experimentation & Results 🔬")
st.markdown("Configure your model, run the experiment, analyze the results, and log your findings—all in one place.")

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

# ... (Add configurations for other models: ProtoNet, RF, PINN) ...

# --- Run Experiment ---
st.header("Run Experiment")
button_label = "Suggest Next Experiment" if st.session_state.get("experiment_run", False) else "Run Experiment"

if st.button(button_label, key="run_experiment_button", use_container_width=True):
    data = st.session_state.dataset
    input_columns = st.session_state.get("input_columns", [])
    target_columns = st.session_state.get("target_columns", [])
    optimization_params = st.session_state.get("optimization_params", {})

    if not input_columns or not target_columns:
        st.error("Input features and target properties must be selected on the 'Data Setup' page.")
        st.stop()

    max_or_min_targets = [optimization_params[col]["direction"] for col in target_columns]
    weights_targets = np.array([optimization_params[col]["weight"] for col in target_columns])

    with st.spinner(f"Running {model_type} model..."):
        model, result_df = None, None
        if model_type == "MAML":
            model = MAMLModel(input_size=len(input_columns), output_size=len(target_columns), hidden_size=hidden_size, num_layers=num_layers, dropout_rate=dropout_rate)
            model, _, _ = meta_train(model, data, input_columns, target_columns, meta_epochs, inner_lr, outer_lr, num_tasks)
            result_df = evaluate_maml(model, data, input_columns, target_columns, curiosity, weights_targets, max_or_min_targets)

        # ... (Add logic for other models) ...

        st.session_state.model = model
        st.session_state.result_df = result_df
        st.session_state["experiment_run"] = True
        st.success("Experiment completed successfully! Results are displayed below.")

# --- Display Results and Visualizations ---
if st.session_state.get("experiment_run", False):
    result_df = st.session_state.get("result_df")
    if result_df is not None:
        st.header("Experiment Summary & Analysis 📊")
        st.markdown("#### Top 10 Suggested Samples:")
        st.dataframe(result_df.head(10), use_container_width=True)

        target_columns = st.session_state.get("target_columns", [])
        input_columns = st.session_state.get("input_columns", [])
        optimization_params = st.session_state.get("optimization_params", {})

        tab1, tab2, tab3 = st.tabs(["📈 Target Analysis", "🧠 Model Insights", "🔬 Advanced Analysis"])

        with tab1:
            # ... (Paste all visualization logic from 3_Results_Analysis.py here) ...
            st.header("Target Property Visualizations")
            if not target_columns:
                st.warning("No target columns selected.")
            else:
                if len(target_columns) >= 2:
                    st.subheader("Pareto Front")
                    # ... (Pareto logic) ...
                st.subheader("Scatter Matrix")
                fig = plot_scatter_matrix_with_uncertainty(result_df, target_columns)
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Parallel Coordinates")
                fig = create_parallel_coordinates(result_df, target_columns)
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.header("Model Behavior and Insights")
            # ... (t-SNE, etc.) ...
            st.subheader("t-SNE Visualization")
            fig = create_tsne_plot_with_hover(result_df, input_columns)
            st.plotly_chart(fig, use_container_width=True)


        # --- Log Experimental Results ---
        st.header("Log Lab Experiment Results")
        st.markdown("After running a physical experiment, log the results here to update the dataset.")

        suggested_sample = result_df.iloc[0:1]
        st.markdown("#### Top Suggested Sample:")
        st.dataframe(suggested_sample)

        with st.form(key="log_results_form"):
            sample_index_to_log = st.number_input("Index of Tested Sample", value=suggested_sample.index[0])
            experimental_results = {col: st.number_input(f"Measured {col}", format="%.4f") for col in target_columns}
            if st.form_submit_button("Add Result to Dataset"):
                for col, val in experimental_results.items():
                    st.session_state.dataset.loc[int(sample_index_to_log), col] = val
                st.success(f"Updated sample {int(sample_index_to_log)}. You can now re-run the experiment.")
