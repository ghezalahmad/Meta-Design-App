
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
from app.lolopy_model import train_lolopy_model, evaluate_lolopy_model
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
model_options = ["MAML", "Reptile", "ProtoNet", "Random Forest", "PINN", "Lolopy Random Forest"]
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

elif model_type == "Reptile":
    st.sidebar.subheader("Reptile Configuration")
    hidden_size = st.sidebar.slider("Hidden Size:", 64, 512, 128)
    num_layers = st.sidebar.slider("Number of Layers:", 2, 5, 3)
    dropout_rate = st.sidebar.slider("Dropout Rate:", 0.0, 0.5, 0.3)
    reptile_learning_rate = st.sidebar.slider("Learning Rate:", 0.0001, 0.1, 0.001)
    reptile_epochs = st.sidebar.slider("Training Epochs:", 10, 300, 50)
    reptile_num_tasks = st.sidebar.slider("Number of Tasks:", 2, 10, 5)

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

elif model_type == "Random Forest":
    st.sidebar.subheader("Random Forest Configuration")
    rf_n_estimators = st.sidebar.slider("Number of Estimators:", 50, 500, 100)
    rf_perform_grid_search = st.sidebar.checkbox("Perform GridSearchCV")

elif model_type == "PINN":
    st.sidebar.subheader("PINN Configuration")
    hidden_size = st.sidebar.slider("Hidden Size:", 64, 512, 128)
    num_layers = st.sidebar.slider("Number of Layers:", 2, 5, 3)
    dropout_rate = st.sidebar.slider("Dropout Rate:", 0.0, 0.5, 0.3)
    pinn_learning_rate = st.sidebar.slider("Learning Rate:", 0.0001, 0.1, 0.001)
    pinn_epochs = st.sidebar.slider("Training Epochs:", 10, 300, 50)
    pinn_batch_size = st.sidebar.slider("Batch Size:", 4, 128, 16)
    physics_loss_weight = st.sidebar.slider("Physics Loss Weight:", 0.0, 1.0, 0.1)

elif model_type == "Lolopy Random Forest":
    st.sidebar.subheader("Lolopy Random Forest Configuration")
    lolopy_n_estimators = st.sidebar.slider("Number of Estimators:", 50, 500, 100)

# --- Run Experiment ---
st.header("Run Experiment")
curiosity = st.slider("Curiosity:", -2.0, 2.0, 0.0, 0.1)
button_label = "Suggest Next Experiment" if st.session_state.get("experiment_run", False) else "Run Experiment"

if st.button(button_label, key="run_experiment_button", width='stretch'):
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

        elif model_type == "Reptile":
            model = ReptileModel(input_size=len(input_columns), output_size=len(target_columns), hidden_size=hidden_size, num_layers=num_layers, dropout_rate=dropout_rate)
            model, _, _ = reptile_train(model, data, input_columns, target_columns, reptile_epochs, reptile_learning_rate, reptile_num_tasks)
            result_df = evaluate_reptile(model, data, input_columns, target_columns, curiosity, weights_targets, max_or_min_targets)

        elif model_type == "ProtoNet":
            model = ProtoNetModel(input_size=len(input_columns), output_size=len(target_columns), embedding_size=embedding_size, num_layers=num_layers, dropout_rate=dropout_rate)
            model, _, _ = protonet_train(model, data, input_columns, target_columns, protonet_epochs, protonet_learning_rate, protonet_num_tasks, num_shot, num_query)
            result_df = evaluate_protonet(model, data, input_columns, target_columns, num_shot, curiosity, weights_targets, max_or_min_targets)

        elif model_type == "Random Forest":
            model, _, _ = train_rf_model(data, input_columns, target_columns, n_estimators=rf_n_estimators, perform_grid_search=rf_perform_grid_search)
            result_df = evaluate_rf_model(model, data, input_columns, target_columns, curiosity, weights_targets, max_or_min_targets)

        elif model_type == "PINN":
            model = PINNModel(input_size=len(input_columns), output_size=len(target_columns), hidden_size=hidden_size, num_layers=num_layers, dropout_rate=dropout_rate)
            model, _, _ = pinn_train(model, data, input_columns, target_columns, pinn_epochs, pinn_learning_rate, physics_loss_weight, pinn_batch_size)
            result_df = evaluate_pinn(model, data, input_columns, target_columns, curiosity, weights_targets, max_or_min_targets)

        elif model_type == "Lolopy Random Forest":
            model, _, _ = train_lolopy_model(data, input_columns, target_columns, n_estimators=lolopy_n_estimators)
            result_df = evaluate_lolopy_model(model, data, input_columns, target_columns, curiosity, weights_targets, max_or_min_targets)

        st.session_state.model = model
        st.session_state.result_df = result_df
        st.session_state["experiment_run"] = True
        st.success("Experiment completed successfully! Results are displayed below.")

# --- Display Results and Visualizations ---
if st.session_state.get("experiment_run", False):
    result_df = st.session_state.get("result_df")
    if result_df is not None:
        result_df["Row number"] = result_df.index
        st.header("Experiment Summary & Analysis 📊")
        st.markdown("#### Full Suggested Samples:")
        st.dataframe(result_df, width='stretch')

        target_columns = st.session_state.get("target_columns", [])
        input_columns = st.session_state.get("input_columns", [])
        optimization_params = st.session_state.get("optimization_params", {})

        tab1, tab2, tab3 = st.tabs(["📈 Target Analysis", "🧠 Model Insights", "🔬 Advanced Analysis"])

        with tab1:
            st.header("Target Property Visualizations")
            if not target_columns:
                st.warning("No target columns selected.")
            else:
                if len(target_columns) >= 2:
                    st.subheader("Pareto Front")
                    col1, col2 = st.columns(2)
                    obj1 = col1.selectbox("First objective:", target_columns, index=0, key="pareto_obj1")
                    obj2 = col2.selectbox("Second objective:", target_columns, index=min(1, len(target_columns)-1), key="pareto_obj2")
                    obj_directions = [optimization_params.get(obj1, {}).get("direction", "max"), optimization_params.get(obj2, {}).get("direction", "max")]
                    fig = create_pareto_front_visualization(result_df, [obj1, obj2], obj_directions)
                    if fig:
                        st.plotly_chart(fig, width='stretch')

                st.subheader("Scatter Matrix")
                fig = plot_scatter_matrix_with_uncertainty(result_df, target_columns, "Utility")
                if fig:
                    st.plotly_chart(fig, width='stretch')

                st.subheader("Property Distributions")
                fig = visualize_property_distributions(result_df, target_columns)
                if fig:
                    st.plotly_chart(fig, width='stretch')

                st.subheader("Parallel Coordinates")
                fig = create_parallel_coordinates(result_df, target_columns)
                if fig:
                    st.plotly_chart(fig, width='stretch')

                if len(target_columns) >= 3:
                    st.subheader("3D Scatter Plot")
                    x_prop = st.selectbox("X-axis property:", target_columns, index=0, key="3d_x")
                    y_prop = st.selectbox("Y-axis property:", target_columns, index=min(1, len(target_columns)-1), key="3d_y")
                    z_prop = st.selectbox("Z-axis property:", target_columns, index=min(2, len(target_columns)-1), key="3d_z")
                    color_by = st.radio("Color by:", ["Utility", "Uncertainty", "Novelty"] + target_columns, horizontal=True, key="3d_color")
                    fig = create_3d_scatter(result_df, x_prop, y_prop, z_prop, color_by=color_by)
                    if fig:
                        st.plotly_chart(fig, width='stretch')
        with tab2:
            st.header("Model Behavior and Insights")
            curiosity = st.session_state.get("curiosity", 0.0)
            fig = create_acquisition_function_visualization(result_df, None, curiosity)
            if fig:
                st.plotly_chart(fig, width='stretch')

            st.subheader("Exploration vs. Exploitation Tradeoff")
            fig = visualize_exploration_exploitation_tradeoff(result_df)
            if fig:
                st.plotly_chart(fig, width='stretch')

            st.subheader("t-SNE Visualization")
            fig = create_tsne_plot_with_hover(result_df, input_columns, "Utility")
            if fig:
                st.plotly_chart(fig, width='stretch')

        with tab3:
            st.header("Advanced Analysis")
            if len(input_columns) >= 2 and len(target_columns) > 0:
                st.subheader("Optimal Regions Analysis")
                max_or_min_targets = [optimization_params.get(col, {}).get("direction", "max") for col in target_columns]
                highlight_df = pd.concat([result_df[target_columns], result_df[input_columns]], axis=1)
                fig = highlight_optimal_regions(highlight_df, target_columns, max_or_min_targets)
                if fig:
                    st.plotly_chart(fig, width='stretch')

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
