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
from app.utils import select_acquisition_function
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
            st.session_state["acquisition_function"] = select_acquisition_function(curiosity, len(data.dropna(subset=target_columns)))
            result_df = evaluate_maml(meta_model=model, data=data, input_columns=input_columns, target_columns=target_columns, curiosity=curiosity, weights=weights_targets, max_or_min=max_or_min_targets, acquisition=st.session_state.get("acquisition_function"), dynamic_acquisition=True, min_labeled_samples=5)

        elif model_type == "Reptile":
            model = ReptileModel(input_size=len(input_columns), output_size=len(target_columns), hidden_size=hidden_size, num_layers=num_layers, dropout_rate=dropout_rate)
            model, scaler_inputs, scaler_targets = reptile_train(model=model, data=data, input_columns=input_columns, target_columns=target_columns, epochs=reptile_epochs, learning_rate=reptile_learning_rate, num_tasks=reptile_num_tasks)
            st.session_state["acquisition_function"] = select_acquisition_function(curiosity, len(data.dropna(subset=target_columns)))
            result_df = evaluate_reptile(model=model, data=data, input_columns=input_columns, target_columns=target_columns, curiosity=curiosity, weights=weights_targets, max_or_min=max_or_min_targets, acquisition=st.session_state.get("acquisition_function"))
        elif model_type == "ProtoNet":
            model = ProtoNetModel(input_size=len(input_columns), output_size=len(target_columns), embedding_size=embedding_size, num_layers=num_layers, dropout_rate=dropout_rate)
            model, scaler_inputs, scaler_targets = protonet_train(model=model, data=data, input_columns=input_columns, target_columns=target_columns, epochs=protonet_epochs, learning_rate=protonet_learning_rate, num_tasks=protonet_num_tasks, num_shot=num_shot, num_query=num_query)
            st.session_state["acquisition_function"] = select_acquisition_function(curiosity, len(data.dropna(subset=target_columns)))
            result_df = evaluate_protonet(model=model, data=data, input_columns=input_columns, target_columns=target_columns, curiosity=curiosity, weights=weights_targets, max_or_min=max_or_min_targets, acquisition=st.session_state.get("acquisition_function"))
        elif model_type == "Random Forest":
            model, scaler_inputs, scaler_targets = train_rf_model(data=data, input_columns=input_columns, target_columns=target_columns, n_estimators=rf_n_estimators, perform_grid_search=rf_perform_grid_search)
            st.session_state["acquisition_function"] = select_acquisition_function(curiosity, len(data.dropna(subset=target_columns)))
            result_df = evaluate_rf_model(rf_model=model, data=data, input_columns=input_columns, target_columns=target_columns, curiosity=curiosity, weights=weights_targets, max_or_min=max_or_min_targets, acquisition=st.session_state.get("acquisition_function"))
        elif model_type == "PINN":
            model = PINNModel(input_size=len(input_columns), output_size=len(target_columns), hidden_size=hidden_size, num_layers=num_layers, dropout_rate=dropout_rate)
            model, scaler_inputs, scaler_targets = pinn_train(model=model, data=data, input_columns=input_columns, target_columns=target_columns, epochs=pinn_epochs, learning_rate=pinn_learning_rate, physics_loss_weight=physics_loss_weight, batch_size=pinn_batch_size)
            st.session_state["acquisition_function"] = select_acquisition_function(curiosity, len(data.dropna(subset=target_columns)))
            result_df = evaluate_pinn(model=model, data=data, input_columns=input_columns, target_columns=target_columns, curiosity=curiosity, weights=weights_targets, max_or_min=max_or_min_targets, acquisition=st.session_state.get("acquisition_function"))

        st.session_state.model = model
        st.session_state.result_df = result_df
        st.session_state["experiment_run"] = True
        st.success("Experiment completed successfully!")


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

# --- Display Results ---
result_df = st.session_state.get("result_df")
if result_df is not None:
    st.header("Experiment Summary")
    st.markdown("#### Top 10 Suggested Samples:")
    st.dataframe(result_df.head(10), use_container_width=True)

    # Retrieve necessary parameters from session state for visualizations
    target_columns = st.session_state.get("target_columns", [])
    input_columns = st.session_state.get("input_columns", [])
    optimization_params = st.session_state.get("optimization_params", {})

    # --- Visualization Tabs ---
    tab1, tab2, tab3 = st.tabs(["📈 Target Analysis", "🧠 Model Insights", "🔬 Advanced Analysis"])

    with tab1:
        st.header("Target Property Visualizations")
        st.info("Explore the relationships and distributions of your target properties based on the model's predictions.")

        if not target_columns:
            st.warning("No target columns selected. Please configure them on the 'Data Setup' page.")
        else:
            # Pareto Front
            if len(target_columns) >= 2:
                st.subheader("Pareto Front")
                col1, col2 = st.columns(2)
                obj1 = col1.selectbox("First objective:", target_columns, index=0, key="pareto_obj1")
                obj2 = col2.selectbox("Second objective:", target_columns, index=min(1, len(target_columns)-1), key="pareto_obj2")
                obj_directions = [optimization_params.get(obj1, {}).get("direction", "max"), optimization_params.get(obj2, {}).get("direction", "max")]
                fig = create_pareto_front_visualization(result_df, [obj1, obj2], obj_directions)
                st.plotly_chart(fig, use_container_width=True)

            # Scatter Matrix
            if len(target_columns) >= 2:
                st.subheader("Scatter Matrix of Target Properties")
                fig = plot_scatter_matrix_with_uncertainty(result_df, target_columns, "Utility")
                st.plotly_chart(fig, use_container_width=True)

            # Property Distributions
            st.subheader("Property Distributions")
            fig = visualize_property_distributions(result_df, target_columns)
            st.plotly_chart(fig, use_container_width=True)

            # Parallel Coordinates
            st.subheader("Parallel Coordinates Plot")
            fig = create_parallel_coordinates(result_df, target_columns)
            st.plotly_chart(fig, use_container_width=True)

            # 3D Scatter
            if len(target_columns) >= 3:
                st.subheader("3D Scatter Plot")
                x_prop = st.selectbox("X-axis property:", target_columns, index=0, key="3d_x")
                y_prop = st.selectbox("Y-axis property:", target_columns, index=min(1, len(target_columns)-1), key="3d_y")
                z_prop = st.selectbox("Z-axis property:", target_columns, index=min(2, len(target_columns)-1), key="3d_z")
                color_by = st.radio("Color by:", ["Utility", "Uncertainty", "Novelty"] + target_columns, horizontal=True, key="3d_color")
                fig = create_3d_scatter(result_df, x_prop, y_prop, z_prop, color_by=color_by)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Model Behavior and Insights")
        st.info("Understand how the model arrived at its suggestions and the balance between exploration and exploitation.")

        # Acquisition Function
        st.subheader("Acquisition Function Analysis")
        # Note: 'curiosity' value might need to be retrieved more robustly from session state
        curiosity = st.session_state.get("curiosity", 0.0)
        acquisition_function = st.session_state.get("acquisition_function", "UCB")
        fig = create_acquisition_function_visualization(result_df, acquisition_function, curiosity)
        st.plotly_chart(fig, use_container_width=True)

        # Exploration vs Exploitation Tradeoff
        st.subheader("Exploration vs. Exploitation Tradeoff")
        fig = visualize_exploration_exploitation_tradeoff(result_df)
        st.plotly_chart(fig, use_container_width=True)

        # t-SNE
        st.subheader("t-SNE Visualization of Input Space")
        if input_columns:
            # Combine labeled data and predictions for a comprehensive t-SNE plot
            labeled_data = st.session_state.dataset.dropna(subset=target_columns)
            labeled_data['is_train_data'] = 'Train'

            predictions_df = result_df.copy()
            predictions_df['is_train_data'] = 'Predicted'

            combined_df = pd.concat([labeled_data, predictions_df], ignore_index=True)

            fig = create_tsne_plot_with_hover(combined_df, input_columns, "Utility")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No input columns selected for t-SNE.")

    with tab3:
        st.header("Advanced Analysis")
        st.info("Perform deeper analysis, such as comparing models or identifying optimal regions in the design space.")

        # Optimal Regions
        if len(input_columns) >= 2 and len(target_columns) > 0:
            st.subheader("Optimal Regions Analysis")
            max_or_min_targets = [optimization_params.get(col, {}).get("direction", "max") for col in target_columns]
            highlight_df = pd.concat([result_df[target_columns], result_df[input_columns]], axis=1)
            fig = highlight_optimal_regions(highlight_df, target_columns, max_or_min_targets)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Optimal Regions Analysis requires at least 2 input features and 1 target property.")

        # Model Comparison
        if "model_history" in st.session_state and len(st.session_state["model_history"]) > 1:
            st.subheader("Model Comparison")
            # This is a placeholder for a more detailed comparison UI
            st.write("Model comparison functionality would be implemented here.")
            # fig = visualize_model_comparison(...)
            # st.plotly_chart(fig)

else:
    st.info("No results to display. Please run an experiment on the 'Experimentation' page first.")
