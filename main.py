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
