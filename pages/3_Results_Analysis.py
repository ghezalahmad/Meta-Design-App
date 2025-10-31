import streamlit as st
import pandas as pd
import numpy as np
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

st.set_page_config(page_title="Results Analysis", layout="wide")

st.title("3. Results Analysis 📊")
st.markdown("Analyze the results of your experiment, visualize the data, and gain insights into the model's suggestions.")

# Check if an experiment has been run
if "experiment_run" not in st.session_state or not st.session_state.experiment_run:
    st.warning("No experiment results found. Please run an experiment on the 'Experimentation' page first.")
    st.stop()

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
        fig = create_acquisition_function_visualization(result_df, None, curiosity)
        st.plotly_chart(fig, use_container_width=True)

        # Exploration vs Exploitation Tradeoff
        st.subheader("Exploration vs. Exploitation Tradeoff")
        fig = visualize_exploration_exploitation_tradeoff(result_df)
        st.plotly_chart(fig, use_container_width=True)

        # t-SNE
        st.subheader("t-SNE Visualization of Input Space")
        if input_columns:
            fig = create_tsne_plot_with_hover(result_df, input_columns, "Utility")
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
