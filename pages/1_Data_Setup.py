import streamlit as st
import pandas as pd
import numpy as np
import os
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
from app.digital_lab import digital_lab_ui

st.set_page_config(page_title="Data Setup", layout="wide")

st.title("1. Data Setup 🧪")
st.markdown("Prepare your dataset for the experimentation phase. You can upload an existing CSV file or generate a new design space using the Digital Lab.")

# --- Dataset Management ---
st.header("Dataset Management")

# --- Data Source Selection ---
# If Digital Lab generated data, make it the default choice.
default_index = 1 if 'generated_df' in st.session_state and st.session_state.generated_df is not None else 0
data_source = st.radio(
    "Choose a data source:",
    ("Upload Dataset", "Create with Digital Lab"),
    horizontal=True,
    key="data_source_selector",
    index=default_index
)

# Initialize data variable
data = None

if data_source == "Upload Dataset":
    # --- Original File Upload and Edit Logic ---
    def load_and_edit_dataset(upload_folder="uploads"):
        uploaded_file = st.file_uploader("Upload Dataset (CSV format):", type=["csv"])

        if "dataset" not in st.session_state:
            st.session_state["dataset"] = None

        if uploaded_file:
            # When a new file is uploaded, clear any existing generated data
            if 'generated_df' in st.session_state:
                del st.session_state['generated_df']

            file_path = os.path.join(upload_folder, uploaded_file.name)
            os.makedirs(upload_folder, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            df = pd.read_csv(file_path)
            st.session_state["dataset"] = df
            st.success("Dataset uploaded successfully!")

        # Display the dataset if available
        if st.session_state.get("dataset") is not None:
            st.markdown("### Editable Dataset")
            df = st.session_state["dataset"].copy()

            df_display = df.reset_index().rename(columns={"index": "Sample Index"})

            gb = GridOptionsBuilder.from_dataframe(df_display)
            gb.configure_default_column(editable=True)
            gb.configure_column("Sample Index", editable=False)

            grid_options = gb.build()

            grid_response = AgGrid(
                df_display,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.VALUE_CHANGED,
                data_return_mode=DataReturnMode.AS_INPUT,
                fit_columns_on_grid_load=True,
                theme="streamlit",
                key='editable_grid'
            )
            df_edited = grid_response["data"].drop(columns=["Sample Index"], errors='ignore')
            st.session_state["dataset"] = df_edited

        return st.session_state.get("dataset")

    data = load_and_edit_dataset()

elif data_source == "Create with Digital Lab":
    # --- Digital Lab UI ---
    generated_df = digital_lab_ui()
    if generated_df is not None:
        # For consistency, let's also use the "dataset" session_state key
        st.session_state.dataset = generated_df
        data = generated_df

# --- This section runs if data is loaded either from upload or digital lab ---
if data is not None:
    st.header("Feature and Property Configuration")

    # Define feature selection and targets
    col1, col2, col3 = st.columns(3)

    with col1:
        input_columns = st.multiselect(
            "Input Features:",
            options=data.columns.tolist(),
            default=st.session_state.get("input_columns", []),
            key="input_columns_multiselect",
            help="Select the material composition variables"
        )
        st.session_state["input_columns"] = input_columns

    with col2:
        remaining_cols_for_target = [col for col in data.columns if col not in input_columns]
        target_columns = st.multiselect(
            "Target Properties:",
            options=remaining_cols_for_target,
            default=st.session_state.get("target_columns", []),
            key="target_columns_multiselect",
            help="Select the material properties you want to optimize"
        )
        st.session_state["target_columns"] = target_columns

    with col3:
        remaining_cols_for_apriori = [col for col in data.columns if col not in input_columns + target_columns]
        apriori_columns = st.multiselect(
            "A Priori Properties:",
            options=remaining_cols_for_apriori,
            default=st.session_state.get("apriori_columns", []),
            key="apriori_columns_multiselect",
            help="Select properties with known values that constrain the optimization"
        )
        st.session_state["apriori_columns"] = apriori_columns

    # Input Feature Constraints Configuration
    if input_columns:
        st.subheader("Input Feature Constraints (Optional)")
        if "constraints" not in st.session_state:
            st.session_state.constraints = {col: {"min": None, "max": None} for col in data.columns}

        with st.expander("Define Min/Max & Sum Constraints for Input Features"):
            for col in input_columns:
                if col not in st.session_state.constraints:
                     st.session_state.constraints[col] = {"min": None, "max": None}

                c1, c2 = st.columns(2)
                current_min = st.session_state.constraints[col]["min"]
                current_max = st.session_state.constraints[col]["max"]

                new_min = c1.number_input(f"Min for {col}:", value=current_min if current_min is not None else np.nan, format="%g", key=f"min_{col}")
                new_max = c2.number_input(f"Max for {col}:", value=current_max if current_max is not None else np.nan, format="%g", key=f"max_{col}")

                st.session_state.constraints[col]["min"] = None if np.isnan(new_min) else float(new_min)
                st.session_state.constraints[col]["max"] = None if np.isnan(new_max) else float(new_max)

            st.markdown("---")
            st.markdown("##### Define Sum Constraint")

            selected_sum_cols = st.multiselect(
                "Select features for sum constraint:",
                options=input_columns,
                default=st.session_state.get("sum_constraint_cols", []),
                key="sum_constraint_features_multiselect"
            )
            st.session_state.sum_constraint_cols = selected_sum_cols

            target_sum_val = st.number_input(
                "Target sum:",
                value=st.session_state.get("sum_constraint_target", 1.0),
                format="%g",
                key="sum_constraint_target_value"
            )
            st.session_state.sum_constraint_target = float(target_sum_val) if target_sum_val is not None else None

            target_sum_tolerance = st.number_input(
                "Tolerance:",
                value=st.session_state.get("sum_constraint_tolerance", 0.01),
                min_value=0.0,
                format="%g",
                key="sum_constraint_tolerance_value"
            )
            st.session_state.sum_constraint_tolerance = float(target_sum_tolerance) if target_sum_tolerance is not None else 0.0

    # Target Properties Configuration
    if target_columns:
        st.subheader("Properties Configuration")
        if "optimization_params" not in st.session_state:
            st.session_state["optimization_params"] = {}

        for col_name in target_columns:
            if col_name not in st.session_state.optimization_params:
                st.session_state.optimization_params[col_name] = {"direction": "Maximize", "weight": 1.0}

        # Cleanup params for columns that are no longer selected
        for col_name in list(st.session_state.optimization_params.keys()):
            if col_name not in target_columns:
                del st.session_state.optimization_params[col_name]

        with st.expander("Define Optimization Direction and Weights"):
            for col_name in target_columns:
                c1, c2 = st.columns(2)
                direction = c1.radio("Optimize for:", ["Maximize", "Minimize"], key=f"opt_{col_name}", horizontal=True, index=["Maximize", "Minimize"].index(st.session_state.optimization_params[col_name]["direction"]))
                weight = c2.number_input("Weight:", value=st.session_state.optimization_params[col_name]["weight"], step=0.1, min_value=0.1, key=f"weight_{col_name}")
                st.session_state.optimization_params[col_name] = {"direction": direction, "weight": weight}
else:
    st.info("Please upload or generate a dataset to begin configuration.")
