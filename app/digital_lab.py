import streamlit as st
import pandas as pd
import numpy as np

def digital_lab_ui():
    st.header("🧪 Digital Lab: Create Your Design Space")

    # Initialize session state for components if it doesn't exist
    if 'components' not in st.session_state:
        st.session_state.components = [
            {'name': 'Cement', 'properties': {'cost': 5, 'density': 3.15}},
            {'name': 'Water', 'properties': {'cost': 0.1, 'density': 1.0}},
            {'name': 'Sand', 'properties': {'cost': 0.5, 'density': 2.65}},
        ]

    st.subheader("1. Define Material Components")

    # --- Component Management ---
    for i, component in enumerate(st.session_state.components):
        with st.container():
            col1, col2, col_remove = st.columns([2, 4, 1])
            component['name'] = col1.text_input(f"Component {i+1} Name", value=component['name'], key=f"name_{i}")

            # Properties
            props = component.get('properties', {})
            props_str = col2.text_area(
                f"Properties (key: value)",
                value='\n'.join([f"{k}: {v}" for k, v in props.items()]),
                key=f"props_{i}",
                height=100
            )
            try:
                # Parse the properties string back into a dictionary
                parsed_props = {}
                for line in props_str.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        parsed_props[key.strip()] = float(value.strip())
                component['properties'] = parsed_props
            except ValueError:
                st.warning(f"Invalid format for properties of {component['name']}. Please use 'key: value' format.")

            if col_remove.button("Remove", key=f"remove_{i}"):
                st.session_state.components.pop(i)
                st.rerun()

    if st.button("Add Another Component"):
        st.session_state.components.append({'name': '', 'properties': {}})
        st.rerun()

    st.subheader("2. Define Constraints and Generation Parameters")

    # --- Constraints ---
    st.write("Constraint: The sum of component ratios must equal 1.")

    # --- Generation Parameters ---
    num_samples = st.number_input(
        "Number of Samples to Generate",
        min_value=10,
        max_value=10000,
        value=1000,
        help="The total number of material formulations to generate in the design space."
    )

    st.subheader("3. Generate Design Space")

    if st.button("Generate Dataset"):
        if not st.session_state.components or not all(c['name'] for c in st.session_state.components):
            st.error("Please define at least one component and give it a name.")
            return

        # --- Generation Logic ---
        with st.spinner("Generating design space..."):
            component_names = [c['name'] for c in st.session_state.components]

            # Generate random ratios that sum to 1
            samples = np.random.dirichlet(np.ones(len(component_names)), size=num_samples)

            df = pd.DataFrame(samples, columns=[f"{name}_ratio" for name in component_names])

            # --- Robust Property Calculation ---
            # 1. Collect all unique property keys from all components
            all_property_keys = set()
            for component in st.session_state.components:
                all_property_keys.update(component.get('properties', {}).keys())

            # 2. Calculate each property for the generated samples
            for prop_name in sorted(list(all_property_keys)): # Sort for consistent column order
                df[prop_name] = 0
                for i, comp in enumerate(st.session_state.components):
                    ratio_col = f"{comp['name']}_ratio"
                    # Use .get(prop_name, 0) to handle cases where a component might not have a specific property
                    prop_value = comp.get('properties', {}).get(prop_name, 0)
                    df[prop_name] += df[ratio_col] * prop_value

            # Store the generated data in session state
            st.session_state.generated_df = df
            st.success(f"Successfully generated a dataset with {num_samples} samples!")
            st.dataframe(df.head())

    # Return the generated dataframe if it exists
    return st.session_state.get('generated_df', None)

if __name__ == '__main__':
    digital_lab_ui()
