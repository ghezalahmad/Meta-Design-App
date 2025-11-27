import streamlit as st
import pandas as pd
import numpy as np

def generate_design_space(components, num_samples):
    """
    Generates a design space DataFrame based on material components and the number of samples.

    This is a pure logic function, free of any Streamlit UI code, making it easily testable.

    Args:
        components (list of dict): A list of component dictionaries, where each dict has
                                   'name' and 'properties' keys.
        num_samples (int): The number of sample formulations to generate.

    Returns:
        pandas.DataFrame: A DataFrame representing the generated design space, with columns
                          for component ratios and calculated properties.
    """
    if not components or not all(c.get('name') for c in components):
        # Return an empty DataFrame or raise an error if components are not well-defined
        return pd.DataFrame()

    component_names = [c['name'] for c in components]

    # Generate random ratios that sum to 1 using the Dirichlet distribution
    samples = np.random.dirichlet(np.ones(len(component_names)), size=num_samples)

    df = pd.DataFrame(samples, columns=[f"{name}_ratio" for name in component_names])

    # --- Robust Property Calculation ---
    all_property_keys = set()
    for component in components:
        all_property_keys.update(component.get('properties', {}).keys())

    for prop_name in sorted(list(all_property_keys)):
        df[prop_name] = 0
        for i, comp in enumerate(components):
            ratio_col = f"{comp['name']}_ratio"
            prop_value = comp.get('properties', {}).get(prop_name, 0)
            df[prop_name] += df[ratio_col] * prop_value

    return df

def digital_lab_ui():
    """
    Renders the Streamlit UI for the Digital Lab and handles user interactions.
    This function is now a "controller" that calls the pure `generate_design_space` logic.
    """
    st.header("🧪 Digital Lab: Create Your Design Space")

    if 'components' not in st.session_state:
        st.session_state.components = [
            {'name': 'Cement', 'properties': {'Cost': 120, 'CO2': 800}},
            {'name': 'Water', 'properties': {'Cost': 1, 'CO2': 0}},
            {'name': 'Sand', 'properties': {'Cost': 20, 'CO2': 5}},
        ]

    st.subheader("1. Define Material Components & Socio-Economic Metrics")

    for i, component in enumerate(st.session_state.components):
        with st.container():
            col1, col2, col3, col_remove = st.columns([2, 2, 2, 1])
            component['name'] = col1.text_input(f"Component {i+1} Name", value=component.get('name', ''), key=f"name_{i}")

            # Dedicated fields for socio-economic metrics
            cost = col2.number_input("Cost ($/unit)", value=component.get('properties', {}).get('Cost', 0.0), key=f"cost_{i}", format="%.2f")
            co2 = col3.number_input("CO2 (kg/unit)", value=component.get('properties', {}).get('CO2', 0.0), key=f"co2_{i}", format="%.2f")

            # Store them back in the properties dictionary
            if 'properties' not in component:
                component['properties'] = {}
            component['properties']['Cost'] = cost
            component['properties']['CO2'] = co2

            if col_remove.button("Remove", key=f"remove_{i}"):
                st.session_state.components.pop(i)
                st.rerun()

    if st.button("Add Another Component"):
        st.session_state.components.append({'name': '', 'properties': {}})
        st.rerun()

    st.subheader("2. Define Constraints and Generation Parameters")

    st.write("Constraint: The sum of component ratios must equal 1.")

    num_samples = st.number_input(
        "Number of Samples to Generate",
        min_value=10,
        max_value=10000,
        value=1000,
        help="The total number of material formulations to generate in the design space."
    )

    st.subheader("3. Generate Design Space")

    if st.button("Generate Dataset"):
        with st.spinner("Generating design space..."):
            # Call the pure logic function
            df = generate_design_space(st.session_state.components, num_samples)

            if df.empty:
                st.error("Please define at least one component and give it a name.")
            else:
                st.session_state.generated_df = df
                st.success(f"Successfully generated a dataset with {num_samples} samples!")
                st.dataframe(df.head())

    return st.session_state.get('generated_df', None)

if __name__ == '__main__':
    digital_lab_ui()
