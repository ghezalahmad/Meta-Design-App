import pytest
import pandas as pd
import numpy as np

# Directly import the pure logic function
from app.utils.digital_lab import generate_design_space

@pytest.fixture
def sample_components():
    """Fixture to provide a sample list of components for testing."""
    return [
        {'name': 'Cement', 'properties': {'cost': 100, 'density': 3.15}},
        {'name': 'Water', 'properties': {'cost': 1, 'density': 1.0}},
        {'name': 'Sand', 'properties': {'cost': 5, 'density': 2.65, 'fineness': 2.5}},
    ]

def test_generate_design_space_success(sample_components):
    """
    Tests the successful generation of a design space DataFrame.
    """
    num_samples = 150
    generated_df = generate_design_space(sample_components, num_samples)

    # --- Assertions ---
    assert isinstance(generated_df, pd.DataFrame), "The function should return a pandas DataFrame."
    assert len(generated_df) == num_samples, "The DataFrame should have the requested number of samples."

    # Check for correct columns
    expected_ratio_cols = ['Cement_ratio', 'Water_ratio', 'Sand_ratio']
    # 'fineness' is unique to Sand to test that all properties are found
    expected_prop_cols = ['cost', 'density', 'fineness']
    for col in expected_ratio_cols + expected_prop_cols:
        assert col in generated_df.columns, f"Column '{col}' is missing from the generated DataFrame."

    # Check the sum-to-one constraint for ratios
    ratio_sums = generated_df[expected_ratio_cols].sum(axis=1)
    assert np.allclose(ratio_sums, 1.0), "The sum of ratios for each row should be approximately 1."

    # Verify a sample property calculation for cost
    first_row = generated_df.iloc[0]
    expected_cost = (
        first_row['Cement_ratio'] * 100 +
        first_row['Water_ratio'] * 1 +
        first_row['Sand_ratio'] * 5
    )
    assert np.isclose(first_row['cost'], expected_cost), "The 'cost' property is not calculated correctly."

    # Verify a sample property calculation for a property (fineness) that only one component has
    expected_fineness = first_row['Sand_ratio'] * 2.5
    assert np.isclose(first_row['fineness'], expected_fineness), "A property unique to one component is not calculated correctly."

def test_generate_design_space_empty_input():
    """
    Tests that the function handles empty or invalid component lists gracefully.
    """
    # Test with an empty list of components
    df_empty = generate_design_space([], 100)
    assert isinstance(df_empty, pd.DataFrame)
    assert df_empty.empty, "Should return an empty DataFrame for an empty component list."

    # Test with components that are missing names
    df_no_name = generate_design_space([{'properties': {'cost': 100}}], 100)
    assert isinstance(df_no_name, pd.DataFrame)
    assert df_no_name.empty, "Should return an empty DataFrame if components are missing names."
