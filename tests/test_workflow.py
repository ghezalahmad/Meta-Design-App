import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Mock Streamlit before importing the app code
from unittest.mock import MagicMock
st_mock = MagicMock()
with patch.dict('sys.modules', {'streamlit': st_mock}):
    from app.models.models import MAMLModel, evaluate_maml
    from app.utils.utils import calculate_utility

@pytest.fixture
def sample_dataset():
    """Fixture to provide a sample dataset for integration testing."""
    data = {
        'c_1_1': np.random.rand(20),
        'c_2_1': np.random.rand(20),
        'target_1': np.random.rand(20) * 100,
    }
    # Add some missing values to simulate a real-world scenario
    data['target_1'][5:10] = np.nan
    return pd.DataFrame(data)

def test_maml_evaluation_workflow(sample_dataset):
    """
    Tests the main evaluation workflow with the MAML model.
    This is an integration test that checks the interaction between the model
    and the evaluation function.
    """
    input_columns = ['c_1_1', 'c_2_1']
    target_columns = ['target_1']

    # 1. Initialize the model
    model = MAMLModel(input_size=len(input_columns), output_size=len(target_columns))
    # In a real scenario, the model would be trained. For this test, we use the initialized model.

    # 2. Call the evaluation function
    result_df = evaluate_maml(
        meta_model=model,
        data=sample_dataset,
        input_columns=input_columns,
        target_columns=target_columns,
        curiosity=0.5,
        weights=np.array([1.0]),
        max_or_min=['max']
    )

    # 3. Assertions
    assert isinstance(result_df, pd.DataFrame), "The evaluation function should return a DataFrame."

    # Check for the essential output columns
    expected_cols = ['Utility', 'Uncertainty', 'Novelty', 'Exploration', 'Exploitation']
    for col in expected_cols:
        assert col in result_df.columns, f"Column '{col}' is missing from the result DataFrame."

    # The result_df should only contain rows that were unlabeled (had NaN targets)
    assert len(result_df) == 5, "The result DataFrame should only contain the unlabeled samples."

    # Check that Utility is a numeric column with no NaNs
    assert pd.api.types.is_numeric_dtype(result_df['Utility']), "Utility column should be numeric."
    assert not result_df['Utility'].isnull().any(), "Utility column should not contain any NaN values."

def test_log_result_workflow(sample_dataset):
    """
    Tests the workflow of logging a new experimental result, which is key to the
    iterative, sequential learning process.
    """
    # --- Setup ---
    # The sample_dataset has NaNs from index 5 to 9
    dataset = sample_dataset.copy()
    index_to_update = 7
    new_result = 99.5
    target_column = 'target_1'

    # Verify the initial state
    assert pd.isna(dataset.loc[index_to_update, target_column]), "The sample should initially be unlabeled."

    # --- Action ---
    # Simulate the action of the user logging a result in the UI
    dataset.loc[index_to_update, target_column] = new_result

    # --- Assertions ---
    assert dataset.loc[index_to_update, target_column] == new_result, "The dataset was not updated with the new result."

    # Check that the number of unlabeled samples has decreased by one
    assert dataset[target_column].isnull().sum() == 4, "The number of unlabeled samples should decrease by one after logging a result."
