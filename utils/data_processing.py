import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, input_columns, target_columns, apriori_columns=None):
    """
    Preprocess the dataset for training and inference.

    Args:
        data (pd.DataFrame): The dataset.
        input_columns (list): Names of input feature columns.
        target_columns (list): Names of target columns.
        apriori_columns (list): Names of a priori feature columns (optional).

    Returns:
        tuple: Processed training inputs, training targets, inference inputs,
               scaled a priori data (if provided), input scaler, target scaler.
    """
    # Identify rows with known target values
    known_targets = ~data[target_columns[0]].isna()

    # Split data into training (known targets) and inference (unknown targets)
    inputs_train = data.loc[known_targets, input_columns].values
    targets_train = data.loc[known_targets, target_columns].values
    inputs_infer = data.loc[~known_targets, input_columns].values

    # Scale inputs
    scaler_inputs = StandardScaler()
    inputs_train_scaled = scaler_inputs.fit_transform(inputs_train)
    inputs_infer_scaled = scaler_inputs.transform(inputs_infer)

    # Scale targets
    scaler_targets = StandardScaler()
    targets_train_scaled = scaler_targets.fit_transform(targets_train)

    # Handle a priori columns (optional)
    if apriori_columns:
        apriori_data = data[apriori_columns].values
        scaler_apriori = StandardScaler()
        apriori_scaled = scaler_apriori.fit_transform(apriori_data[known_targets])
        apriori_infer_scaled = scaler_apriori.transform(apriori_data[~known_targets])
    else:
        apriori_infer_scaled = np.zeros((inputs_infer.shape[0], 1))  # Default to zeros if not provided

    return (
        inputs_train_scaled,
        targets_train_scaled,
        inputs_infer_scaled,
        apriori_infer_scaled,
        scaler_inputs,
        scaler_targets,
    )
