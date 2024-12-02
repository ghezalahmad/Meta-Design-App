import json
import pandas as pd


def restore_session(session_file):
    """
    Restore session variables from a JSON file.

    Args:
        session_file (io.BytesIO): Uploaded session file in JSON format.

    Returns:
        dict: Restored session variables.
    """
    session_data = json.load(session_file)

    restored_session = {
        "input_columns": session_data.get("input_columns", []),
        "target_columns": session_data.get("target_columns", []),
        "apriori_columns": session_data.get("apriori_columns", []),
        "weights_targets": session_data.get("weights_targets", []),
        "weights_apriori": session_data.get("weights_apriori", []),
        "thresholds_targets": session_data.get("thresholds_targets", []),
        "thresholds_apriori": session_data.get("thresholds_apriori", []),
        "curiosity": session_data.get("curiosity", 0.0),
        "results": pd.DataFrame(session_data.get("results", [])) if "results" in session_data else pd.DataFrame(),
    }

    return restored_session


def save_session(input_columns, target_columns, apriori_columns, weights_targets, weights_apriori,
                 thresholds_targets, thresholds_apriori, curiosity, results):
    """
    Save session variables to a JSON file.

    Args:
        input_columns (list): List of input feature columns.
        target_columns (list): List of target property columns.
        apriori_columns (list): List of a priori property columns.
        weights_targets (list): Weights for target properties.
        weights_apriori (list): Weights for a priori properties.
        thresholds_targets (list): Thresholds for target properties.
        thresholds_apriori (list): Thresholds for a priori properties.
        curiosity (float): Curiosity factor for utility calculation.
        results (pd.DataFrame): Results dataframe containing predictions, utility scores, etc.

    Returns:
        str: JSON string representing the session data.
    """
    session_data = {
        "input_columns": input_columns,
        "target_columns": target_columns,
        "apriori_columns": apriori_columns,
        "weights_targets": weights_targets,
        "weights_apriori": weights_apriori,
        "thresholds_targets": thresholds_targets,
        "thresholds_apriori": thresholds_apriori,
        "curiosity": curiosity,
        "results": results.to_dict(orient="records") if not results.empty else [],
    }

    return json.dumps(session_data, indent=4)
