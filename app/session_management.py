import os
import pandas as pd
import json



def restore_session(session_data):
    """
    Restore session variables from a session file.
    Args:
        session_data (dict): Parsed JSON data from the session file.
    Returns:
        dict: Restored session variables.
    """
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