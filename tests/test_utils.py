import pytest
import numpy as np
from app.utils import calculate_utility

def test_calculate_utility_basic():
    """
    Tests the basic functionality of the calculate_utility function.
    """
    predictions = np.array([[10], [20], [30]])
    uncertainties = np.array([[1], [2], [3]])
    novelty = np.array([0.1, 0.5, 0.9])
    weights = np.array([1.0])
    max_or_min = ["max"]
    curiosity = 0.5  # Balanced curiosity

    # Calculate utility
    utility = calculate_utility(
        predictions, uncertainties, novelty, curiosity, weights, max_or_min
    )

    # Assertions
    assert utility.shape == (3,), "Utility should be a 1D array with length equal to the number of samples."

    # Check that the utility score increases with higher prediction values (for maximization)
    assert utility[2] > utility[1] > utility[0], "Utility should be higher for better predictions."

def test_calculate_utility_minimization():
    """
    Tests the utility calculation for a minimization objective.
    """
    predictions = np.array([[10], [20], [30]])
    uncertainties = np.array([[1], [2], [3]])
    novelty = np.array([0.1, 0.5, 0.9])
    weights = np.array([1.0])
    max_or_min = ["min"]
    curiosity = 0.0 # Pure exploitation

    utility = calculate_utility(
        predictions, uncertainties, novelty, curiosity, weights, max_or_min
    )

    # For minimization, a lower prediction should result in a higher utility score.
    assert utility[0] > utility[1] > utility[2], "For minimization, lower predictions should have higher utility."

def test_calculate_utility_multi_objective():
    """
    Tests the utility calculation for a multi-objective optimization scenario.
    """
    # Two objectives: Maximize the first, minimize the second
    predictions = np.array([[10, 100], [20, 50], [30, 25]])
    uncertainties = np.array([[1, 10], [2, 5], [3, 2]])
    novelty = np.array([0.1, 0.5, 0.9])
    weights = np.array([0.7, 0.3]) # Weight the first objective more
    max_or_min = ["max", "min"]
    curiosity = 0.0 # Pure exploitation

    utility = calculate_utility(
        predictions, uncertainties, novelty, curiosity, weights, max_or_min
    )

    assert utility.shape == (3,), "Utility should be a 1D array for multi-objective scenarios as well."
    # The third sample ([30, 25]) is the best on both objectives (highest on first, lowest on second)
    # and should therefore have the highest utility.
    assert np.argmax(utility) == 2, "The sample that is best on both objectives should have the highest utility."
