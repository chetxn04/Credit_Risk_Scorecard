# tests/test_scorecard.py
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.scorecard import compute_scaling_params, score_to_probability, probability_to_score

def test_scaling_params():
    """B and A should be computed correctly from PDO formula"""
    A, B = compute_scaling_params(PDO=20, anchor_score=600, anchor_odds=10)
    assert abs(B - 20 / np.log(2)) < 1e-6
    assert abs(A - (600 + B * np.log(10))) < 1e-6

def test_round_trip_conversion():
    """Converting prob -> score -> prob should return original probability"""
    A, B = compute_scaling_params()
    probs = np.array([0.05, 0.10, 0.20, 0.35, 0.50])
    scores = probability_to_score(probs, A, B)
    recovered_probs = score_to_probability(scores, A, B)
    np.testing.assert_array_almost_equal(probs, recovered_probs, decimal=6)

def test_higher_pd_lower_score():
    """Higher default probability should produce lower score"""
    A, B = compute_scaling_params()
    low_risk_score = probability_to_score(0.05, A, B)
    high_risk_score = probability_to_score(0.40, A, B)
    assert low_risk_score > high_risk_score, \
        f"Low risk score {low_risk_score:.1f} should exceed high risk score {high_risk_score:.1f}"


def test_pdo_property():
    """Increasing score by PDO should double the odds"""
    A, B = compute_scaling_params(PDO=20, anchor_score=600, anchor_odds=10)
    prob1 = probability_to_score(0.10, A, B)
    prob2 = probability_to_score(0.10, A, B) + 20  # add one PDO
    odds1 = (1 - 0.10) / 0.10
    prob2_converted = score_to_probability(prob2, A, B)
    odds2 = (1 - prob2_converted) / prob2_converted
    assert abs(odds2 - 2 * odds1) < 1e-6, \
        f"Adding PDO should double odds: expected {2*odds1:.4f}, got {odds2:.4f}"