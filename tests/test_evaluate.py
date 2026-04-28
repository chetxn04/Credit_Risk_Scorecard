# tests/test_evaluate.py
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.evaluate import compute_gini, compute_ks, compute_psi

def test_gini_random_model():
    """A random model should have Gini close to 0"""
    np.random.seed(42)
    y_true = np.array([0, 1] * 500)
    y_prob = np.random.uniform(0, 1, 1000)
    gini = compute_gini(y_true, y_prob)
    assert abs(gini) < 0.1, f"Random model Gini should be near 0, got {gini}"

def test_gini_perfect_model():
    """A perfect model should have Gini close to 1"""
    y_true = np.array([0] * 500 + [1] * 500)
    y_prob = np.array([0.1] * 500 + [0.9] * 500)
    gini = compute_gini(y_true, y_prob)
    assert gini > 0.95, f"Perfect model Gini should be near 1, got {gini}"

def test_ks_random_model():
    """A random model should have KS close to 0"""
    np.random.seed(42)
    y_true = np.array([0, 1] * 500)
    y_prob = np.random.uniform(0, 1, 1000)
    ks = compute_ks(y_true, y_prob)
    assert ks < 0.1, f"Random model KS should be near 0, got {ks}"

def test_ks_perfect_model():
    """A perfect model should have KS close to 1"""
    y_true = np.array([0] * 500 + [1] * 500)
    y_prob = np.array([0.1] * 500 + [0.9] * 500)
    ks = compute_ks(y_true, y_prob)
    assert ks > 0.95, f"Perfect model KS should be near 1, got {ks}"

def test_psi_identical_distributions():
    """Identical distributions should have PSI close to 0"""
    np.random.seed(42)
    dist = np.random.normal(0.2, 0.05, 1000)
    psi = compute_psi(dist, dist)
    assert psi < 0.01, f"Identical distributions PSI should be near 0, got {psi}"

def test_psi_different_distributions():
    """Very different distributions should have PSI above 0.25"""
    np.random.seed(42)
    expected = np.random.normal(0.2, 0.05, 1000)
    actual = np.random.normal(0.6, 0.05, 1000)
    psi = compute_psi(expected, actual)
    assert psi > 0.25, f"Different distributions PSI should be high, got {psi}"