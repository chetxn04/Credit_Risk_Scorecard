# src/evaluate.py
import numpy as np

def compute_gini(y_true, y_prob):
    sorted_indices = np.argsort(-y_prob)
    y_sorted = y_true.iloc[sorted_indices] if hasattr(y_true, 'iloc') else y_true[sorted_indices]
    
    total_bads = y_sorted.sum()
    total_goods = len(y_sorted) - total_bads
    
    cum_bads = np.cumsum(y_sorted) / total_bads
    cum_goods = np.cumsum(1 - y_sorted) / total_goods
    
    cum_bads = np.insert(cum_bads, 0, 0)
    cum_goods = np.insert(cum_goods, 0, 0)
    
    auc = np.trapezoid(cum_bads, cum_goods)
    gini = 2 * auc - 1
    
    return gini

def compute_ks(y_true, y_prob):
    sorted_indices = np.argsort(-y_prob)
    y_sorted = y_true.iloc[sorted_indices] if hasattr(y_true, 'iloc') else y_true[sorted_indices]
    
    total_bads = y_sorted.sum()
    total_goods = len(y_sorted) - total_bads
    
    cum_bads = np.cumsum(y_sorted) / total_bads
    cum_goods = np.cumsum(1 - y_sorted) / total_goods
    
    ks = np.max(np.abs(cum_bads - cum_goods))
    
    return ks

def compute_psi(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    psi_bins = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = psi_bins.sum()
    
    return psi