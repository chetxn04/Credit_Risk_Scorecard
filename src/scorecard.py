# src/scorecard.py
import numpy as np

def compute_scaling_params(PDO=20, anchor_score=600, anchor_odds=10):
    B = PDO / np.log(2)
    A = anchor_score + B * np.log(anchor_odds)
    return A, B

def score_to_probability(score, A, B):
    log_odds = (A - score) / B
    prob = 1 / (1 + np.exp(-log_odds))  
    return prob

def probability_to_score(prob, A, B):
    log_odds = np.log(prob / (1 - prob))
    score = A - B * log_odds
    return score