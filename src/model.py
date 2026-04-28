# src/model.py
import numpy as np
import statsmodels.api as sm

def train_logistic_regression(X_train, y_train):
    X_const = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_const)
    result = logit_model.fit()
    return result

def predict_proba(result, X):
    X_const = sm.add_constant(X)
    return result.predict(X_const)