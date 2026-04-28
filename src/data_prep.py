# src/data_prep.py
import pandas as pd
import numpy as np

def load_data(train_path, test_path):
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    return train, test

def add_credit_history_months(df):
    df = df.copy()
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["credit_history_months"] = (
        (df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12 +
        (df["issue_d"].dt.month - df["earliest_cr_line"].dt.month)
    )
    cols_to_drop = ["issue_d", "earliest_cr_line", "url", "issues_d", "pymnt_plan"]
    df = df.drop(columns=cols_to_drop)
    return df

def cap_outliers(df, col, percentile=99.9):
    cap = df[col].quantile(percentile / 100)
    df[col] = df[col].clip(upper=cap)
    return df