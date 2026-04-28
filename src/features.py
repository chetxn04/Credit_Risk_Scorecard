# src/features.py
import pandas as pd
import numpy as np

def apply_woe_transformation(df, binning_models, feature_cols):
    df_woe = df.copy()
    for col in feature_cols:
        optb = binning_models[col]
        df_woe[col] = optb.transform(df[col].values, metric="woe")
    return df_woe

def get_iv_table(binning_models):
    iv_results = []
    for col, optb in binning_models.items():
        binning_table = optb.binning_table.build()
        iv = binning_table["IV"].iloc[-1]
        iv_results.append({"feature": col, "iv": iv})
    return pd.DataFrame(iv_results).sort_values("iv", ascending=False).reset_index(drop=True)