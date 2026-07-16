import pandas as pd
import numpy as np

def normalize_columns(df):
    df = df.copy()

    if "Year" in df.columns:
        df.rename(columns={"Year": "Crop_Year"}, inplace=True)

    if "Rainfall" in df.columns:
        df.rename(columns={"Rainfall": "Annual_Rainfall"}, inplace=True)

    return df


def add_features(df):
    df = df.copy()

    df["Area"] = df["Area"].replace(0, 1e-6)

    df["Fertilizer_Intensity"] = df["Fertilizer"] / df["Area"]
    df["Water_Index"] = df["Annual_Rainfall"] + (df["Pesticide"] * 10)

    return df


def prepare_target(df):
    df = df.copy()
    df["Yield_log"] = np.log1p(df["Yield"])
    return df