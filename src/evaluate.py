import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from features import normalize_columns, add_features, prepare_target

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_yield.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_pipeline.pkl")

df = pd.read_csv(DATA_PATH)

df = normalize_columns(df)
df = add_features(df)
df = prepare_target(df)

categorical = ["Crop", "Season", "State"]
numerical = [
    "Crop_Year", "Area", "Annual_Rainfall",
    "Fertilizer", "Pesticide",
    "Fertilizer_Intensity", "Water_Index"
]

X = df[categorical + numerical]
y = df["Yield_log"]

pipeline = joblib.load(MODEL_PATH)

pred = pipeline.predict(X)

print("R2:", r2_score(y, pred))
print("RMSE:", np.sqrt(mean_squared_error(y, pred)))