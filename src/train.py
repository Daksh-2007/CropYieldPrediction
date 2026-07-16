import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

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

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numerical)
])

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline.fit(X_train, y_train)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print("✅ Model saved successfully")