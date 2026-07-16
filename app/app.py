import sys
import os

# ---------------- PATH FOR STREAMLIT CLOUD ----------------
# Get the absolute path of app.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------- STACK IMPORTS ----------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from src.features import normalize_columns, add_features

# ---------------- PATH RESOLUTION FOR STREAMLIT CLOUD ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ---------------- IMPORTS FROM SRC ----------------
from src.features import normalize_columns, add_features

# ---------------- CACHED LOADERS ----------------
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "models", "model_pipeline.pkl")
    if not os.path.exists(model_path):
        st.error("❌ Trained model not found in models/model_pipeline.pkl")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "data", "crop_yield.csv")
    if not os.path.exists(data_path):
        st.error("❌ Dataset not found in data/crop_yield.csv")
        st.stop()
    return pd.read_csv(data_path)

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources
model = load_model()
df = load_data()

# ---------------- STYLES ----------------
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2e7d32;
    text-align: center;
}
.prediction-box {
    font-size: 2rem;
    color: #1b5e20;
    font-weight: bold;
    text-align: center;
    padding: 1rem;
    background-color: #e8f5e9;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<h1 class="main-header">🌾 Crop Yield Prediction System</h1>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigation",
    ["🎯 Prediction", "📊 Data Explorer", "ℹ️ About"]
)

# =========================================================
# 🎯 PREDICTION PAGE
# =========================================================
if page == "🎯 Prediction":

    st.header("🎯 Predict Crop Yield")

    col1, col2 = st.columns(2)

    with col1:
        crop = st.selectbox("Crop", sorted(df["Crop"].unique()))
        season = st.selectbox("Season", sorted(df["Season"].unique()))
        state = st.selectbox("State", sorted(df["State"].unique()))
        year = st.slider("Year", 1997, 2025, 2020)

    with col2:
        area = st.number_input("Area (hectares)", min_value=0.1, value=1.0)
        rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1000.0)
        fertilizer = st.number_input("Fertilizer", min_value=0.0, value=100.0)
        pesticide = st.number_input("Pesticide", min_value=0.0, value=10.0)

    st.markdown("---")

    if st.button("🚀 Predict Yield", use_container_width=True):

        try:
            # Create input dataframe (MUST match training columns)
            input_df = pd.DataFrame([{
                "Crop": crop,
                "Season": season,
                "State": state,
                "Crop_Year": year,
                "Area": area,
                "Annual_Rainfall": rainfall,
                "Fertilizer": fertilizer,
                "Pesticide": pesticide
            }])

            # Apply preprocessing
            input_df = normalize_columns(input_df)
            input_df = add_features(input_df)

            # Predict
            pred_log = model.predict(input_df)[0]
            prediction = np.expm1(pred_log)

            # ---------------- OUTPUT ----------------
            st.markdown("### 🌿 Prediction Result")

            colA, colB, colC = st.columns(3)

            with colA:
                st.metric("Yield", f"{prediction:,.2f} units/ha")

            with colB:
                total = prediction * area
                st.metric("Total Production", f"{total:,.2f}")

            with colC:
                avg = df[df["Crop"] == crop]["Yield"].mean()
                diff = prediction - avg
                pct = (diff / avg) * 100 if avg != 0 else 0
                st.metric("vs Avg", f"{pct:+.2f}%", delta=f"{diff:,.2f}")

            # ---------------- VISUAL ----------------
            st.markdown("### 📊 Yield Comparison")

            crop_data = df[df["Crop"] == crop]["Yield"]

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=crop_data, name="Historical Yield"))

            fig.add_vline(
                x=prediction,
                line_dash="dash",
                line_color="red",
                annotation_text="Prediction"
            )

            fig.update_layout(
                xaxis_title="Yield",
                yaxis_title="Frequency"
            )

            st.plotly_chart(fig, use_container_width=True)

            # ---------------- RECOMMENDATIONS ----------------
            st.markdown("### 💡 Recommendations")

            rec = []

            if rainfall < 800:
                rec.append("Increase irrigation (low rainfall detected)")
            if fertilizer < 100:
                rec.append("Increase fertilizer usage")
            if pesticide < 5:
                rec.append("Consider pest control measures")

            if rec:
                for r in rec:
                    st.info(r)
            else:
                st.success("Conditions look optimal")

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

# =========================================================
#  DATA EXPLORER
# =========================================================
elif page == "📊 Data Explorer":

    st.header("📊 Data Explorer")

    st.write(df.head())

    fig1 = px.box(df, x="Crop", y="Yield", title="Yield Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="Annual_Rainfall", y="Yield",
                      title="Rainfall vs Yield")
    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# ABOUT PAGE
# =========================================================
else:

    st.header("About")

    st.markdown("""
This application predicts crop yield using a trained Machine Learning pipeline.

### Features:
- Automated preprocessing
- No manual encoding
- Improved accuracy

### Model:
- Random Forest (with feature engineering)
                
### Developer:
- Daksh Parmar
""")