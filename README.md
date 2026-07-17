# 🌾 Crop Yield Forecasting System

An end-to-end Machine Learning web application designed to help farmers, agronomists, and agricultural planners forecast crop yields based on climate metrics, regional location, and agricultural inputs (fertilizer & pesticides).

🚀 **Live Demo:** [[cropyieldprediction-daksh-2007](https://cropyieldprediction-daksh2007.streamlit.app/)]

---

## 📌 Project Overview

Predicting agricultural yield is crucial for food security, resource management, and economic planning. This project leverages historical agricultural data across various Indian states, seasons, and crop types to build a predictive Machine Learning pipeline.

The application features a clean, responsive **Streamlit** user interface backed by an automated **Scikit-Learn** pipeline, interactive **Plotly** data visualizations, and automated agronomic recommendations.

---

## 📊 Model Performance & Accuracy

The core predictive engine is powered by a **Random Forest Regressor** trained on **19,689 historical agricultural records**. To handle the high variance and skewness in crop yields, the target variable is transformed using a logarithmic scale ($\log(1 + y)$).

| Metric | Log Space ($\log(1 + \text{Yield})$) | Actual Yield Scale ($\text{Yield}$) |
| :--- | :---: | :---: |
| **$R^2$ Score** | **`0.9420` (94.20%)** | **`0.9491` (94.91%)** |
| **Root Mean Squared Error (RMSE)** | `0.2735` | `198.11` |
| **Mean Absolute Error (MAE)** | `0.1245` | **`6.24` units/ha** |

---

## ✨ Key Features

- **🎯 Interactive Yield Prediction:** Real-time yield and total production estimates based on user-defined inputs (Crop, State, Season, Area, Rainfall, Fertilizer, and Pesticides).
- **📊 Dynamic Yield Comparison:** Interactive Plotly histograms comparing individual predictions against historical regional crop averages.
- **💡 Smart Agronomic Recommendations:** Rule-based decision support suggesting actions for low rainfall, under-fertilization, or pest management.
- **📈 Data Explorer Dashboard:** Visual exploration of crop distributions, rainfall-yield correlations, and regional trends.
- **🔄 Robust Pipeline:** Built with Scikit-Learn `Pipeline` and `ColumnTransformer` featuring `OneHotEncoder` and `StandardScaler` to prevent data leakage.

---

## 🛠️ Tech Stack

- **Frontend / Web Framework:** Streamlit, HTML5, CSS3
- **Machine Learning & Data Processing:** Python 3.8+, Scikit-Learn, Pandas, NumPy, Joblib
- **Data Visualization:** Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib

---

## 📁 Project Structure

```text
├── data/
│   └── crop_yield.csv          # Historical crop yield dataset (19,689 rows)
├── models/
│   └── model_pipeline.pkl      # Trained Scikit-Learn Pipeline model
├── src/
│   ├── features.py             # Feature engineering & normalization modules
│   ├── train.py                # Model training and pipeline serialization script
│   └── evaluate.py             # Model evaluation and metrics calculation
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
