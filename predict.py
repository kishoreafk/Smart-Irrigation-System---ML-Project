import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# --- Load Models ---
print("\n🔄 Loading saved models...")
try:
    rf_model = joblib.load("random_forest_model.pkl")
    print("✅ Random Forest model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load Random Forest model: {e}")

try:
    xgb_booster = xgb.Booster()
    xgb_booster.load_model("xgboost_model.json")
    print("✅ XGBoost model loaded successfully!\n")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load XGBoost model: {e}")

# --- Input New Sample ---
data = {
    'Soil Moisture': [25],
    'Temperature': [30],
    'Soil Humidity': [60],
    'Time': [14],
    'Wind speed (Km/h)': [5],
    'Air humidity (%)': [55],
    'Wind gust (Km/h)': [7],
    'Pressure (KPa)': [101],
    'ph': [6.8],
    'rainfall': [10]
}

df = pd.DataFrame(data)

# Feature Engineering (match training pipeline)
def get_day_part(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

df['DayPart'] = df['Time'].apply(get_day_part)

# Rolling averages (for single sample, use current values)
weather_cols = ['Soil Moisture', 'Temperature', 'Soil Humidity', 'Wind speed (Km/h)', 'Air humidity (%)', 'Wind gust (Km/h)', 'Pressure (KPa)']
for col in weather_cols:
    df[f'{col}_rolling_mean'] = df[col]  # For single prediction, use current value
    df[f'{col}_rolling_std'] = 0  # Assume no variation

# Feature columns (match training)
feature_cols = ['Soil Moisture', 'Temperature', 'Soil Humidity', 'Time',
                'Wind speed (Km/h)', 'Air humidity (%)', 'Wind gust (Km/h)',
                'Pressure (KPa)', 'ph', 'rainfall'] + \
               [f'{col}_rolling_mean' for col in weather_cols] + \
               [f'{col}_rolling_std' for col in weather_cols]

print("🌱 Input features (after engineering):")
print(df[feature_cols])

# --- 1️⃣ Predict Irrigation Status (Random Forest) ---
X_class = df[feature_cols]
status_pred = rf_model.predict(X_class)[0]
status_label = 'ON' if status_pred == 1 else 'OFF'
print(f"\n🌿 Random Forest → Irrigation Status Prediction: {status_label} ({status_pred})")

# --- 2️⃣ Prepare input for XGBoost ---
df_xgb = df[feature_cols].copy()
df_xgb['Status'] = status_pred

# Create dummy variables for 'Status'
df_xgb = pd.get_dummies(df_xgb, columns=['Status'], drop_first=False)

# Expected features for XGBoost (must match training order)
xgb_features = feature_cols + ['Status_0', 'Status_1']

# Add missing columns
for col in xgb_features:
    if col not in df_xgb.columns:
        df_xgb[col] = 0

# Reorder columns
df_xgb = df_xgb[xgb_features]

print("\n🧩 Input to XGBoost model (after encoding):")
print(df_xgb)

# --- 3️⃣ Predict Water Requirement (XGBoost) ---
try:
    dmatrix = xgb.DMatrix(df_xgb)
    water_required = xgb_booster.predict(dmatrix)[0]
    print(f"\n💧 XGBoost → Predicted Water Requirement: {water_required:.2f} liters")
except Exception as e:
    print("\n⚠️ XGBoost prediction failed.")
    print("Error details:", e)

print("\n✅ Prediction process completed successfully.")
