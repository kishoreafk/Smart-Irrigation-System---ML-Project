from flask import Flask, jsonify
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load Models
print("🔄 Loading saved models...")
try:
    rf_model = joblib.load("random_forest_model.pkl")
    print("✅ Random Forest model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load Random Forest model: {e}")

try:
    xgb_booster = xgb.Booster()
    xgb_booster.load_model("xgboost_model.json")
    print("✅ XGBoost Booster (JSON) loaded successfully!\n")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load XGBoost JSON model: {e}")

# OpenWeather API Configuration
API_KEY = os.getenv('API_KEY')
LAT = "13.08784"  # Chennai latitude
LON = "80.27847"  # Chennai longitude
WEATHER_URL = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

def get_weather_data():
    """Fetch weather data from OpenWeather API"""
    try:
        response = requests.get(WEATHER_URL)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def prepare_ml_input(weather_data):
    """Prepare input data for ML models using weather data and placeholders"""
    if not weather_data:
        return None

    # Extract available weather data
    temp = weather_data.get('main', {}).get('temp', 30)
    humidity = weather_data.get('main', {}).get('humidity', 55)
    pressure = weather_data.get('main', {}).get('pressure', 101)
    wind_speed = weather_data.get('wind', {}).get('speed', 5)
    wind_gust = weather_data.get('wind', {}).get('gust', 7)

    # Placeholder values for agricultural features not available from weather API
    soil_moisture = 25
    soil_humidity = 60
    ph = 6.8
    rainfall = 10
    time = datetime.now().hour

    data = {
        'Soil Moisture': [soil_moisture],
        'Temperature': [temp],
        'Soil Humidity': [soil_humidity],
        'Time': [time],
        'Wind speed (Km/h)': [wind_speed * 3.6],  # Convert m/s to km/h
        'Air humidity (%)': [humidity],
        'Wind gust (Km/h)': [wind_gust * 3.6],  # Convert m/s to km/h
        'Pressure (KPa)': [pressure / 10],  # Convert hPa to KPa
        'ph': [ph],
        'rainfall': [rainfall]
    }

    df = pd.DataFrame(data)

    # Feature Engineering (match enhanced training pipeline)
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

    return df

def make_predictions(df):
    """Make predictions using both models"""
    try:
        # Feature columns (match enhanced training pipeline)
        feature_cols = ['Soil Moisture', 'Temperature', 'Soil Humidity', 'Time',
                        'Wind speed (Km/h)', 'Air humidity (%)', 'Wind gust (Km/h)',
                        'Pressure (KPa)', 'ph', 'rainfall'] + \
                       [f'{col}_rolling_mean' for col in ['Soil Moisture', 'Temperature', 'Soil Humidity', 'Wind speed (Km/h)', 'Air humidity (%)', 'Wind gust (Km/h)', 'Pressure (KPa)']] + \
                       [f'{col}_rolling_std' for col in ['Soil Moisture', 'Temperature', 'Soil Humidity', 'Wind speed (Km/h)', 'Air humidity (%)', 'Wind gust (Km/h)', 'Pressure (KPa)']]

        # Random Forest prediction
        X_class = df[feature_cols]
        status_pred = rf_model.predict(X_class)[0]

        # Prepare input for XGBoost
        df_xgb = df[feature_cols].copy()
        df_xgb['Status'] = status_pred
        df_xgb = pd.get_dummies(df_xgb, columns=['Status'], drop_first=False)

        # Expected features for XGBoost
        xgb_features = feature_cols + ['Status_0', 'Status_1']

        # Add missing columns
        for col in xgb_features:
            if col not in df_xgb.columns:
                df_xgb[col] = 0

        # Reorder columns
        df_xgb = df_xgb[xgb_features]

        # XGBoost prediction
        dmatrix = xgb.DMatrix(df_xgb)
        water_required = xgb_booster.predict(dmatrix)[0]

        return {
            'irrigation_status': int(status_pred),
            'water_requirement': float(water_required)
        }
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

@app.route('/predict', methods=['GET'])
def predict():
    """API endpoint to get weather data and predictions"""
    weather_data = get_weather_data()
    if not weather_data:
        return jsonify({'error': 'Failed to fetch weather data'}), 500

    df = prepare_ml_input(weather_data)
    if df is None:
        return jsonify({'error': 'Failed to prepare input data'}), 500

    predictions = make_predictions(df)
    if predictions is None:
        return jsonify({'error': 'Failed to make predictions'}), 500

    # Prepare response
    response = {
        'weather': {
            'temperature': weather_data['main']['temp'],
            'humidity': weather_data['main']['humidity'],
            'pressure': weather_data['main']['pressure'],
            'wind_speed': weather_data['wind']['speed'],
            'wind_gust': weather_data['wind'].get('gust', 0),
            'description': weather_data['weather'][0]['description'],
            'city': weather_data['name']
        },
        'predictions': predictions,
        'input_features': df.to_dict('records')[0]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
