# Smart Irrigation Prediction System

A machine learning-powered irrigation prediction system that uses weather data and soil conditions to optimize water usage for agriculture. The system combines decision tree classification for irrigation status prediction and XGBoost regression for water requirement estimation.

## Features

- **Real-time Weather Integration**: Fetches live weather data from OpenWeather API
- **Dual ML Models**:
  - Decision Tree for irrigation ON/OFF classification
  - XGBoost for precise water requirement prediction
- **Interactive Web Dashboard**: Modern React-based frontend with 3D visualizations
- **RESTful API**: Flask-based backend API for predictions
- **Comprehensive Analytics**: Model performance metrics and feature importance analysis

## Project Structure

```
├── ml/                          # Python virtual environment
├── frontend/                    # React/Vite frontend application
├── TARP.csv/                    # Dataset directory
├── enhanced_training_pipeline.py # Model training script
├── predict.py                   # Standalone prediction script
├── app.py                       # Flask API server
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys)
├── .gitignore                   # Git ignore file
├── decision_tree_model.pkl      # Trained Decision Tree model
├── xgboost_model.json           # Trained XGBoost model (JSON format)
├── docs/                        # Documentation
└── README.md                    # This file
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 16+
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ml-project
```

### 2. Backend Setup (Python)

#### Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv ml
ml\Scripts\activate  # On Windows
# or
source ml/bin/activate  # On Linux/Mac
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Configure Environment Variables

Create a `.env` file in the project root and add your OpenWeather API key:

```bash
API_KEY=your_openweather_api_key_here
```

#### Train the Models

```bash
python enhanced_training_pipeline.py
```

This will:
- Load and preprocess the dataset
- Train Decision Tree and XGBoost models
- Generate performance plots
- Save models as `decision_tree_model.pkl` and `xgboost_model.json`

#### Test Predictions (Optional)

```bash
python predict.py
```

### 3. Frontend Setup (React)

#### Install Dependencies

```bash
cd frontend
npm install
```

#### Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

#### Build for Production

```bash
npm run build
npm run preview
```

### 4. Run the Complete Application

#### Start Backend API

```bash
# From project root (with virtual environment activated)
python app.py
```

The API will be available at `http://localhost:5000`

#### Start Frontend

```bash
# In another terminal, from frontend directory
npm run dev
```

## Model Details

### Decision Tree Classifier
- **Purpose**: Predicts whether irrigation should be ON (1) or OFF (0)
- **Features**: Soil moisture, temperature, soil humidity, time, wind conditions, etc.
- **Accuracy**: ~91%

### XGBoost Regressor
- **Purpose**: Predicts exact water requirement in liters
- **Features**: Same as Decision Tree plus irrigation status
- **Performance**: R² score ~0.85

## Configuration

### Weather API
The system uses OpenWeather API. To use your own API key:

1. Get an API key from [OpenWeather](https://openweathermap.org/api)
2. Add your API key to the `.env` file: `API_KEY=your_api_key_here`
3. Optionally change `LAT` and `LON` in `app.py` for different locations

### Model Parameters
Modify hyperparameters in `enhanced_training_pipeline.py`:
- Decision Tree: `max_depth`, `min_samples_split`
- XGBoost: `n_estimators`, `max_depth`, `learning_rate`

## Troubleshooting

### Common Issues

1. **Missing Models**: Run `python enhanced_training_pipeline.py` first
2. **Weather API Errors**: Check API key in `.env` file and internet connection
3. **Port Conflicts**: Change port in `app.py` if 5000 is occupied
4. **Virtual Environment**: Ensure virtual environment is activated

### Dataset Issues
- Ensure `TARP.csv` is in the correct location
- The dataset should contain the required feature columns
- Missing values are automatically handled during training

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenWeather API for weather data
- Scikit-learn and XGBoost for ML algorithms
- React and Three.js for frontend visualization
