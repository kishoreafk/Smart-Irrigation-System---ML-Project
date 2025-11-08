import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import csv
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Check GPU availability for XGBoost
try:
    test_model = XGBRegressor(device='cuda', n_estimators=1, verbosity=0)
    test_model.fit([[1, 2, 3]], [1])
    gpu_available = True
    print("✓ GPU available for XGBoost training")
except Exception as e:
    gpu_available = False
    print("✗ GPU not available, using CPU for XGBoost training")

print("="*60)
print("ENHANCED IRRIGATION PREDICTION MODEL TRAINING")
print("="*60)

# Load dataset
df = pd.read_csv('TARP.csv/TARP.csv')
df.columns = df.columns.str.strip()

# Basic Analysis
print("\n1. DATASET OVERVIEW")
print(f"Shape: {df.shape}")
print(f"Missing Values:\n{df.isnull().sum()}")
print(f"\nStatus Distribution:\n{df['Status'].value_counts()}")
print(f"\nBasic Statistics:\n{df.describe()}")

# Feature Engineering
print("\n2. FEATURE ENGINEERING")

# Create DayPart from Time
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

# Rolling averages for weather data (assuming data is sequential)
weather_cols = ['Soil Moisture', 'Temperature', 'Soil Humidity', 'Wind speed (Km/h)', 'Air humidity (%)', 'Wind gust (Km/h)', 'Pressure (KPa)']
for col in weather_cols:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=5, min_periods=1).mean()
    df[f'{col}_rolling_std'] = df[col].rolling(window=5, min_periods=1).std()

# Update feature columns
feature_cols = ['Soil Moisture', 'Temperature', 'Soil Humidity', 'Time',
                'Wind speed (Km/h)', 'Air humidity (%)', 'Wind gust (Km/h)',
                'Pressure (KPa)', 'ph', 'rainfall'] + \
               [f'{col}_rolling_mean' for col in weather_cols] + \
               [f'{col}_rolling_std' for col in weather_cols]

# Handle Missing Data with Iterative Imputation
print("\n3. HANDLING MISSING DATA")
print(f"Missing values before imputation:\n{df[feature_cols].isnull().sum()}")

imputer = IterativeImputer(random_state=42)
df_imputed = df.copy()
df_imputed[feature_cols] = imputer.fit_transform(df[feature_cols])

print(f"Missing values after imputation:\n{df_imputed[feature_cols].isnull().sum()}")

# Prepare data
df_clean = df_imputed[feature_cols + ['Status']].dropna(subset=['Status'])  # Only drop if Status is missing

# ============================================
# RANDOM FOREST - IRRIGATION CLASSIFICATION
# ============================================
print("\n" + "="*60)
print("4. RANDOM FOREST MODEL (Irrigation ON/OFF)")
print("="*60)

X_class = df_clean[feature_cols]
y_class = df_clean['Status'].map({'ON': 1, 'OFF': 0})

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train_c, y_train_c)
print("✓ Random Forest trained")

# Cross-validation
cv_scores = cross_validate(rf_model, X_class, y_class, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'])
print(f"\nCross-Validation Results (5-fold):")
print(f"Accuracy: {cv_scores['test_accuracy'].mean():.4f} (+/- {cv_scores['test_accuracy'].std() * 2:.4f})")
print(f"Precision: {cv_scores['test_precision'].mean():.4f} (+/- {cv_scores['test_precision'].std() * 2:.4f})")
print(f"Recall: {cv_scores['test_recall'].mean():.4f} (+/- {cv_scores['test_recall'].std() * 2:.4f})")
print(f"F1-Score: {cv_scores['test_f1'].mean():.4f} (+/- {cv_scores['test_f1'].std() * 2:.4f})")

# Predictions on test set
y_pred_c = rf_model.predict(X_test_c)
y_pred_proba = rf_model.predict_proba(X_test_c)[:, 1]

# Metrics on test set
print(f"\nTest Set Metrics:")
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test_c, y_pred_c, target_names=['OFF', 'ON'])}")

# Log results
log_data = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'RandomForestClassifier',
    'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
    'cv_accuracy_std': cv_scores['test_accuracy'].std(),
    'cv_precision_mean': cv_scores['test_precision'].mean(),
    'cv_recall_mean': cv_scores['test_recall'].mean(),
    'cv_f1_mean': cv_scores['test_f1'].mean(),
    'test_accuracy': accuracy_score(y_test_c, y_pred_c)
}

# Write to CSV
with open('model_logs.csv', 'a', newline='') as csvfile:
    fieldnames = log_data.keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if csvfile.tell() == 0:  # Write header if file is empty
        writer.writeheader()
    writer.writerow(log_data)

print("✓ Results logged to model_logs.csv")

# Plot 1: Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_c, y_pred_c)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['OFF', 'ON'], yticklabels=['OFF', 'ON'])
plt.title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('RF_01_confusion_matrix.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ Confusion Matrix saved")

# Plot 2: Feature Importance
plt.figure(figsize=(12, 10))
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]  # Top 15 features
plt.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
plt.xlabel('Importance', fontsize=12)
plt.title('Random Forest - Top 15 Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('RF_02_feature_importance.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ Feature Importance saved")

# Plot 3: ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test_c, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Random Forest - ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('RF_03_roc_curve.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ ROC Curve saved")

# ============================================
# XGBOOST - WATER PREDICTION (REGRESSION)
# ============================================
print("\n" + "="*60)
print("5. XGBOOST MODEL (Water Prediction)")
print("="*60)

# Create synthetic water requirement target based on features
df_clean['Water_Needed'] = (
    df_clean['Soil Moisture'] * -0.5 +  # Less water if soil is moist
    df_clean['Temperature'] * 2.0 +      # More water if hot
    df_clean['rainfall'] * -0.3 +        # Less water if recent rain
    100                                   # Base amount
).clip(0, 500)  # Clip to reasonable range

# Prepare features for XGBoost (include Status as dummies)
df_clean['Status'] = df_clean['Status'].map({'ON': 1, 'OFF': 0})
df_clean = pd.get_dummies(df_clean, columns=['Status'], drop_first=False)
X_reg = df_clean[feature_cols + ['Status_0', 'Status_1']]
y_reg = df_clean['Water_Needed']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train XGBoost
device = 'cuda' if gpu_available else 'cpu'
xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, device=device, verbosity=1)
print(f"Training XGBoost on {device.upper()}...")
xgb_model.fit(X_train_r, y_train_r, eval_set=[(X_test_r, y_test_r)], verbose=True)
print(f"✓ XGBoost trained on {device.upper()}")

# Predictions
y_pred_r = xgb_model.predict(X_test_r)

# Metrics
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print(f"\nR² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Plot 4: Actual vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_test_r, y_pred_r, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Water Needed', fontsize=12)
plt.ylabel('Predicted Water Needed', fontsize=12)
plt.title(f'XGBoost - Actual vs Predicted (R² = {r2:.3f})', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('XGB_01_actual_vs_predicted.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ Actual vs Predicted saved")

# Plot 5: Residual Plot
plt.figure(figsize=(10, 6))
residuals = y_test_r - y_pred_r
plt.scatter(y_pred_r, residuals, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Water Needed', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('XGBoost - Residual Plot', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('XGB_02_residual_plot.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ Residual Plot saved")

# Plot 6: Feature Importance
plt.figure(figsize=(12, 10))
importances_xgb = xgb_model.feature_importances_
xgb_features = feature_cols + ['Status_0', 'Status_1']
indices_xgb = np.argsort(importances_xgb)[::-1][:15]  # Top 15
plt.barh(range(len(indices_xgb)), importances_xgb[indices_xgb], color='coral', edgecolor='black')
plt.yticks(range(len(indices_xgb)), [xgb_features[i] for i in indices_xgb])
plt.xlabel('Importance', fontsize=12)
plt.title('XGBoost - Top 15 Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('XGB_03_feature_importance.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ Feature Importance saved")

# Plot 7: Prediction Error Distribution
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Prediction Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('XGBoost - Prediction Error Distribution', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('XGB_04_error_distribution.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ Error Distribution saved")

# Plot 8: Learning Curve (Training Progress)
plt.figure(figsize=(10, 6))
results = xgb_model.evals_result()
epochs = len(results['validation_0']['rmse'])
plt.plot(range(epochs), results['validation_0']['rmse'], label='Validation RMSE', color='coral', linewidth=2)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('XGBoost - Learning Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('XGB_05_learning_curve.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ Learning Curve saved")

# Plot 9: Model Comparison
plt.figure(figsize=(10, 6))
models = ['Random Forest\n(Classification)', 'XGBoost\n(Regression)']
scores = [cv_scores['test_accuracy'].mean(), r2]
colors = ['steelblue', 'coral']
bars = plt.bar(models, scores, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
for bar, score in zip(bars, scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{score:.3f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('COMP_01_model_comparison.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ Model Comparison saved")

# Save models
import pickle
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
xgb_model.save_model('xgboost_model.json')

print("\n" + "="*60)
print("ENHANCED TRAINING COMPLETE!")
print("="*60)
print("\nModels Saved:")
print("  - random_forest_model.pkl (Irrigation ON/OFF)")
print("  - xgboost_model.json (Water Prediction)")
print("\nResults Logged:")
print("  - model_logs.csv")
print("\nGenerated Plots:")
print("  Random Forest: 3 plots")
print("  XGBoost: 5 plots")
print("  Comparison: 1 plot")
print("\nAll images saved in high-resolution PNG format (600 DPI)!")
