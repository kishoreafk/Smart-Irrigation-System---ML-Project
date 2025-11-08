
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BASIC EDA ANALYSIS - IRRIGATION DATASET")
print("="*60)

# Load dataset
df = pd.read_csv('TARP.csv/TARP.csv')
df.columns = df.columns.str.strip()

# Basic info
print(f"\nShape: {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nStatus Distribution:\n{df['Status'].value_counts()}")

# 1. Missing Values Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_missing_values.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ Missing values heatmap saved")

# 2. Correlation Matrix
num_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 10))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_correlation_matrix.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ Correlation matrix saved")

# 3. Distribution Plots
for col in ['Soil Moisture', 'Temperature', 'rainfall', 'ph', 'N', 'P', 'K']:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col].dropna(), kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col}', fontsize=14, fontweight='bold')
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'03_dist_{col.replace(" ", "_")}.png', dpi=600, bbox_inches='tight')
    plt.close()
print("✓ Distribution plots saved")

# 4. Status Bar Chart
plt.figure(figsize=(8, 6))
status_counts = df['Status'].value_counts()
bars = plt.bar(status_counts.index, status_counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='black')
plt.title('Status Category Counts', fontsize=14, fontweight='bold')
plt.xlabel('Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('04_status_counts.png', dpi=600, bbox_inches='tight')
plt.close()
print("✓ Status bar chart saved")

# 5. Boxplots
for col in ['Soil Moisture', 'Temperature', 'rainfall', 'N', 'P', 'K']:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Status', y=col, palette=['#2ecc71', '#e74c3c'])
    plt.title(f'Boxplot: {col} by Status', fontsize=14, fontweight='bold')
    plt.xlabel('Status', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'05_box_{col.replace(" ", "_")}.png', dpi=600, bbox_inches='tight')
    plt.close()
print("✓ Boxplots saved")

print("\n" + "="*60)
print("EDA COMPLETE! All plots saved at 600 DPI.")
print("="*60)
